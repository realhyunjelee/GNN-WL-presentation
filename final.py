import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
import numpy as np


# ============================================================
# 1. Build graphs
# ============================================================

def build_path6_with_leaf():
    """Path on 6 vertices with one extra leaf attached to node 2"""
    n = 7
    A = torch.zeros(n, n)
    edges = [
        (0, 1), (1, 2), (2, 3),
        (3, 4), (4, 5),
        (2, 6)   # extra leaf
    ]
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


def build_cycle_6():
    """6-cycle: 0-1-2-3-4-5-0"""
    n = 6
    A = torch.zeros(n, n)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


def build_two_triangles():
    """Two disjoint triangles: 0-1-2-0 and 3-4-5-3"""
    n = 6
    A = torch.zeros(n, n)
    edges = [
        (0, 1), (1, 2), (2, 0),
        (3, 4), (4, 5), (5, 3),
    ]
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


# ============================================================
# 2. 1-WL implementation
# ============================================================

def wl1_colour_refinement(A, max_iters=10):
    """
    1-dimensional WL / colour refinement on adjacency matrix A.
    Returns: list of tensors, one per iteration, shape (n,)
    """
    n = A.shape[0]
    
    # t = 0: initial colours = degree
    degrees = A.sum(dim=1).long()
    col = degrees.clone()
    hist = [col.clone()]
    
    for _ in range(max_iters):
        sigs = []
        for v in range(n):
            neigh = (A[v] == 1).nonzero().flatten().tolist()
            neigh_cols = sorted(int(col[u]) for u in neigh)
            sigs.append((int(col[v]), tuple(neigh_cols)))
        
        uniq = {s: i for i, s in enumerate(sorted(set(sigs)))}
        col = torch.tensor([uniq[s] for s in sigs], dtype=torch.long)
        hist.append(col.clone())
        
        # Check stability
        if torch.equal(hist[-1], hist[-2]):
            break
    
    return hist


# ============================================================
# 3. GNN model
# ============================================================

class SimpleGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim, bias=True)
        self.W_neigh = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, A, h):
        neigh_agg = torch.matmul(A, h)
        out = self.W_self(h) + self.W_neigh(neigh_agg)
        return F.relu(out)


class RecurrentGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer = SimpleGNNLayer(in_dim, hidden_dim)
        # Subsequent layers
        self.layers = nn.ModuleList([
            SimpleGNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])

    def forward(self, A, h0):
        """Returns list of embeddings at each step, all with shape (n, hidden_dim)"""
        history = []
        h = h0
        
        # First layer
        h = self.layer(A, h)
        history.append(h.clone())
        
        # Subsequent layers
        for layer in self.layers:
            h = layer(A, h)
            history.append(h.clone())
        
        return history


# ============================================================
# 4. Training
# ============================================================

def train_gnn(A, wl_hist, num_epochs=300, lr=0.01):
    """Train GNN to predict WL colours at each iteration"""
    n = A.shape[0]
    num_layers = len(wl_hist)
    
    # Initial features: one-hot degree
    deg = A.sum(1).long()
    max_deg = int(deg.max().item()) + 1
    h0 = F.one_hot(deg, num_classes=max_deg).float()
    
    # Get output dimension (max number of colors across all iterations)
    out_dim = max(int(wl.max().item()) + 1 for wl in wl_hist)
    
    # Model
    hidden_dim = 16
    gnn = RecurrentGNN(in_dim=h0.shape[1], hidden_dim=hidden_dim, num_layers=num_layers)
    clf = nn.Linear(hidden_dim, out_dim)
    
    opt = torch.optim.Adam(
        list(gnn.parameters()) + list(clf.parameters()), 
        lr=lr
    )
    
    # Training loop
    for epoch in range(num_epochs):
        opt.zero_grad()
        
        gnn_hist = gnn(A, h0)
        
        loss = 0
        for t in range(len(wl_hist)):
            if t < len(gnn_hist):
                logits = clf(gnn_hist[t])
                labels = wl_hist[t]
                loss += F.cross_entropy(logits, labels)
        
        loss /= len(wl_hist)
        loss.backward()
        opt.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}")
    
    return gnn, clf, h0


# ============================================================
# 5. Visualization
# ============================================================

def get_graph_positions(A, graph_type="path"):
    """Get 2D positions for graph visualization"""
    n = A.shape[0]
    
    if graph_type == "path":
        # Path with leaf
        coords = {
            0: (0, 0),
            1: (1, 0),
            2: (2, 0),
            3: (3, 0),
            4: (4, 0),
            5: (5, 0),
            6: (2, -1),
        }
    elif graph_type == "cycle":
        # Circle layout
        coords = {}
        for i in range(n):
            angle = 2 * np.pi * i / n
            coords[i] = (np.cos(angle), np.sin(angle))
    else:
        # Fallback
        coords = {i: (i % 3, i // 3) for i in range(n)}
    
    return coords


def draw_graph_with_colors(ax, A, colors, coords, title=""):
    """Draw graph with node colors"""
    n = A.shape[0]
    
    # Set background
    ax.set_facecolor("#f8f9fa")
    
    # Draw edges first (behind nodes)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 1:
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                ax.plot([x1, x2], [y1, y2], color="#d0d0d0", linewidth=3, zorder=1, solid_capstyle="round")
    
    # Draw nodes
    x_pos = [coords[i][0] for i in range(n)]
    y_pos = [coords[i][1] for i in range(n)]
    
    color_values = colors.cpu().numpy() if isinstance(colors, torch.Tensor) else colors
    
    scatter = ax.scatter(
        x_pos, y_pos,
        c=color_values,
        cmap="tab20",
        s=600,
        edgecolors="white",
        linewidth=3,
        zorder=3,
        alpha=0.95
    )
    
    # Label nodes with better styling
    for i in range(n):
        ax.text(coords[i][0], coords[i][1], str(i), 
                ha="center", va="center", fontweight="bold", fontsize=11, 
                color="white", zorder=4, family="monospace")
    
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15, color="#2c3e50")
    ax.set_aspect("equal")
    ax.set_xlim(-1.4, 6.4) if max(coords.keys()) > 5 else ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.4, 0.5) if max(coords.keys()) > 5 else ax.set_ylim(-1.3, 1.3)
    ax.axis("off")
    
    return scatter


def visualize_wl_vs_gnn(A, wl_hist, gnn, clf, h0, graph_type="path"):
    """Visualize WL colors and GNN predictions side-by-side over iterations"""
    n = A.shape[0]
    T = len(wl_hist)
    coords = get_graph_positions(A, graph_type)
    
    # Get GNN predictions at each layer
    with torch.no_grad():
        gnn_hist = gnn(A, h0)
        gnn_preds = []
        for t in range(len(gnn_hist)):
            logits = clf(gnn_hist[t])
            pred = torch.argmax(logits, dim=1)
            gnn_preds.append(pred)
    
    # Create subplots: T rows, 2 columns (WL vs GNN)
    fig, axes = plt.subplots(T, 2, figsize=(14, 5.5*T))
    fig.patch.set_facecolor("white")
    
    if T == 1:
        axes = axes.reshape(1, -1)
    
    for t in range(T):
        # WL colors
        draw_graph_with_colors(
            axes[t, 0], A, wl_hist[t], coords,
            title=f"1-WL iteration t={t}"
        )
        
        # GNN predictions
        if t < len(gnn_preds):
            draw_graph_with_colors(
                axes[t, 1], A, gnn_preds[t], coords,
                title=f"GNN layer t={t}"
            )
    
    plt.tight_layout()
    return fig


def compare_datasets(A_success, A_fail):
    """Compare WL behavior on two graphs (success vs failure case)"""
    wl_success = wl1_colour_refinement(A_success, max_iters=10)
    wl_fail = wl1_colour_refinement(A_fail, max_iters=10)
    
    # Final multisets
    multiset_success = sorted(wl_success[-1].tolist())
    multiset_fail = sorted(wl_fail[-1].tolist())
    
    print("\n" + "="*60)
    print("WL COMPARISON")
    print("="*60)
    print(f"Graph 1 (6-cycle):")
    print(f"  Final WL multiset: {multiset_success}")
    print(f"\nGraph 2 (two triangles):")
    print(f"  Final WL multiset:  {multiset_fail}")
    
    if multiset_success == multiset_fail:
        print(f"\n=> 1-WL CANNOT distinguish these graphs!")
    else:
        print(f"\n=> 1-WL CAN distinguish these graphs.")
    print("="*60 + "\n")
    
    return wl_success, wl_fail


# ============================================================
# 6. Main
# ============================================================

def main():
    print("\n" + "="*60)
    print("GNN vs 1-WL: Training and Visualization")
    print("="*60)
    
    # Choose dataset
    print("\nSelect dataset:")
    print("1. Path with leaf (WL distinguishes nodes)")
    print("2. Cycle vs Two Triangles (WL FAILS to distinguish)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        print("\n>>> Using FAILURE case: Cycle vs Two Triangles")
        A1 = build_cycle_6()
        A2 = build_two_triangles()
        print("\nGraph 1: C6 (6-cycle)")
        print("Graph 2: Two disjoint triangles")
        
        wl1, wl2 = compare_datasets(A1, A2)
        
        # Train on both graphs
        for idx, (A, name) in enumerate([(A1, "C6"), (A2, "Two Triangles")]):
            print(f"\n{'-'*60}")
            print(f"Training GNN on {name}")
            print(f"{'-'*60}")
            gnn, clf, h0 = train_gnn(A, wl1 if idx == 0 else wl2, num_epochs=300)
            fig = visualize_wl_vs_gnn(A, wl1 if idx == 0 else wl2, gnn, clf, h0, "cycle")
            fig.suptitle(f"GNN vs 1-WL on {name}", fontsize=14, fontweight="bold", y=0.995)
            plt.show()
    else:
        print("\n>>> Using SUCCESS case: Path with leaf")
        A = build_path6_with_leaf()
        print("Graph: Path 0-1-2-3-4-5 with leaf node 6 attached to node 2")
        
        wl_hist = wl1_colour_refinement(A, max_iters=10)
        
        print(f"\n{'-'*60}")
        print(f"Training GNN")
        print(f"{'-'*60}")
        gnn, clf, h0 = train_gnn(A, wl_hist, num_epochs=300)
        
        fig = visualize_wl_vs_gnn(A, wl_hist, gnn, clf, h0, "path")
        fig.suptitle("GNN vs 1-WL on Path Graph", fontsize=14, fontweight="bold", y=0.995)
        plt.show()


if __name__ == "__main__":
    main()