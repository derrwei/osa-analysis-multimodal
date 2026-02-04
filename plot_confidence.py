import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# -------- helpers --------
def _segments(labels):
    labels = np.asarray(labels, dtype=int)
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    ends = np.r_[starts[1:], labels.size]
    return list(zip(starts, ends, labels[starts]))

def _plot_band(ax, t, labels, color_norm="#38a169", color_osa="#2b6cb0", title=""):
    for s, e, v in _segments(labels):
        ax.axvspan(t[s], t[e-1], ymin=0.05, ymax=0.95,
                   color=(color_osa if v==1 else color_norm), lw=0)
    ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_xlim(t[0], t[-1])
    ax.set_title(title, loc="left", fontsize=10)

def _majority(v):
    # majority for 0/1 labels; ties -> 0 (NORMAL)
    return int(v.mean() > 0.5)

# -------- main --------
def plot_binary_osa_timeline(
    t,                  # (T,) time (seconds/minutes or datetime64)
    y_true,             # (T,) 0=NORMAL, 1=OSA
    p_osa,              # (T,) predicted probability of OSA
    subject_title=None,
    bins=600,           # ↓ reduce to this many coarse time steps for a broad view
    thresh=0.5,         # decision threshold for OSA
    colors=("#38a169", "#2b6cb0"),  # NORMAL, OSA
    figsize=(12, 4)
):
    t = np.asarray(t); y_true = np.asarray(y_true).astype(int); p_osa = np.asarray(p_osa)
    assert t.ndim==y_true.ndim==p_osa.ndim==1 and t.size==y_true.size==p_osa.size

    # ----- bin to fewer steps for a broad view -----
    # make bin edges across the full duration
    idx = np.linspace(0, t.size, bins+1).astype(int)
    # ensure unique increasing indices
    idx = np.unique(np.clip(idx, 0, t.size))
    # aggregate per bin
    t_bin, y_true_bin, p_bin = [], [], []
    for a, b in zip(idx[:-1], idx[1:]):
        if b <= a: 
            continue
        t_bin.append(t[(a+b)//2])                 # center timestamp
        y_true_bin.append(_majority(y_true[a:b])) # majority label
        p_bin.append(p_osa[a:b].mean())           # mean prob
    t_bin = np.asarray(t_bin)
    y_true_bin = np.asarray(y_true_bin)
    p_bin = np.asarray(p_bin)
    y_pred_bin = (p_bin >= thresh).astype(int)

    # ----- plot -----
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize,
                                        gridspec_kw={"height_ratios":[1,1,1.4]},
                                        sharex=True)
    _plot_band(ax1, t_bin, y_true_bin, colors[0], colors[1], "Observed (PSG)")
    _plot_band(ax2, t_bin, y_pred_bin, colors[0], colors[1], "Estimated")

    # Confidence (stacked NORMAL / OSA)
    ax3.stackplot(t_bin, 1.0 - p_bin, p_bin, labels=["NORMAL", "OSA"],
                  colors=[colors[0], colors[1]], linewidth=0.0)
    ax3.set_ylim(0, 1); ax3.set_title("Confidence", loc="left", fontsize=10)
    ax3.set_xlabel("Time")

    handles = [plt.Line2D([0],[0], lw=8, color=c) for c in colors]
    fig.legend(handles, ["NORMAL", "OSA"], loc="lower center", ncol=2, frameon=True)
    if subject_title: fig.suptitle(subject_title, y=1.02, fontsize=11)
    plt.tight_layout(); 
    plt.savefig("osa_timeline.png", dpi=300)


# -------------------------
# Example usage (dummy data)
if __name__ == "__main__":
    T = 2000
    t = np.linspace(0, 8*60, T)  # 8 hours in minutes (for example)
    # fake probabilities for 3 classes
    rng = np.random.default_rng(42)
    raw = rng.random((2, T))
    proba = raw / raw.sum(axis=0, keepdims=True)
    y_true = rng.integers(0, 2, size=T)

    plot_binary_osa_timeline(t, y_true, proba[1], subject_title="Subject XYZ")