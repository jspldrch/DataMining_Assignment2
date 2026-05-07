import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── colour palette ────────────────────────────────────────────────────────────
C_INPUT  = '#dae8fc'
C_PH1    = '#d5e8d4'
C_PH2    = '#ffe6cc'
C_KMEANS = '#e1d5e7'
C_OUT    = '#f8cecc'
E_INPUT  = '#4472c4'
E_PH1    = '#6aab5b'
E_PH2    = '#d6a020'
E_KMEANS = '#7b5ea7'
E_OUT    = '#c0392b'

fig = plt.figure(figsize=(17, 12))
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 17)
ax.set_ylim(0, 12)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── helpers ───────────────────────────────────────────────────────────────────
def box(cx, cy, w, h, lines, fc, ec, fs=9.0):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.14',
        facecolor=fc, edgecolor=ec, linewidth=1.8, zorder=3))
    ax.text(cx, cy, '\n'.join(lines),
            ha='center', va='center', fontsize=fs,
            fontweight='bold', linespacing=1.55, zorder=4)

def arr(x1, y1, x2, y2, rad=0, col='#444444'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->', lw=2.0, color=col,
                    connectionstyle=f'arc3,rad={rad}'),
                zorder=5)

def step_badge(x, y, label, fc, ec):
    ax.text(x, y, label, ha='center', va='center', fontsize=8,
            fontweight='bold', color=ec, zorder=6,
            bbox=dict(boxstyle='circle,pad=0.18', fc=fc, ec=ec, lw=1.4))

# ── phase labels ──────────────────────────────────────────────────────────────
ax.text(4.0, 9.0, 'PHASE 1 — AR Feature Weighting',
        ha='center', fontsize=11.5, fontweight='bold', color='#2a7a2a', zorder=6)
ax.text(13.0, 9.0, 'PHASE 2 — Centroid Initialisation',
        ha='center', fontsize=11.5, fontweight='bold', color='#a07000', zorder=6)

# ── title ─────────────────────────────────────────────────────────────────────
ax.text(8.5, 11.75,
        'Two-Phase AR-Enhanced K-Means  —  Framework Diagram',
        ha='center', fontsize=15, fontweight='bold', color='#1a1a2e')

# ── INPUT ─────────────────────────────────────────────────────────────────────
box(8.5, 11.1, 5.5, 0.72,
    ['mobile_price.csv   (2000 samples  ×  20 features,  4 classes)'],
    C_INPUT, E_INPUT, fs=9.5)
arr(8.5, 10.74, 8.5, 10.24)
box(8.5, 9.9, 5.5, 0.62,
    ['Z-score Standardisation  →  X_scaled'],
    C_INPUT, E_INPUT, fs=9.5)

# branch to phases
arr(5.75, 9.9, 4.0, 8.67, col=E_PH1)
arr(11.25, 9.9, 13.0, 8.67, col=E_PH2)

# ── PHASE 1 steps (x = 4.0) ──────────────────────────────────────────────────
BOX_W = 6.8
BOX_H = 0.74

ph1 = [
    (8.30, ['Quintile Discretisation',
            '5 equal-frequency bins per feature   (pd.qcut, q=5)']),
    (7.30, ['FP-Growth   (min_support = 0.05)',
            'Mine all frequent itemsets from binary transaction matrix']),
    (6.30, ['Association Rules   (confidence ≥ 0.5)',
            'Extract rules  A → B  from frequent itemsets']),
    (5.30, ['Predictability Score   pⱼ',
            'pⱼ = mean confidence of rules where feature j is consequent']),
    (4.30, ['Feature Weights   wⱼ = clip(1−pⱼ,  0.1,  1.0),   mean(w)=1',
            'X̃ = X_scaled  ⊙  w']),
]

prev_y = None
for i, (y, lines) in enumerate(ph1):
    box(4.0, y, BOX_W, BOX_H, lines, C_PH1, E_PH1, fs=8.6)
    step_badge(0.72, y, f'S{i+1}', C_PH1, E_PH1)
    if prev_y:
        arr(4.0, prev_y - BOX_H/2, 4.0, y + BOX_H/2)
    prev_y = y

# ── PHASE 2 steps (x = 13.0) ─────────────────────────────────────────────────
ph2 = [
    (8.30, ['Quartile Level Assignment',
            'Assign each value to level  q ∈ {0, 1, 2, 3}  per feature']),
    (7.20, ['Archetype Sample Selection',
            'top-50 samples with highest prop. of features at level q']),
    (6.10, ['Initial Centroids   μᴄ⁻⁰',
            'Mean of archetype set in X̃ space  →  seed for cluster q']),
]

prev_y = None
for i, (y, lines) in enumerate(ph2):
    box(13.0, y, BOX_W, BOX_H, lines, C_PH2, E_PH2, fs=8.6)
    step_badge(16.28, y, f'S{i+1}', C_PH2, E_PH2)
    if prev_y:
        arr(13.0, prev_y - BOX_H/2, 13.0, y + BOX_H/2)
    prev_y = y

# ── merge arrows → K-Means ────────────────────────────────────────────────────
# Phase 1: bottom-center of S5 box → K-Means top-left
arr(4.0, 3.93, 7.2, 2.86, rad=-0.2, col=E_PH1)
# Phase 2: bottom-center of S3 box → K-Means top-right
arr(13.0, 5.73, 9.8, 2.86, rad=0.2, col=E_PH2)

# ── K-MEANS ───────────────────────────────────────────────────────────────────
box(8.5, 2.5, 6.0, 0.72,
    ['K-Means Clustering   (n_clusters = 4)',
     'init = μ⁻⁰,   n_init = 1,   random_state = seed'],
    C_KMEANS, E_KMEANS, fs=9.5)
arr(8.5, 2.14, 8.5, 1.6)

# ── OUTPUT ────────────────────────────────────────────────────────────────────
box(8.5, 1.25, 8.0, 0.62,
    ['Hungarian Label Mapping  →  Accuracy  /  Precision  /  Recall  /  F1-score'],
    C_OUT, E_OUT, fs=9.5)

# ── legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(fc=C_INPUT,  ec=E_INPUT,  label='Input / Preprocessing'),
    mpatches.Patch(fc=C_PH1,    ec=E_PH1,    label='Phase 1: AR Feature Weighting'),
    mpatches.Patch(fc=C_PH2,    ec=E_PH2,    label='Phase 2: Centroid Initialisation'),
    mpatches.Patch(fc=C_KMEANS, ec=E_KMEANS, label='K-Means Clustering'),
    mpatches.Patch(fc=C_OUT,    ec=E_OUT,    label='Evaluation / Metrics'),
]
ax.legend(handles=legend_handles, loc='lower left',
          fontsize=9.5, framealpha=0.95,
          bbox_to_anchor=(0.01, 0.01))

plt.savefig('q5_framework.png', dpi=300, bbox_inches='tight', facecolor='white')
print('Saved: q5_framework.png')
plt.show()
