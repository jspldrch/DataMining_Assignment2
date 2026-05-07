import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'

fig, ax = plt.subplots(figsize=(13, 14))
ax.axis('off')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.patch.set_facecolor('white')

def section(y, text):
    ax.add_patch(plt.Rectangle((0.02, y - 0.02), 0.96, 0.045,
                               transform=ax.transAxes,
                               color='#e8f0fe', zorder=0, clip_on=False))
    ax.text(0.05, y + 0.002, text, transform=ax.transAxes,
            ha='left', va='center', fontsize=12, color='#1a237e', fontweight='bold')

def formula(y, label, expr):
    ax.text(0.05, y + 0.030, label, transform=ax.transAxes,
            ha='left', va='center', fontsize=9, color='#555555', style='italic')
    ax.text(0.5, y - 0.010, expr, transform=ax.transAxes,
            ha='center', va='center', fontsize=14, color='#1a1a2e')
    ax.add_line(Line2D([0.04, 0.96], [y - 0.052, y - 0.052],
                       transform=ax.transAxes, color='#dddddd', linewidth=0.8))

# Title
ax.text(0.5, 0.97,
        r'$\mathbf{Two{-}Phase\ AR{-}Enhanced\ K{-}Means\ -\ Key\ Formulas}$',
        transform=ax.transAxes, ha='center', va='center', fontsize=15, color='#1a1a2e')

# Phase 1
section(0.91, 'Phase 1:  AR Feature Weighting')

formula(0.855,
        'Step 3 — Rule confidence:',
        r'$\mathrm{conf}(A \to B)\ =\ \dfrac{\mathrm{support}(A \cup B)}{\mathrm{support}(A)}\ \geq\ 0.5$')

formula(0.760,
        'Step 4 — Predictability of feature j   ( R_j = all rules with j as consequent ):',
        r'$p_j\ =\ \dfrac{1}{|R_j|}\ \sum_{r\,\in\,R_j} \mathrm{conf}(r)'
        r'\qquad \left(p_j = 0 \ \ \mathrm{if} \ \ |R_j|=0\right)$')

formula(0.670,
        'Step 5a — Raw weight:',
        r'$\tilde{w}_j\ =\ \mathrm{clip}\!\left(1 - p_j,\ \ 0.1,\ \ 1.0\right)$')

formula(0.585,
        'Step 5b — Normalised weight  (so that  mean(w) = 1,   d = 20 ):',
        r'$w_j\ =\ \dfrac{\tilde{w}_j}{\dfrac{1}{d}\,\sum_{k=1}^{d}\tilde{w}_k}$')

formula(0.490,
        'Step 6 — Weighted feature matrix:',
        r'$\widetilde{X}\ =\ X_{\mathrm{scaled}}\ \odot\ \mathbf{w}$')

formula(0.415,
        'Resulting weighted Euclidean distance in K-Means:',
        r'$\|\,\tilde{x}_i - \tilde{\mu}\,\|^2\ =\ \sum_{j=1}^{d}\ w_j^2\,(x_{ij} - \mu_j)^2$')

# Phase 2
section(0.330, 'Phase 2:  AR-Guided Centroid Initialisation')

formula(0.268,
        'Step 2 — Proportion score for sample i at quartile level q:',
        r'$\mathrm{prop}_q(i)\ =\ \dfrac{1}{d}\ \sum_{j=1}^{d}\ '
        r'\mathbf{1}\!\left[\,\mathrm{quartile\_level}(x_{ij}) = q\,\right]$')

formula(0.160,
        'Step 3 — Initial centroid for cluster q   ( A_q = top-50 archetypes at level q ):',
        r'$\mu_q^{(0)}\ =\ \dfrac{1}{|\mathcal{A}_q|}\ \sum_{i\,\in\,\mathcal{A}_q}\ \tilde{x}_i$')

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig('q5_formulas.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: q5_formulas.png")
plt.show()
