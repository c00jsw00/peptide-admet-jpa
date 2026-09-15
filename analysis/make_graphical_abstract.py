# Graphical abstract for JPA submission
# Layout: left = protocol flow, right = two endpoint bar panels (PAMPA / Caco-2), bottom = key message
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig = plt.figure(figsize=(12, 6.6), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis("off")

# palette
C_BASE   = "#8fa3bf"   # muted blue  (baseline)
C_TAB    = "#4e8f5b"   # green  (TabPFN)
C_KPGT   = "#c2571d"   # orange (KPGT)
C_CEIL   = "#333333"   # black  (ceiling)
C_BOX    = "#eef2f7"
C_EDGE   = "#5b708c"
C_MSG    = "#f5efe6"

def box(x, y, w, h, text, fc=C_BOX, ec=C_EDGE, fs=9, bold_first=None, lw=1.2, tc="#222"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                       fc=fc, ec=ec, lw=lw, mutation_aspect=1)
    ax.add_patch(b)
    if bold_first:
        head, rest = text.split("||")
        ax.text(x + w/2, y + h*0.66, head, ha="center", va="center", fontsize=fs+0.5,
                fontweight="bold", color=tc)
        ax.text(x + w/2, y + h*0.28, rest, ha="center", va="center", fontsize=fs-0.8, color="#444")
    else:
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc)

def arrow(x1, y1, x2, y2, color=C_EDGE):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                        color=color, lw=1.6)
    ax.add_patch(a)

# ---------------- title ----------------
ax.text(50, 63.5, "Censored-floor-aware benchmark of cyclic-peptide permeability prediction",
        ha="center", va="center", fontsize=12.5, fontweight="bold", color="#1a2433")

# ---------------- left: protocol flow ----------------
ax.text(14.5, 58.5, "v4.2 protocol", ha="center", fontsize=10.5, fontweight="bold", color=C_EDGE)
box(3, 49.5, 23, 7, "pepADMET data ||7,283 PAMPA  /  7,429 Caco-2\nunique-SMILES 70/10/20, seed 42")
arrow(14.5, 49.5, 14.5, 45.6)
box(3, 38.5, 23, 7, "leakage-controlled split ||no stereo/tautomer\ncross-contamination")
arrow(14.5, 38.5, 14.5, 34.6)
box(3, 27.5, 23, 7, "left-censored floor ||PAMPA −10.0 (3.7 %)\nCaco-2 3.3 %  →  oracle ceilings")
arrow(14.5, 27.5, 14.5, 23.6)
box(3, 14.5, 23, 9, "ten routes ||8 classical + foundation models\n(TabPFN v2, KPGT fine-tune)\n→ Caco-2 extension (route 10)",
    fc="#dfe9f5", ec=C_EDGE)

# ---------------- right: endpoint panels ----------------
def panel(x0, title, bars, ceiling, note):
    # bars: list of (label, value, color)
    ax.text(x0 + 19, 58.5, title, ha="center", fontsize=10.5, fontweight="bold", color=C_EDGE)
    base_y, maxv, unit_h = 14.5, 0.70, 34.0
    # axis
    ax.plot([x0+3, x0+35], [base_y, base_y], color="#999", lw=1.0)
    bw, gap = 5.4, 3.4
    x = x0 + 4.5
    for label, val, col in bars:
        h = (val / maxv) * unit_h
        ax.add_patch(plt.Rectangle((x, base_y), bw, h, fc=col, ec="none"))
        ax.text(x + bw/2, base_y + h + 0.9, f"{val:.3f}", ha="center", fontsize=8.5,
                fontweight="bold", color=col)
        ax.text(x + bw/2, base_y - 2.2, label, ha="center", fontsize=7.6, color="#333")
        x += bw + gap
    # ceiling line
    cy = base_y + (ceiling / maxv) * unit_h
    ax.plot([x0+3, x0+35], [cy, cy], color=C_CEIL, lw=1.6, ls=(0, (5, 3)))
    ax.text(x0 + 34.5, cy + 1.0, f"ceiling {ceiling:.3f}", ha="right", fontsize=8,
            style="italic", color=C_CEIL)
    ax.text(x0 + 19, base_y - 5.2, note, ha="center", fontsize=8, color="#555")

panel(36, "PAMPA — R$^2$",
      [("baseline", 0.464, C_BASE), ("TabPFN v2", 0.496, C_TAB), ("KPGT ft", 0.513, C_KPGT)],
      0.539, "KPGT: 95 % of ceiling  ·  gains sit in the censored subset")
panel(70, "Caco-2 — R$^2$",
      [("baseline", 0.393, C_BASE), ("TabPFN v2", 0.442, C_TAB), ("KPGT ft", 0.411, C_KPGT)],
      0.570, "ranking reverses  ·  TabPFN +0.048, KPGT +0.018")

# arrows from routes box to panels
arrow(26.3, 22.5, 35.4, 30, color="#aaa")
arrow(26.3, 17.5, 69.4, 22, color="#aaa")

# ---------------- key message ----------------
b = FancyBboxPatch((3, 2.2), 94, 8.2, boxstyle="round,pad=0.7,rounding_size=1.4",
                   fc=C_MSG, ec="#b09a6d", lw=1.4, mutation_aspect=1)
ax.add_patch(b)
ax.text(50, 7.9, "Foundation models are the first to beat the baselines on both endpoints — but the censored ceilings",
        ha="center", va="center", fontsize=10, color="#222")
ax.text(50, 4.6, "(0.539 PAMPA, 0.570 Caco-2), not model capacity, are the true bottleneck.",
        ha="center", va="center", fontsize=10, fontweight="bold", color="#8a5a10")

out = "graphical_abstract.png"
fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
print("saved", out)
