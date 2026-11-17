"""Shared look for the public benchmark charts: a white spec-sheet layout, deep-violet type, our runtime in Qwen violet
(#615CED, the logo colour), every other runtime in neutral slate, Barlow / Barlow Condensed (OFL, bundled in fonts/).
The headline graphic uses the dark HERO_* palette."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
for f in os.listdir(os.path.join(HERE, "fonts")):
    if f.endswith(".ttf"):
        font_manager.fontManager.addfont(os.path.join(HERE, "fonts", f))

# palette: dark violet (same look as the headline graphic)
BG_TOP = "#2A1B5E"     # background gradient, top
BG_BOTTOM = "#0E0A24"  # background gradient, bottom
PAPER = "#170F3A"      # fallback face colour (savefig, legends)
INK = "#FFFFFF"        # titles, values
MUTED = "#B8B2E0"      # subtitles, axis labels
GRID = "#3A3270"
ON_LIGHT = "#140E33"   # text placed on light bars
OURS = "#8F88FF"       # Qwen violet, brightened for the dark background: our runtime, always
OURS_MID = "#6A62F2"   # our third series
OURS_2 = "#C9C5FF"     # our second series (e.g. the other GPU)
OTHERS = ["#7D77A8", "#5E5889", "#4A4475", "#A29DC8"]   # competitors, in order
REF = "#FFFFFF"        # reference lines (BF16, official)
BADGE = "#3B2F86"      # annotation boxes
# headline (hero) aliases
HERO_BG_TOP = BG_TOP; HERO_BG_BOTTOM = BG_BOTTOM; HERO_TILE = "#FFFFFF"; HERO_TILE_ALPHA = 0.06
HERO_TEXT = INK; HERO_MUTED = MUTED; HERO_OURS = OURS; HERO_OURS_2 = OURS_2; HERO_OTHER = "#5D5786"

DISPLAY = "Barlow Condensed"
BODY = "Barlow"

plt.rcParams.update({
    "font.family": BODY, "font.size": 11, "text.color": INK, "axes.labelcolor": MUTED,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0, "axes.facecolor": "none", "figure.facecolor": PAPER,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 11, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 10.5, "savefig.dpi": 200, "savefig.facecolor": BG_BOTTOM,
    "legend.labelcolor": INK, "axes.titlecolor": INK,
})

FOOT = "Methods and sources: public-benchmarks/SOURCES.md · others' numbers as published"


def frame(title, subtitle, figsize=(11, 6.2), source=None):
    """Figure with a left-aligned display title, a subtitle line and a footnote; returns (fig, ax)."""
    fig = plt.figure(figsize=figsize)
    background(fig)
    ax = fig.add_axes([0.075, 0.14, 0.9, 0.62])
    ax.set_facecolor("none")
    fig.text(0.075, 0.915, title, fontfamily=DISPLAY, fontsize=27, fontweight="bold", color=INK, va="baseline")
    fig.text(0.075, 0.855, subtitle, fontfamily=BODY, fontsize=12.5, color=MUTED, va="baseline")
    if source:
        fig.text(0.075, 0.052, source, fontsize=8.5, color=MUTED)
    fig.text(0.075, 0.025, FOOT, fontsize=8.5, color=MUTED)
    return fig, ax


def background(fig):
    """Dark violet vertical gradient with a soft glow top right (shared with the headline)."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    bg = fig.add_axes([0, 0, 1, 1], zorder=-10); bg.axis("off")
    bg.imshow(np.linspace(0, 1, 256).reshape(-1, 1), aspect="auto", extent=[0, 1, 0, 1], origin="upper",
              cmap=LinearSegmentedColormap.from_list("bg", [BG_TOP, BG_BOTTOM]))
    yy, xx = np.mgrid[0:1:300j, 0:1:300j]
    glow = np.exp(-(((xx - 0.85) / 0.35) ** 2 + ((yy - 0.1) / 0.3) ** 2))
    bg.imshow(glow, aspect="auto", extent=[0, 1, 0, 1], origin="upper",
              cmap=LinearSegmentedColormap.from_list("g", [(0.38, 0.36, 0.93, 0), (0.38, 0.36, 0.93, 0.45)]))
    return bg


def bar_labels(ax, bars, fmt="{:.0f}", color=INK, bold=False, pad=3, size=11):
    for b in bars:
        h = b.get_height()
        if h != h or h == 0:   # NaN / missing
            continue
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, pad), textcoords="offset points",
                    ha="center", va="bottom", fontsize=size, color=color, fontweight="bold" if bold else "normal",
                    fontfamily=DISPLAY)


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    return os.path.join(out, f"{name}.png")
