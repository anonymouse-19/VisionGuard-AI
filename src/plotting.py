"""One matplotlib style for every figure (palette validated with the dataviz validator)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in fixed order; class colours follow the entity, never its rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CLASS_COLORS = dict(zip(["crack", "corrosion", "oil_leak", "surface_wear", "damage"], SERIES))
# Status palette is reserved for maintenance actions.
STATUS = {"Routine Monitoring": "#0ca30c", "Schedule Maintenance": "#fab219",
          "Urgent Intervention": "#ec835a", "Critical/Withdraw": "#d03b3b"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "lines.linewidth": 2.0, "lines.markersize": 5, "font.size": 9,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES), "legend.frameon": False,
        "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def savefig(fig, path):
    from src.common import rpath
    path = rpath(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


apply_style()
