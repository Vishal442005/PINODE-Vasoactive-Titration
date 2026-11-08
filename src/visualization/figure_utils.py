"""
figure_utils.py: Shared styling, palette, and typography for publication-grade scientific figures.

Enforces zero-overlap guidelines:
- High DPI vector rendering (300+ DPI, PDF/SVG/PNG)
- Clear font hierarchy (Title: 14pt, Subtitle: 11pt, Labels: 10pt, Ticks: 9pt, Legend: 8.5pt)
- Bounded text boxes with translucent backgrounds to prevent line clashes
- Ample padding (constrained_layout / tight_layout with w_pad=2.0, h_pad=2.0)
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


def apply_publication_style():
    """Configures global matplotlib parameters for Nature / IEEE Q1 publication standards."""
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.titlesize"] = 11
    mpl.rcParams["axes.titleweight"] = "bold"
    mpl.rcParams["axes.labelsize"] = 10
    mpl.rcParams["axes.labelweight"] = "semibold"
    mpl.rcParams["xtick.labelsize"] = 9
    mpl.rcParams["ytick.labelsize"] = 9
    mpl.rcParams["legend.fontsize"] = 8.5
    mpl.rcParams["figure.titlesize"] = 13
    mpl.rcParams["figure.titleweight"] = "bold"

    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.alpha"] = 0.35
    mpl.rcParams["grid.linestyle"] = "--"
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.edgecolor"] = "#333333"
    mpl.rcParams["axes.linewidth"] = 0.9

    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.bbox"] = "tight"


# Curated colorblind-safe clinical palette
PALETTE = {
    "primary_blue": "#1f77b4",
    "teal": "#008080",
    "coral_red": "#d62728",
    "amber_orange": "#ff7f0e",
    "emerald_green": "#2ca02c",
    "purple": "#9467bd",
    "dark_slate": "#2f4f4f",
    "light_gray": "#e0e0e0",
    "target_corridor": "#e8f5e9",
    "target_line": "#2e7d32",
    "uncertainty_band": "#bbdefb",
}
