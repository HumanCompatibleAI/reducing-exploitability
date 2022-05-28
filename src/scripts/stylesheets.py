"""matplotlib styles."""
# Based on https://github.com/HumanCompatibleAI/evaluating-rewards/blob/master/src/evaluating_rewards/analysis/stylesheets.py

import contextlib
import os
from typing import Iterable, Iterator

LATEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latex")

STYLES = {
    # Matching ICML 2022 style
    "paper": {
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "mathtext.fontset": "cm",
        # Increase the font siyes, so that when we scale the figues down, the
        # font size is about 9pt
        "font.size": 12,
        "legend.fontsize": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "lines.linewidth": 1.5,
    },
    "huge": {"figure.figsize": (20, 10)},
    "pointmass-2col": {
        "figure.figsize": (5.5, 2.04),
        "figure.subplot.left": 0.2,
        "figure.subplot.right": 1.0,
        "figure.subplot.top": 0.92,
        "figure.subplot.bottom": 0.16,
        "figure.subplot.hspace": 0.2,
        "figure.subplot.wspace": 0.25,
    },
    "heatmap": {},
    "heatmap-1col": {
        "figure.figsize": (5.5, 3.9),
        "figure.subplot.top": 0.99,
        "figure.subplot.right": 0.92,
        "figure.subplot.left": 0.08,
        "figure.subplot.bottom": 0.09,
    },
    "heatmap-1col-fatlabels": {
        "figure.subplot.right": 0.91,
        "figure.subplot.left": 0.145,
        "figure.subplot.bottom": 0.2,
    },
    "heatmap-2col": {
        "figure.figsize": (2.7, 2.025),
        "figure.subplot.top": 0.99,
        "figure.subplot.bottom": 0.16,
        "figure.subplot.left": 0.17,
        "figure.subplot.right": 0.91,
    },
    "heatmap-3col": {
        "figure.subplot.top": 0.96,
        "figure.subplot.bottom": 0.3,
        "figure.subplot.left": 0.00,
        "figure.subplot.right": 1.00,
    },
    "heatmap-3col-left": {
        # includes y-axis labels
        "figure.figsize": (1.90, 1.43),
        "figure.subplot.left": 0.26,
    },
    "heatmap-3col-middle": {"figure.figsize": (1.40, 1.43)},
    "heatmap-3col-right": {
        # includes colorbar
        "figure.figsize": (2.10, 1.43),
        "figure.subplot.right": 0.84,
    },
    "training-curve": {
        "legend.columnspacing": 1.0,
        "legend.handletextpad": 0.4,
        "legend.handleheight": 0.1,
        "legend.borderpad": 0.4,
        "legend.borderaxespad": 0.1,
        "legend.labelspacing": 0.3,
    },
    "training-curve-1col": {
        "figure.figsize": (3.5, 2.6),
        "figure.subplot.top": 0.93,
        "figure.subplot.right": 0.93,
        "figure.subplot.left": 0.21,
        "figure.subplot.bottom": 0.21,
    },
    "training-curve-2col": {
        "figure.figsize": (3.5, 2.6),
        "figure.subplot.top": 0.93,
        "figure.subplot.right": 0.93,
        "figure.subplot.left": 0.17,
        "figure.subplot.bottom": 0.21,
    },
    "barplot-2col": {
        "figure.figsize": (3.5, 2.6),
        "figure.subplot.top": 0.93,
        "figure.subplot.right": 0.93,
        "figure.subplot.left": 0.21,
        "figure.subplot.bottom": 0.26,
    },
    "barplot-2col-bigger": {
        "figure.figsize": (3.5, 2.6),
        "figure.subplot.top": 0.93,
        "figure.subplot.right": 0.93,
        "figure.subplot.left": 0.14,
        "figure.subplot.bottom": 0.25,
    },
    "training-curve-1col-tall-legend": {
        "figure.figsize": (7, 2.6),
        "figure.subplot.top": 0.93,
        "figure.subplot.right": 0.8,
        "figure.subplot.bottom": 0.20,
        "figure.subplot.left": 0.1,
    },
    "small-labels": {"xtick.labelsize": 8, "ytick.labelsize": 8},
    "tiny-font": {"font.size": 6, "xtick.labelsize": 6, "ytick.labelsize": 6},
    "gridworld-heatmap": {
        "axes.facecolor": "lightgray",
        "image.cmap": "RdBu",
        "hatch.linewidth": 0.1,
    },
    "gridworld-heatmap-2in1": {
        "figure.figsize": (5.5, 2.77),
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.subplot.left": 0.03,
        "figure.subplot.right": 0.96,
        "figure.subplot.top": 0.97,
        "figure.subplot.bottom": 0.03,
        "figure.subplot.wspace": 0.1,
        "figure.subplot.hspace": 0.14,
    },
    "gridworld-heatmap-4in1": {
        "image.cmap": "RdBu",
        "figure.figsize": (5.5, 1.66),
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.subplot.left": 0.03,
        "figure.subplot.right": 0.96,
        "figure.subplot.top": 0.85,
        "figure.subplot.bottom": 0.14,
        "figure.subplot.wspace": 0.1,
    },
    "tex": {
        "text.usetex": True,
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "pgf.preamble": "\n".join([r"\usepackage{figsymbols}", r"\usepackage{times}"]),
    },
}


@contextlib.contextmanager
def setup_styles(styles: Iterable[str]) -> Iterator[None]:
    """Context manager: uses specified matplotlib styles while in context.
    Side-effect: if "tex" is in styles, will switch `matplotlib` backend to `pgf`.
    Args:
        styles: keys of styles defined in `STYLES`.
    Returns:
        A ContextManager. While entered in the context, the specified styles are applied,
        and (if "tex" is one of the styles) the environment variable "TEXINPUTS" is set
        to support custom macros."""
    old_tex_inputs = os.environ.get("TEXINPUTS")
    tainted = False
    try:
        if "tex" in styles:
            import matplotlib  # pylint:disable=import-outside-toplevel

            # PGF backend best for LaTeX. matplotlib probably already imported:
            # but should be able to switch as non-interactive.
            matplotlib.use("pgf", force=True)
            os.environ["TEXINPUTS"] = LATEX_DIR + ":"
            tainted = True
        styles = [STYLES[style] for style in styles]

        import matplotlib.pyplot as plt  # pylint:disable=import-outside-toplevel

        with plt.style.context(styles):
            yield
    finally:
        if tainted:
            if old_tex_inputs is None:
                del os.environ["TEXINPUTS"]
            else:
                os.environ["TEXINPUTS"] = old_tex_inputs
