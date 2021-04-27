import math

GR = (1.0 + math.sqrt(5)) / 2.0

# custom colors
colors = {
    "V": "#346ebf",
    "T": "#2faf41",
    "VT": "#ee1d23",
}

# custom fontsizes
fontsize_medium = 12
fontsize_xsmall = 0.7 * fontsize_medium
fontsize_tiny = 0.6 * fontsize_medium

# custom mpl style
mpl_style = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,  # figure dots per inch
    "xtick.labelsize": fontsize_xsmall,
    "ytick.labelsize": fontsize_xsmall,
    "axes.labelsize": fontsize_medium,
}
