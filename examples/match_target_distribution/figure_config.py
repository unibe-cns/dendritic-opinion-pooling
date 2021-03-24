import math

GR = (1. + math.sqrt(5)) / 2.

# custom colors
colors = {
    'prior': '#909090',
    'priorweak': '#D0D0D0',
    'V': '#346ebf',
    'Vgray': '#698dbf',
    'Vweak': '#b8caec',
    'Valt': '#b8e5ed',
    'T': '#2faf41',
    'Tgray': '#60af6b',
    'Tweak': '#b7edc0',
    'VT': '#ee1d23',
    'VTalt': '#ed1c8b',
    'VTweak': '#eda6a8',
    'energy': '#111111',
    'MAP': '0.3',
    'sym': '0.8',
    'E': '#1F2041',
    'I': '#FFC857',
}

# custom fontsizes
fontsize_medium = 12
fontsize_small = 0.8 * fontsize_medium
fontsize_xsmall = 0.7 * fontsize_medium
fontsize_tiny = 0.6 * fontsize_medium
fontsize_xtiny = 0.5 * fontsize_medium
fontsize_xxtiny = 0.4 * fontsize_medium
fontsize_large = 1.2 * fontsize_medium
fontsize_xlarge = 1.4 * fontsize_medium
fontsize_xxlarge = 1.6 * fontsize_medium

# custom line widths
lw_medium = 2.
lw_narrow = 1.

# custom figure sizes
fig_scaling = 0.8
single_figure = (fig_scaling * 6.4, fig_scaling * 3.96)  # using default width and golden ratio
double_figure_horizontal = (1.5 * single_figure[0], single_figure[1])
triple_figure_horizontal = (2.2 * single_figure[0], single_figure[1])
double_figure_vertical = (single_figure[0], 1.3 * single_figure[1])
triple_figure_vertical = (single_figure[0], 1.6 * single_figure[1])
quad_figure = (6.4, 6.4)

mpl_style = {
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300, ## figure dots per inch
    'xtick.labelsize': fontsize_xsmall,
    'ytick.labelsize': fontsize_xsmall,
    'axes.labelsize': fontsize_medium,
}
