"""
Samuel Grant
Feb 2024

Utility functions for phase space analysis.

"""

import math

def Round(value, sf):

    if value == 0.00:
        return "0"
    elif math.isnan(value):
        return "NaN"
    else:

        # Determine the order of magnitude
        magnitude = math.floor(math.log10(abs(value))) + 1

        # Calculate the scale factor
        scale_factor = sf - magnitude

        # Truncate the float to the desired number of significant figures
        truncated_value = math.trunc(value * 10 ** scale_factor) / 10 ** scale_factor

        # Convert the truncated value to a string
        truncated_str = str(truncated_value).rstrip('0').rstrip('.')

        return truncated_str

import numpy as np

def PadShorterArray(x, y):
    # Pad the shorter array with zeros to match the length of the longer array.
    max_length = max(len(x), len(y))
    
    # Pad shorter array with zeros to match the length of the longer array
    x_padded = np.pad(x, (0, max_length - len(x)), mode='constant')
    y_padded = np.pad(y, (0, max_length - len(y)), mode='constant')
    
    return x_padded, y_padded

def TruncateLongerArray(x, y):
    # Truncate the longer array to match the length of the shorter array.

    min_length = min(len(x), len(y))
    
    # Truncate longer array to match the length of the shorter array
    x_truncated = x[:min_length]
    y_truncated = y[:min_length]
    
    return x_truncated, y_truncated

from scipy import stats

# Stats for histograms tends to assume a normal distribution
# ROOT does the same thing with TH1
def GetBasicStats(data, xmin, xmax):

    filtered_data = data[(data >= xmin) & (data <= xmax)]  # Filter data within range

    N = len(filtered_data)                      
    mean = np.mean(filtered_data)  
    meanErr = stats.sem(filtered_data) # Mean error (standard error of the mean from scipy)
    stdDev = np.std(filtered_data) # Standard deviation
    stdDevErr = np.sqrt(stdDev**2 / (2*N)) # Standard deviation error assuming normal distribution
    underflows = len(data[data < xmin]) # Number of underflows
    overflows = len(data[data > xmax])

    return N, mean, meanErr, stdDev, stdDevErr, underflows, overflows

import matplotlib.pyplot as plt

# Define the colourmap colours
colours = [
    (0., 0., 0.),                                                   # Black
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # Blue
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # Red
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), # Green
    (1.0, 0.4980392156862745, 0.054901960784313725),                # Orange
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),    # Purple
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),   # Cyan
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),   # Pink
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), # Brown
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),   # Gray 
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333)  # Yellow
]

def GetExtendedColours(N, gradient='viridis'):
    # Generate a color map gradient starting from black
    cmap = plt.cm.get_cmap(gradient, N)
    # Extract the RGB values from the color map
    colours_extended = [cmap(i)[:3] for i in range(N)]
    return colours_extended

def Plot1D(data, nbins=100, xmin=-1.0, xmax=1.0, title=None, xlabel=None, ylabel=None, fout="hist.png", legPos="best", stats=True, peak=False, underOver=False, errors=False, NDPI=300):
    
    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot the histogram with outline
    counts, bin_edges, _ = ax.hist(data, bins=nbins, range=(xmin, xmax), histtype='step', edgecolor='black', linewidth=1.0, fill=False, density=False)

    # Set x-axis limits
    ax.set_xlim(xmin, xmax)

    # Calculate statistics
    N, mean, meanErr, stdDev, stdDevErr, underflows, overflows = GetBasicStats(data, xmin, xmax)

    # Create legend text
    legend_text = f"Entries: {N}\nMean: {Round(mean, 3)}\nStd Dev: {Round(stdDev, 3)}"
    if errors: legend_text = f"Entries: {N}\nMean: {Round(mean, 3)}$\pm${Round(meanErr, 1)}\nStd Dev: {Round(stdDev, 3)}$\pm${Round(stdDevErr, 1)}"
    if underOver: legend_text += f"\nUnderflows: {underflows}\nOverflows: {overflows}"

    # Add legend to the plot
    if stats: ax.legend([legend_text], loc=legPos, frameon=False, fontsize=14)

    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xlabel(xlabel, fontsize=14, labelpad=10) 
    ax.set_ylabel(ylabel, fontsize=14, labelpad=10) 

    # Set font size of tick labels on x and y axes
    ax.tick_params(axis='x', labelsize=14)  # Set x-axis tick label font size
    ax.tick_params(axis='y', labelsize=14)  # Set y-axis tick label font size

    if (ax.get_xlim()[1] > 9999 or ax.get_xlim()[1] < 9.999e-3):
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.xaxis.offsetText.set_fontsize(14)
    if (ax.get_ylim()[1] > 9999 or ax.get_ylim()[1] < 9.999e-3):
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.yaxis.offsetText.set_fontsize(14)

    # Save the figure
    plt.savefig(fout, dpi=NDPI, bbox_inches="tight")
    print("---> Written", fout)

    # Clear
    plt.close()

from matplotlib.ticker import ScalarFormatter

def Plot1DOverlay(data_dict, nbins=100, xmin=-1.0, xmax=1.0, title=None, xlabel=None, ylabel=None, fout="hist.png", legPos="best", NDPI=300, includeBlack=False, logY=False, legFontSize=12, colours_extended=False):

    # Create figure and axes
    fig, ax = plt.subplots()

    # colours = colours
    # if colours_extended: colours = GetExtendedColours(len(data_dict))

    # Iterate over the hists_dict and plot each one
    for i, (label, hist) in enumerate(data_dict.items()):
        colour = colours[i]
        if not includeBlack: colour = colours[i+1]
        # ax.hist(hist, bins=nbins, range=(xmin, xmax), histtype='step', linewidth=1.0, fill=False, density=False, label=label, log=logY)
        counts, bin_edges, _ = ax.hist(hist, bins=nbins, range=(xmin, xmax), histtype='step', edgecolor=colour, linewidth=1.0, fill=False, density=False, color=colour, label=label, log=logY)

    # Set x-axis limits
    ax.set_xlim(xmin, xmax)

    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel(xlabel, fontsize=13, labelpad=10) 
    ax.set_ylabel(ylabel, fontsize=13, labelpad=10) 

    # Set font size of tick labels on x and y axes
    ax.tick_params(axis='x', labelsize=13)  # Set x-axis tick label font size
    ax.tick_params(axis='y', labelsize=13)  # Set y-axis tick label font size
    
    if (ax.get_xlim()[1] > 9999 or ax.get_xlim()[1] < 9.999e-3):
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.xaxis.offsetText.set_fontsize(13)
    if (ax.get_ylim()[1] > 9999 or ax.get_ylim()[1] < 9.999e-3):
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.yaxis.offsetText.set_fontsize(13)

    # Add legend to the plot
    ax.legend(loc=legPos, frameon=False, fontsize=legFontSize)

    # Save the figure
    plt.savefig(fout, dpi=NDPI, bbox_inches="tight")
    print("---> Written", fout)

    # Clear memory
    plt.close()

def PlotGraph(x, y, xerr=[], yerr=[], title=None, xlabel=None, ylabel=None, fout="scatter.png", NDPI=300):

   # Create a scatter plot with error bars using NumPy arrays 

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot scatter with error bars
    if len(xerr)==0: xerr = [0] * len(x) # Sometimes we only use yerr
    if len(yerr)==0: yerr = [0] * len(y) # Sometimes we only use yerr

    if len(x) != len(y): print("Warning: x has length", len(x),", while y has length", len(y))

    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', color='black', markersize=4, ecolor='black', capsize=2, elinewidth=1, linestyle='None')

    # Set title, xlabel, and ylabel
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel(xlabel, fontsize=13, labelpad=10) 
    ax.set_ylabel(ylabel, fontsize=13, labelpad=10) 

    # Set font size of tick labels on x and y axes
    ax.tick_params(axis='x', labelsize=13)  # Set x-axis tick label font size
    ax.tick_params(axis='y', labelsize=13)  # Set y-axis tick label font size

     # Scientific notation
    if (ax.get_xlim()[1] > 9999 or ax.get_xlim()[1] < 9.999e-3):
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.xaxis.offsetText.set_fontsize(13)
    if (ax.get_ylim()[1] > 9999 or ax.get_ylim()[1] < 9.999e-3):
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.yaxis.offsetText.set_fontsize(13)

    # Save the figure
    plt.savefig(fout, dpi=NDPI, bbox_inches="tight")
    print("---> Written", fout)

    # Clear memory
    plt.close()

def PlotNormEntriesGraph(x, y, xerr=[], yerr=[], title=None, xlabel=None, ylabel=None, fout="scatter.png", NDPI=300):

   # Create a scatter plot with error bars using NumPy arrays 

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot scatter with error bars
    if len(xerr)==0: xerr = [0] * len(x) # Sometimes we only use yerr
    if len(yerr)==0: yerr = [0] * len(y) # Sometimes we only use yerr

    if len(x) != len(y): print("Warning: x has length", len(x),", while y has length", len(y))

    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', color=colours[2], markersize=4, ecolor=colours[2], capsize=2, elinewidth=1, linestyle='None')

    # ax.set_ylim(0, 1.1)
    plt.axhline(y=1, color='gray', linestyle='-', linewidth=1, zorder=0)
    plt.axhline(y=1.04, color='gray', linestyle='--', linewidth=1, zorder=0)
    plt.axhline(y=0.96, color='gray', linestyle='--', linewidth=1, zorder=0)
    # Add label indicating the dashed lines represent +/- 4%
    # ax.text(0.5, 1.05, r"$\pm 4\%$", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=13)

    # Set title, xlabel, and ylabel
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel(xlabel, fontsize=13, labelpad=10) 
    ax.set_ylabel(ylabel, fontsize=13, labelpad=10) 

    # Set font size of tick labels on x and y axes
    ax.tick_params(axis='x', labelsize=13)  # Set x-axis tick label font size
    ax.tick_params(axis='y', labelsize=13)  # Set y-axis tick label font size

     # Scientific notation
    if (ax.get_xlim()[1] > 9999 or ax.get_xlim()[1] < 9.999e-3):
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.xaxis.offsetText.set_fontsize(13)
    if (ax.get_ylim()[1] > 9999 or ax.get_ylim()[1] < 9.999e-3):
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.yaxis.offsetText.set_fontsize(13)

    # Save the figure
    plt.savefig(fout, dpi=NDPI, bbox_inches="tight")
    print("---> Written", fout)

    # Clear memory
    plt.close()

from matplotlib.colors import LogNorm

def Plot2D(x, y, nBinsX=100, xmin=-1.0, xmax=1.0, nBinsY=100, ymin=-1.0, ymax=1.0, title=None, xlabel=None, ylabel=None, fout="hist.png", cb=True, NDPI=300, logZ=False):

    # Create 2D histogram
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=[nBinsX, nBinsY], range=[[xmin, xmax], [ymin, ymax]])

    # Set up the plot
    fig, ax = plt.subplots()

    # Plot the 2D histogram
    norm = None
    if logZ: norm=LogNorm()
    im = ax.imshow(hist.T, cmap='inferno', extent=[xmin, xmax, ymin, ymax], aspect='auto', origin='lower', vmax=np.max(hist), norm=norm) # , norm=cb.LogNorm())

    # Add colourbar
    if cb: plt.colorbar(im)

    plt.title(title, fontsize=15, pad=10)
    plt.xlabel(xlabel, fontsize=13, labelpad=10)
    plt.ylabel(ylabel, fontsize=13, labelpad=10)

     # Scientific notation
    if (ax.get_xlim()[1] > 9999 or ax.get_xlim()[1] < 9.999e-3):
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.xaxis.offsetText.set_fontsize(13)
    if (ax.get_ylim()[1] > 9999 or ax.get_ylim()[1] < 9.999e-3):
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.yaxis.offsetText.set_fontsize(13)

    plt.savefig(fout, dpi=NDPI, bbox_inches="tight")
    print("---> Written", fout)

    plt.close()

def Plot3D(x, y, z, nBinsX=100, xmin=-1.0, xmax=1.0, nBinsY=100, ymin=-1.0, ymax=1.0, zmax=1.0, title=None, xlabel=None, ylabel=None, zlabel=None, fout="3d_plot.png", contours=False, cb=True, NDPI=300):

    # Create a 2D histogram in xy, with the average z values on the colorbar
    hist_xy, x_edges_xy, y_edges_xy = np.histogram2d(x, y, bins=[nBinsX, nBinsY], range=[[xmin, xmax], [ymin, ymax]], weights=z)

    # Calculate the histogram bin counts
    hist_counts, _, _ = np.histogram2d(x, y, bins=[nBinsX, nBinsY], range=[[xmin, xmax], [ymin, ymax]])
    # Avoid division by zero and invalid values
    non_zero_counts = hist_counts > 0
    hist_xy[non_zero_counts] /= hist_counts[non_zero_counts]

    # Set up the plot

    fig, ax = plt.subplots()

    # Plot the 2D histogram
    im = ax.imshow(hist_xy.T, cmap='inferno', extent=[xmin, xmax, ymin, ymax], aspect='auto', origin='lower', vmax=zmax) # z.max()) # , norm=cm.LogNorm())

    # Add colourbar
    cbar = plt.colorbar(im) # , ticks=np.linspace(zmin, zmax, num=10)) 

    # Add contour lines to visualize bin boundaries
    if contours:
        contour_levels = np.linspace(zmin, zmax, num=nBinsZ)
        print(contour_levels)
        # ax.contour(hist_xy.T, levels=[66], extent=[xmin, xmax, ymin, ymax], colors='white', linewidths=0.7)
        ax.contour(hist_xy.T, levels=contour_levels, extent=[xmin, xmax, ymin, ymax], colors='white', linewidths=0.7)

    plt.title(title, fontsize=16, pad=10)
    plt.xlabel(xlabel, fontsize=14, labelpad=10)
    plt.ylabel(ylabel, fontsize=14, labelpad=10)
    cbar.set_label(zlabel, fontsize=14, labelpad=10)

    # Scientific notation
    if ax.get_xlim()[1] > 999:
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.xaxis.offsetText.set_fontsize(14)
    if ax.get_ylim()[1] > 999:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.yaxis.offsetText.set_fontsize(14)
    # if ax.get_zlim()[1] > 999:
    #     ax.zaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #     ax.ticklabel_format(style='sci', axis='z', scilimits=(0,0))
    #     ax.zaxis.offsetText.set_fontsize(14)

    plt.savefig(fout, dpi=NDPI, bbox_inches="tight")
    print("---> Written", fout)

    # Clear memory
    plt.close()

    return

def Plot2DWith1DProj(x, y, nBinsX=100, xmin=-1.0, xmax=1.0, nBinsY=100, ymin=-1.0, ymax=1.0, title=None, xlabel=None, ylabel=None, fout="hist.png", NDPI=300, logZ=False):

    # Create 2D histogram
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=[nBinsX, nBinsY], range=[[xmin, xmax], [ymin, ymax]])

    # Set up the plot
    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], wspace=0.1, hspace=0.1)

    # Plot the 2D histogram
    ax1 = fig.add_subplot(gs[0, 0])

    norm = None
    if logZ: norm=LogNorm()
    im = ax1.imshow(hist.T, cmap='inferno', extent=[xmin, xmax, ymin, ymax], aspect='auto', origin='lower', norm=norm)

    # Format main plot axes
    ax1.set_title(title, fontsize=16, pad=10)
    ax1.set_xlabel(xlabel, fontsize=14, labelpad=10) 
    ax1.set_ylabel(ylabel, fontsize=14, labelpad=10) 

    # Set font size of tick labels on x and y axes
    ax1.tick_params(axis='x', labelsize=14)  # Set x-axis tick label font size
    ax1.tick_params(axis='y', labelsize=14)  # Set y-axis tick label font size

    # Scientific notation
    if ax1.get_xlim()[1] > 999:
        ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax1.xaxis.offsetText.set_fontsize(14)
    if ax1.get_ylim()[1] > 999:
        ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax1.yaxis.offsetText.set_fontsize(14)

    # Draw a horizontal white line at y=50 
    # ax1.axhline(y=50, color='white', linestyle='--', linewidth=1)

    # Add colourbar
    ax_cb = fig.add_subplot(gs[0, 1])
    cbar = plt.colorbar(im, cax=ax_cb)
    cbar.ax.set_position(cbar.ax.get_position().shrunk(0.25, 1))
    cbar.ax.set_position(cbar.ax.get_position().translated(0.10, 0))
    cbar.ax.tick_params(labelsize=14)

    # Project the 2D histogram onto the x-axis
    hist_x = np.sum(hist, axis=1)
    bin_centers_x = (x_edges[:-1] + x_edges[1:]) / 2

    # Create a dummy copy of ax1 to be shared with projection axes,
    # this prevents the original being modified
    ax1_dummy = ax1.figure.add_axes(ax1.get_position(), frame_on=False)
    ax1_dummy.set_xticks([])  # Turn off x-axis ticks of ax1_dummy
    ax1_dummy.set_yticks([])  # Turn off y-axis ticks of ax1_dummy
    # Make sure that the ranges are the same between ax1 and ax1_dummy
    ax1_dummy.set_xlim(ax1.get_xlim())
    ax1_dummy.set_ylim(ax1.get_ylim())

    # Plot the 1D histogram along the x-axis
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1_dummy)

    counts, bin_edges, _ = ax2.hist(x, bins=nBinsX, range=(xmin, xmax), histtype='step', edgecolor='black', linewidth=1.0, fill=False)

    # Set the position of ax2 while maintaining the same width and adjusting the height
    ax2.set_position([ax1.get_position().x0, ax1.get_position().y0 + ax1.get_position().height + 0.01, ax1.get_position().width, ax1.get_position().height / 5])

    # Turn off appropriate tick markers/numbering and spines 
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(True)

    ax2.yaxis.set_ticklabels([])
    # ax2.tick_params(axis='x', which='major', length=ax1.get_tick_params('length'))  # Turn on tick markers
    ax2.tick_params(axis='y', which='major', length=0)  # Hide tick markers

    # Project the 2D histogram onto the y-axis
    hist_y = np.sum(hist, axis=0)
    bin_centers_y = (y_edges[:-1] + y_edges[1:]) / 2

    # Plot the 1D histogram along the y-axis (rotated +90 degrees)
    ax3 = fig.add_subplot(gs[1, 1], sharey=ax1_dummy)
    counts, bin_edges, _ = ax3.hist(y, bins=nBinsY, range=(ymin, ymax), histtype='step', edgecolor='black', linewidth=1.0, fill=False, orientation='horizontal')

    # Set the position of ax3 while maintaining the same height and adjusting the width
    ax3.set_position([ax1.get_position().x0 + ax1.get_position().width + 0.01, ax1.get_position().y0, ax1.get_position().width / 5, ax1.get_position().height])

    # Turn off appropriate tick markers/numbering and spines 
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(True)
    ax3.spines['bottom'].set_visible(False)

    ax3.xaxis.set_ticklabels([])
    ax3.tick_params(axis='x', which='major', length=0)  # Hide tick markers
    # ax3.tick_params(axis='y', which='major', length=ax1.get_tick_params('length')) 

    # Save figure
    plt.savefig(fout, dpi=NDPI, bbox_inches="tight")
    print("---> Written", fout)

    # Clear memory
    plt.close()

def Plot1DLossesOverlay(data_dict, nbins=100, xmin=-1.0, xmax=1.0, title=None, xlabel=None, ylabel=None, fout="hist.png", legPos="best", NDPI=300, includeBlack=False, logY=False, legFontSize=12, colours_extended=False):

    # Create figure and axes
    fig, ax = plt.subplots()

    # colours = colours
    # if colours_extended: colours = GetExtendedColours(len(data_dict))
    # norm_hist = data_dict["No wedge"]
    # Iterate over the hists_dict and plot each one
    for i, (label, hist) in enumerate(data_dict.items()):
        colour = colours[i+1]
        # if not includeBlack: colour = colours[i+1]
        if label == "No wedge": colour = colours[0]
        #     no_wedge_counts, no_wedge_bin_edges, _ = np.histogram(hist, bins=nbins, range=(xmin, xmax), density=False)

        # hist = hist / norm_hist

        # ax.hist(hist, bins=nbins, range=(xmin, xmax), histtype='step', linewidth=1.0, fill=False, density=False, label=label, log=logY)
        counts, bin_edges, _ = ax.hist(hist, bins=nbins, range=(xmin, xmax), histtype='step', edgecolor=colour, linewidth=1.0, fill=False, density=False, color=colour, label=label, log=logY)

    # no_wedge_hist_counts = None
    # no_wedge_hist_counts, bin_edges = np.histogram(data_dict["No wedge"], bins=nbins, range=(xmin, xmax), density=False)
    # # Iterate over the hists_dict and calculate normalization
    # for i, (label, hist) in enumerate(data_dict.items()):
    #     if i == 0: continue
    #     colour = colours[i]
    #     # if label == "No wedge":
    #     #     # Calculate the histogram for "No wedge"
    #     #     no_wedge_hist_counts, bin_edges = np.histogram(hist, bins=nbins, range=(xmin, xmax), density=False)
    #     # else:
    #     # Calculate the histogram for the current data
    #     counts, _ = np.histogram(hist, bins=nbins, range=(xmin, xmax), density=False)
    #     # Normalize the current histogram based on "No wedge" bin-by-bin
    #     weights = 1 / no_wedge_hist_counts
    #     # Plot the normalized histogram
    #     ax.hist(bin_edges[:-1], bins=bin_edges, weights=weights, histtype='step', edgecolor=colour, linewidth=1.0, fill=False, density=False, color=colour, label=label, log=logY)
        
    # Set x-axis limits
    ax.set_xlim(xmin, xmax)

    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel(xlabel, fontsize=13, labelpad=10) 
    ax.set_ylabel(ylabel, fontsize=13, labelpad=10) 

    plt.axvline(x=6000, color='gray', linestyle='--', linewidth=1, zorder=0)

    # Set font size of tick labels on x and y axes
    ax.tick_params(axis='x', labelsize=13)  # Set x-axis tick label font size
    ax.tick_params(axis='y', labelsize=13)  # Set y-axis tick label font size
    
    if (ax.get_xlim()[1] > 9999 or ax.get_xlim()[1] < 9.999e-3):
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.xaxis.offsetText.set_fontsize(13)
    if (ax.get_ylim()[1] > 9999 or ax.get_ylim()[1] < 9.999e-3):
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.yaxis.offsetText.set_fontsize(13)

    # Add legend to the plot
    ax.legend(loc="upper left", frameon=False, fontsize=legFontSize)

    # Save the figure
    plt.savefig(fout, dpi=NDPI, bbox_inches="tight")
    print("---> Written", fout)

    # Clear memory
    plt.close()
