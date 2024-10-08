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

def bin(data, bin_width=1.0): 
    # Bin the data
    bin_edges = np.arange(min(data), max(data) + bin_width, bin_width)
    bin_indices = np.digitize(data, bin_edges)
    bin_counts = np.bincount(bin_indices)
    return bin_edges, bin_indices, bin_counts
    
def GetMode(data, nbins, xmin, xmax): 
    # Get bin width
    bin_width = abs(xmax - xmin) / nbins 
    # Filter data within range
    data = data[(data >= xmin) & (data <= xmax)] 
    # Bin
    bin_edges, bin_indices, bin_counts = bin(data, bin_width)
    # Get mode index
    mode_bin_index = np.argmax(bin_counts)
    # Get mode count
    mode_count = bin_counts[mode_bin_index]
    # Get bin width
    # bin_width = bin_edges[mode_bin_index] - bin_edges[mode_bin_index + 1]
    # Calculate the bin center corresponding to the mode
    mode_bin_center = (bin_edges[mode_bin_index] + bin_edges[mode_bin_index + 1]) / 2
    # Mode uncertainty 
    N = len(data)
    mode_bin_center_err = np.sqrt(N / (N - mode_count)) * bin_width
    return mode_bin_center, abs(mode_bin_center_err)


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

# colours = [
#     (0., 0., 0.),                                                   # Black
#     (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # Blue
#     (1.0, 0.4980392156862745, 0.054901960784313725),                # Orange
#     (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # Red
#     (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), # Green

#     (0.5803921568627451, 0.403921568627451, 0.7411764705882353),    # Purple
#     (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),   # Cyan
#     (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),   # Pink
#     (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), # Brown
#     (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),   # Gray 
#     (0.7372549019607844, 0.7411764705882353, 0.13333333333333333)  # Yellow
# ]

def GetExtendedColours(N, gradient='viridis'):
    # Generate a color map gradient starting from black
    cmap = plt.cm.get_cmap(gradient, N)
    # Extract the RGB values from the color map
    colours_extended = [cmap(i)[:3] for i in range(N)]
    return colours_extended

def Plot1D(data, nbins=100, xmin=-1.0, xmax=1.0, title=None, xlabel=None, ylabel=None, fout="hist.png", legPos="best", stats=True, peak=False, underOver=False, errors=False, logY=True, NDPI=300):
    
    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot the histogram with outline
    counts, bin_edges, _ = ax.hist(data, bins=nbins, range=(xmin, xmax), histtype='step', edgecolor='black', linewidth=1.0, fill=False, density=False, log=logY)

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

        # Colours
        colour = colours[i]
        if not includeBlack: colour = colours[i+1]

        # ax.hist(hist, bins=nbins, range=(xmin, xmax), histtype='step', linewidth=1.0, fill=False, density=False, label=label, log=logY)

        print(label, colour)
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

def Plot1DOverlayWithStats(data_dict, nbins=100, xmin=-1.0, xmax=1.0, title=None, xlabel=None, ylabel=None, fout="hist.png", legPos="best", NDPI=300, includeBlack=False, logY=False, legFontSize=12, colours_extended=False):

    # Create figure and axes
    fig, ax = plt.subplots()

    # colours = colours
    # if colours_extended: colours = GetExtendedColours(len(data_dict))

    # Iterate over the hists_dict and plot each one
    for i, (label, hist) in enumerate(data_dict.items()):
        colour = colours[i]
        if not includeBlack: colour = colours[i+1]

        # Calculate statistics
        N, mean, meanErr, stdDev, stdDevErr, underflows, overflows = GetBasicStats(hist, xmin, xmax)
        peak, peakErr = GetMode(hist, nbins, xmin, xmax)

        # Create legend text
        # label_text = f"Entries: {N}\nMean: {Round(mean, 6)}$\pm${Round(meanErr, 1)}\nStd Dev: {Round(stdDev, 4)}$\pm${Round(stdDevErr, 1)}\nPeak: {Round(peak, 4)}$\pm${Round(peakErr, 1)}"
        # label_text = f"Entries: {N}\nMean: {Round(mean, 6)}$\pm${Round(meanErr, 1)}\nPeak: {Round(peak, 4)}$\pm${Round(peakErr, 1)}"
        label_text = f"Mean: {Round(mean, 6)}$\pm${Round(meanErr, 1)}\nPeak: {Round(peak, 4)}$\pm${Round(peakErr, 1)}"
        # label = r"$\bf{ "+label+ r"}$"
        label = label + "\n" + label_text
        # label=r"$\bf{"+label+"}$"+"\n"+label_text

        # label = f"\033[1m{label}\033[0m\n{label_text}"

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

def ProfileX(x, y, nBinsX=100, xmin=-1.0, xmax=1.0, nBinsY=100, ymin=-1.0, ymax=1.0): 
   
    # Create 2D histogram with one bin on the y-axis 
    hist, xEdge_, yEdge_ = np.histogram2d(x, y, bins=[nBinsX, nBinsY], range=[[xmin, xmax], [ymin, ymax]])

    # hist, xEdge_ = np.histogram(x, bins=nBinsX, range=[xmin, xmax]) # , [ymin, ymax]])

    # bin widths
    xBinWidths = xEdge_[1]-xEdge_[0]

    # Calculate the mean and RMS values of each vertical slice of the 2D distribution
    xSlice_, xSliceErr_, ySlice_, ySliceErr_, ySliceRMS_ = [], [], [], [], []

    for i in range(len(xEdge_) - 1):

        # Average x-value
        xSlice = x[ (xEdge_[i] < x) & (x <= xEdge_[i+1]) ]

        # Get y-slice within current x-bin
        ySlice = y[ (xEdge_[i] < x) & (x <= xEdge_[i+1]) ]

        # Avoid empty slices
        if len(xSlice) == 0 or len(ySlice) == 0:
            continue

        # Central values are means and errors are standard errors on the mean
        xSlice_.append(np.mean(xSlice))
        xSliceErr_.append(stats.sem(xSlice)) # RMS/sqrt(n)
        ySlice_.append(ySlice.mean()) 
        ySliceErr_.append(stats.sem(ySlice)) 
        ySliceRMS_.append(np.std(ySlice))

    return np.array(xSlice_), np.array(xSliceErr_), np.array(ySlice_), np.array(ySliceErr_), np.array(ySliceRMS_)


def PlotOffsetScanGraph(x, y, xerr=[], yerr=[], title=None, xlabel=None, ylabel=None, fout="scatter.png", NDPI=300):

   # Create a scatter plot with error bars using NumPy arrays 

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot scatter with error bars
    if len(xerr)==0: xerr = [0] * len(x) # Sometimes we only use yerr
    if len(yerr)==0: yerr = [0] * len(y) # Sometimes we only use yerr

    if len(x) != len(y): print("Warning: x has length", len(x),", while y has length", len(y))

    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', color=colours[2], markersize=4, ecolor=colours[2], capsize=2, elinewidth=1, linestyle='None')

    # ax.set_ylim(0, 1.1)
    plt.axhline(y=1, color='gray', linestyle='--', linewidth=1, zorder=0)
    # plt.axhline(y=1.04, color='gray', linestyle='--', linewidth=1, zorder=0)
    # plt.axhline(y=0.96, color='gray', linestyle='--', linewidth=1, zorder=0)
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
    im = ax.imshow(hist.T, cmap='inferno', extent=[xmin, xmax, ymin, ymax], aspect='auto', origin='lower', norm=norm) # vmax=np.max(hist), , norm=cb.LogNorm())

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

def Plot2DWith1DWedge(x, y, nBinsX=100, xmin=-1.0, xmax=1.0, nBinsY=100, ymin=-1.0, ymax=1.0, x_offset=0.0, title=None, xlabel=None, ylabel=None, fout="hist.png", cb=True, NDPI=300, logZ=False):

    # Create 2D histogram
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=[nBinsX, nBinsY], range=[[xmin, xmax], [ymin, ymax]])

    # Set up the plot
    fig, ax = plt.subplots()

    # Plot the 2D histogram
    norm = None
    if logZ: norm=LogNorm()
    im = ax.imshow(hist.T, cmap='inferno', extent=[xmin, xmax, ymin, ymax], aspect='auto', origin='lower', norm=norm) # , norm=cb.LogNorm())

    # Add colourbar
    if cb: plt.colorbar(im)

    plt.title(title, fontsize=15, pad=10)
    plt.xlabel(xlabel, fontsize=13, labelpad=10)
    plt.ylabel(ylabel, fontsize=13, labelpad=10)

    # Draw wedge
    wedge_length = (3.15-2.37) * 24.5
    x_start = (x_offset - wedge_length) # already in mm
    x_end = x_offset 
    y_start = (1.97/2) * 24.5
    y_end = -y_start 

    ax.plot([x_offset, x_offset], [ax.get_ylim()[0], ax.get_ylim()[1]], 'w--', linewidth=1)  
    # ax.plot([x_end, x_end], [y_start, y_end], 'w--')      # Right edge
    # ax.plot([x_end, x_start], [y_end, y_end], 'w--')      # Top edge
    # ax.plot([x_start, x_start], [y_end, y_start], 'w--')  # Left edge


    # Add the rectangle to the plot
    # ax.add_patch(wedge)

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


def Plot2DWith2DWedge(x, y, nBinsX=100, xmin=-1.0, xmax=1.0, nBinsY=100, ymin=-1.0, ymax=1.0, x_offset=0.0, title=None, xlabel=None, ylabel=None, fout="hist.png", cb=True, NDPI=300, logZ=False):

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

    # Draw wedge
    wedge_length = (3.15-2.37) * 24.5
    x_start = (x_offset - wedge_length) # already in mm
    x_end = x_offset 
    y_start = (1.97/2) * 24.5
    y_end = -y_start 

    ax.plot([x_start, x_end], [y_start, y_start], 'w--')  # Bottom edge
    ax.plot([x_end, x_end], [y_start, y_end], 'w--')      # Right edge
    ax.plot([x_end, x_start], [y_end, y_end], 'w--')      # Top edge
    ax.plot([x_start, x_start], [y_end, y_start], 'w--')  # Left edge


    # Add the rectangle to the plot
    # ax.add_patch(wedge)

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
    if (ax1.get_xlim()[1] > 9999 or ax1.get_xlim()[1] < 9.999e-3):
        ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax1.xaxis.offsetText.set_fontsize(13)
    if (ax1.get_ylim()[1] > 9999 or ax1.get_ylim()[1] < 9.999e-3):
        ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax1.yaxis.offsetText.set_fontsize(13)

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
    ax.legend(loc="best", frameon=False, fontsize=legFontSize)

    # Save the figure
    plt.savefig(fout, dpi=NDPI, bbox_inches="tight")
    print("---> Written", fout)

    # Clear memory
    plt.close()

def PlotOffsetScanHists(data_dict, nbins=100, xmin=-1.0, xmax=1.0, title=None, xlabel=None, ylabel=None, fout="hist.png", legPos="best", NDPI=300, includeBlack=False, logY=False, legFontSize=12, colours_extended=False):

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
    
    # plt.axvline(x=0, color='gray', linestyle='-', linewidth=1, zorder=0)
    # plt.axvline(x=0.002, color='gray', linestyle='--', linewidth=1, zorder=0)
    # plt.axvline(x=-0.002, color='gray', linestyle='--', linewidth=1, zorder=0)

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

def PlotHistAndGraph(hist, graph, labels = [None, None], title=None, xlabel=None, ylabel=None, fout="hist_graph_overlay.png"): 

    # Create figure and axes
    fig, ax = plt.subplots()
    colour="black"
    logY=False
    counts, bin_edges, _ = ax.hist(hist*1e3, bins=48, range=(-48, 48), histtype='step', edgecolor=colour, linewidth=1.0, fill=False, density=False, color=colour, label=labels[0], log=logY)

    # Plot scatter with error bars
    xerr=[] 
    yerr=[]
    if len(xerr)==0: xerr = [0] * graph["x"] # Sometimes we only use yerr
    if len(yerr)==0: yerr = [0] * graph["y"] # Sometimes we only use yerr

    ax.errorbar(x=graph["x"], y=(graph["y"]/np.max(graph["y"]))*np.max(counts), xerr=xerr, yerr=yerr, fmt='o', color='black', markersize=4, ecolor='black', capsize=2, elinewidth=1, linestyle='None', label=labels[1])

    # for i in range(len(dataHPWC["x"])):
    #     ax.plot([dataHPWC["x"][i], dataHPWC["x"][i]], [dataHPWC["y"][i], 0], color='black', linestyle='-')

    ax.set_title(title, fontsize=13, pad=10)
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
    ax.legend(loc="best", frameon=False, fontsize=13)

    plt.savefig(fout, dpi=300, bbox_inches="tight")

    print("---> Written", fout)

    # Clear memory
    plt.close()

    return

# Specific to the PWC comparisons in current scheme.
def PlotHistAndGraphWithRatio(hist, graph, labels = [None, None], title=None, xlabel=None, ylabel=None, fout="hist_graph_overlay.png", invertRatio=False, limitRatio=False): 

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6))

    # # Create figure and axes
    # fig, ax = plt.subplots()
    colour="black"
    logY=False
    counts, bin_edges, _ = ax1.hist(hist*1e3, bins=48, range=(-48, 48), histtype='step', edgecolor=colour, linewidth=1.0, fill=False, density=False, color=colour, label=labels[0], log=logY)

    # Plot scatter with error bars
    xerr=[] 
    yerr=[]
    if len(xerr)==0: xerr = [0] * graph["x"] # Sometimes we only use yerr
    if len(yerr)==0: yerr = [0] * graph["y"] # Sometimes we only use yerr

    x = graph["x"]
    y = (graph["y"]/np.max(graph["y"]))*np.max(counts)
    ax1.errorbar(x=x, y=y, xerr=xerr, yerr=yerr, fmt='o', color='black', markersize=4, ecolor='black', capsize=0, elinewidth=0, linestyle='None', label=labels[1])

    # for i in range(len(dataHPWC["x"])):
    #     ax.plot([dataHPWC["x"][i], dataHPWC["x"][i]], [dataHPWC["y"][i], 0], color='black', linestyle='-')

    ax1.set_title(title, fontsize=13, pad=10)
    ax1.set_xlabel(xlabel, fontsize=13, labelpad=10) 
    ax1.set_ylabel(ylabel, fontsize=13, labelpad=10) 

    # Set font size of tick labels on x and y axes
    ax1.tick_params(axis='x', labelsize=13)  # Set x-axis tick label font size
    ax1.tick_params(axis='y', labelsize=13)  # Set y-axis tick label font size

    if (ax1.get_xlim()[1] > 9999 or ax1.get_xlim()[1] < 9.999e-3):
        ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax1ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax1.xaxis.offsetText.set_fontsize(13)
    if (ax1.get_ylim()[1] > 9999 or ax1.get_ylim()[1] < 9.999e-3):
        ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax1.yaxis.offsetText.set_fontsize(13)

    # Add legend to the plot
    ax1.legend(loc="best", frameon=False, fontsize=13)

    # Ratio part 

    # Calculate the ratio of the histograms with a check for division by zero
    ratio = np.divide(y, counts, out=np.full_like(y, np.nan), where=((counts!=0) & (y!=0)))

    if invertRatio: ratio = np.divide(1, ratio)

    # Plot the ratio in the lower frame with error bars
    ratio_err=yerr
    ax2.errorbar(x, ratio, yerr=ratio_err, color='black', fmt='o',  markerfacecolor="none", markersize=4, linewidth=1)

    # Draw a line at zero
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)

    # More formatting 

    # Remove markers for main x-axis
    ax1.set_xticks([])

    ax2.set_xlabel(xlabel, fontsize=13, labelpad=10)
    ax2.set_ylabel("Ratio", fontsize=13, labelpad=10)
    ax2.tick_params(axis='x', labelsize=13)
    ax2.tick_params(axis='y', labelsize=13)

    # Create a second y-axis for the ratio
    ax2.yaxis.tick_left()
    ax2.xaxis.tick_bottom()
    ax2.xaxis.set_tick_params(width=0.5)
    ax2.yaxis.set_tick_params(width=0.5)

    # # Set x-axis limits for the ratio plot to match the top frame
    if limitRatio:
        ax2.set_ylim(0, 1.25)

    # ax2.set_xlim(xmin, xmax)

    # # Set titles and labels for both frames
    # ax1.set_title(title, fontsize=16, pad=10)
    # # ax1.set_xlabel("", fontsize=0, labelpad=10)
    # ax1.set_ylabel(ylabel, fontsize=14, labelpad=10)

    # # Set font size of tick labels on x and y axes for both frames
    # # ax1.tick_params(axis='x', labelsize=14)
    # ax1.tick_params(axis='y', labelsize=14)

    # # Scientific notation for top frame
    # if ax2.get_xlim()[1] > 9999:
    #     ax2.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #     ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #     ax2.xaxis.offsetText.set_fontsize(14)
    # if ax1.get_ylim()[1] > 9999:
    #     ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #     ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    #     ax1.yaxis.offsetText.set_fontsize(14)

    # # Add legend to the top frame
    # ax1.legend(loc=legPos, frameon=False, fontsize=14)

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Adjust the spacing between subplots to remove space between them
    plt.subplots_adjust(hspace=0.0)

    plt.savefig(fout, dpi=300, bbox_inches="tight")

    print("---> Written", fout)

    # Clear memory
    plt.close()

    return

    # if len(hists) > 2: 
    #     print("!!! ERROR: Plot1DRatio must take two histograms as input !!!")
    #     return

    # # Create figure and axes
    # fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6))

    # # Define a colormap
    # colours = [
    #     (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # Blue
    #     (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # Red
    # ]

    # # Create the colormap
    # cmap = ListedColormap(colours)

    # counts_ = []

    # # Iterate over the histograms and plot each one in the top frame
    # for i, hist in enumerate(hists):

    #     colour = cmap(i)

    #     # Calculate statistics for the current histogram
    #     N, mean, meanErr, stdDev, stdDevErr, underflows, overflows = GetBasicStats(hist, xmin, xmax)

    #     # Create legend text
    #     legend_text = f"Entries: {N}\nMean: {Round(mean, 3)}\nStd Dev: {Round(stdDev, 3)}"
    #     if errors:
    #         legend_text = f"Entries: {N}\nMean: {Round(mean, 4)}$\pm${Round(meanErr, 1)}\nStd Dev: {Round(stdDev, 3)}$\pm${Round(stdDevErr, 1)}"
    #     if peak and not errors:
    #         legend_text += f"\nPeak: {Round(GetMode(hist, nbins / (xmax - xmin))[0], 3)}"
    #     if peak and errors:
    #         legend_text += f"\nPeak: {Round(GetMode(hist, nbins / (xmax - xmin))[0], 3)}$\pm${Round(GetMode(hist, nbins / (xmax - xmin))[1], 1)}"

    #     if stats:
    #         label = r"$\bf{"+labels[i]+"}$"+"\n"+legend_text
    #     else:
    #         label = labels[i]

    #     # Plot the current histogram in the top frame
    #     counts, bin_edges, _ = ax1.hist(hist, bins=nbins, range=(xmin, xmax), histtype='step', edgecolor=colour, linewidth=1.0, fill=False, density=False, color=colour, label=label) 

    #     # Plot the current histogram in the top frame with error bars
    #     # hist_err = np.sqrt(hist)
    #     # bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    #     # ax1.bar(bin_centers, hist, width=0.1, align='center', alpha=0.7, label=label)
    #     # ax1.errorbar(bin_centers, hist, yerr=hist_err, fmt='none', color=colour, capsize=2)


    #     counts_.append(counts)

    # # Calculate the ratio of the histograms with a check for division by zero
    # ratio = np.divide(counts_[0], counts_[1], out=np.full_like(counts_[0], np.nan), where=(counts_[1] != 0))

    # # Calculate the statistical uncertainty for the ratio
    # ratio_err = np.divide(np.sqrt(counts_[0]), counts_[1], out=np.full_like(counts_[0], np.nan), where=(counts_[1] != 0))

    # # Create a second y-axis for the ratio
    # # ax2 = ax1.twinx() # This overlays them
    # # Create a separate figure and axis for the ratio plot
    # # fig2, ax2 = plt.subplots(figsize=(8, 2))  # Adjust the height as needed

    # # Add line at 1.0 
    # ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)


    # if invertRatio: ratio = np.divide(1, ratio)

    # # Plot the ratio in the lower frame with error bars
    # ax2.errorbar(bin_edges[:-1], ratio, yerr=ratio_err, color='black', fmt='o', markersize=4, linewidth=1)

    # # # Plot the ratio in the lower frame
    # # ax2.plot(bin_edges[:-1], ratio, color='black', marker='o', markersize=4, linewidth=0)

    # # Format 

    # # Set x-axis limits for the top frame
    # ax1.set_xlim(xmin, xmax)


    # # Remove markers for main x-axis
    # ax1.set_xticks([])

    # ax2.set_xlabel(xlabel, fontsize=14, labelpad=10)
    # ax2.set_ylabel("Ratio", fontsize=14, labelpad=10)
    # ax2.tick_params(axis='x', labelsize=14)
    # ax2.tick_params(axis='y', labelsize=14)

    # # Create a second y-axis for the ratio
    # ax2.yaxis.tick_left()
    # ax2.xaxis.tick_bottom()
    # ax2.xaxis.set_tick_params(width=0.5)
    # ax2.yaxis.set_tick_params(width=0.5)

    # # Set x-axis limits for the ratio plot to match the top frame
    # if limitRatio:
    #     ax2.set_ylim(ratioMin, ratioMax)

    # ax2.set_xlim(xmin, xmax)

    # # Set titles and labels for both frames
    # ax1.set_title(title, fontsize=16, pad=10)
    # # ax1.set_xlabel("", fontsize=0, labelpad=10)
    # ax1.set_ylabel(ylabel, fontsize=14, labelpad=10)

    # # Set font size of tick labels on x and y axes for both frames
    # # ax1.tick_params(axis='x', labelsize=14)
    # ax1.tick_params(axis='y', labelsize=14)

    # # Scientific notation for top frame
    # if ax2.get_xlim()[1] > 9999:
    #     ax2.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #     ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #     ax2.xaxis.offsetText.set_fontsize(14)
    # if ax1.get_ylim()[1] > 9999:
    #     ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #     ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    #     ax1.yaxis.offsetText.set_fontsize(14)

    # # Add legend to the top frame
    # ax1.legend(loc=legPos, frameon=False, fontsize=14)

    # # Adjust the spacing between subplots
    # plt.tight_layout()

    # # Adjust the spacing between subplots to remove space between them
    # plt.subplots_adjust(hspace=0.0)

    # # Save the figure
    # plt.savefig(fout, dpi=NDPI, bbox_inches="tight")
    # print("---> Written", fout)

    # # Clear memory
    # plt.close()

    # return