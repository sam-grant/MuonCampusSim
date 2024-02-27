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

def Plot1DOverlay(data_dict, nbins=100, xmin=-1.0, xmax=1.0, title=None, xlabel=None, ylabel=None, fout="hist.png", legPos="best", NDPI=300, includeBlack=False, logY=False, legFontSize=12):

    # Create figure and axes
    fig, ax = plt.subplots()

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

# Example usage:
# hists_dict = {
#     'Label1': data1,
#     'Label2': data2,
#     ...
# }
# Plot1DOverlay(hists_dict, title='Histogram Overlay', xlabel='X Label', ylabel='Y Label', fout='output.png')


# def Plot1DOverlay(hists, nbins=100, xmin=-1.0, xmax=1.0, title=None, xlabel=None, ylabel=None, fout="hist.png", labels=None, legPos="upper right", NDPI=300, includeBlack=False, logY=False, legFontSize=12):

#     # Create figure and axes
#     fig, ax = plt.subplots()

#     # Iterate over the hists and plot each one
#     for i, hist in enumerate(hists):
#         colour = colours[i]
#         if not includeBlack: colour = colours[i+1]
#         counts, bin_edges, _ = ax.hist(hist, bins=nbins, range=(xmin, xmax), histtype='step', edgecolor=colours, linewidth=1.0, fill=False, density=False, color=colour, label=labels[i], log=logY)

#     # Set x-axis limits
#     ax.set_xlim(xmin, xmax)

#     ax.set_title(title, fontsize=16, pad=10)
#     ax.set_xlabel(xlabel, fontsize=14, labelpad=10) 
#     ax.set_ylabel(ylabel, fontsize=14, labelpad=10) 

#     # Set font size of tick labels on x and y axes
#     ax.tick_params(axis='x', labelsize=14)  # Set x-axis tick label font size
#     ax.tick_params(axis='y', labelsize=14)  # Set y-axis tick label font size
    
#     if (ax.get_xlim()[1] > 9999 or ax.get_xlim()[1] < 9.999e-3):
#         ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#         ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#         ax.xaxis.offsetText.set_fontsize(14)
#     if (ax.get_ylim()[1] > 9999 or ax.get_ylim()[1] < 9.999e-3):
#         ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
#         ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#         ax.yaxis.offsetText.set_fontsize(14)

#     # Add legend to the plot
#     ax.legend(loc=legPos, frameon=False, fontsize=legFontSize)

#     # Save the figure
#     plt.savefig(fout, dpi=NDPI, bbox_inches="tight")
#     print("---> Written", fout)

#     # Clear memory
#     plt.close()
