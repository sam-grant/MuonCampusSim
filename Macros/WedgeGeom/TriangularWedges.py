import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

colours = [
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


def PlotLines(x, y_ = [], label=None, title=None, xlabel=None, ylabel=None,labels_=[], fout="line_plot.png"):

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot line without markers
    for i, y in enumerate(y_): 
        ax.plot(x, y, color=colours[i], linestyle='-', linewidth=1, label=labels_[i])

    # Set title, xlabel, and ylabel
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel(xlabel, fontsize=13, labelpad=10) 
    ax.set_ylabel(ylabel, fontsize=13, labelpad=10) 

    # Set font size of tick labels on x and y axes
    ax.tick_params(axis='x', labelsize=13)  # Set x-axis tick label font size
    ax.tick_params(axis='y', labelsize=13)  # Set y-axis tick label font size

    # Scientific notation
    if ax.get_xlim()[1] > 9999:
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.xaxis.offsetText.set_fontsize(13)
    if ax.get_ylim()[1] > 9999:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.yaxis.offsetText.set_fontsize(13)

    # Save the figure
    plt.grid(True)

    # Legned
    plt.legend(frameon=True, loc="best")
    plt.savefig(fout, dpi=300, bbox_inches="tight")
    print("---> Written", fout)

    # Clear memory
    plt.clf()

    return

def Thickness(t0, x, drel_thickness_dx):
    return t0 * (1 + drel_thickness_dx * x)

def Run(t0, xmax): # , label):

    N = 100 
    drel_thickness_dx = -1/xmax # Need t=0 at x=x_max and t=t0 at x=0
    x_ = np.linspace(0, xmax, 100)
    t_ = np.array([Thickness(t0, x, drel_thickness_dx) for x in x_])

    # PlotLine(t_, x_, label, "Thickness [inches]", "x [inches]", label+"_wedge.png")
    
    print("---> drel_thickness_dx = ", drel_thickness_dx)

    return t_, x_ 

def main():

    t1_, x1_ = Run(t0 = 2.56*2, xmax = 3.15 - 2.74) # , label="Polyurethane") 
    t2_, x2_ = Run(t0 = 2.56*2, xmax = 3.15 - 2.37) #, label="Boron Carbide")

    PlotLines(t1_, [x1_, x2_], xlabel="Thickness [inches]", ylabel="x [inches]", labels_=["Polyurethane\n$1/t_{0}\cdot dt/dx=-2.439$ in$^{-1}$", "Boron Carbide\n$1/t_{0}\cdot dt/dx=-1.282$ in$^{-1}$"], fout="wedges.png")

    return

if __name__ == "__main__":
    main()

