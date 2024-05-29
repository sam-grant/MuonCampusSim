""" 
Samuel Grant
Feb 2024
Testing transverse wedge offset.
"""

import matplotlib.pyplot as plt
import numpy as np
import math
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

def Plot(coords_x, coords_z, x1_, z1_, x2_, z2_, title=None, xlabel="z [in]", ylabel="x [in]", fout="../../Images/WedgeGeom.png"):

    # Create figure and axes
    fig, ax = plt.subplots()

    # Draw wedge
    ax.plot(coords_z, coords_x, "b-") 
    ax.fill(coords_z, coords_x, 'b', alpha=0.3)  
   
    # Plot line
    # ax.plot(x1_, z1_, color="red", linestyle='-', linewidth=1, label="t0 = 2.51604 * wedge_offset")
    ax.plot(z2_, x2_, color="red", linestyle='-', linewidth=1.5, label = r"$1/t_{0} \cdot dt/dx = -1 / \Delta x$" + "\n" + r"$t_{0} = 2 \cdot \Delta x \cdot \cot(17 \pi / 180)$") # label = r"$1/t_{0}dt/dx = -1/\Delta x$\n$t_{0} = 2 \cdot \Delta x \cdot \cot(17 \pi/180)$")


    # Set title, xlabel, and ylabel
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel(xlabel, fontsize=13, labelpad=10) 
    ax.set_ylabel(ylabel, fontsize=13, labelpad=10) 

    # Set font size of tick labels on x and y axes
    ax.tick_params(axis='x', labelsize=13)  # Set x-axis tick label font size
    ax.tick_params(axis='y', labelsize=13)  # Set y-axis tick label font size

    # Scientific notation
    if ax.get_xlim()[1] > 9999 or ax.get_xlim()[1] < 9.999e-3:
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.xaxis.offsetText.set_fontsize(13)
    if ax.get_ylim()[1] > 9999 or ax.get_ylim()[1] < 9.999e-3:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.yaxis.offsetText.set_fontsize(13)

    ax.set_aspect('equal')

    # ax.set_ylim(-0.5, 0.5)
    # ax.set_xlim(-3, 3)

    # Save the figure
    # plt.grid(True)

    # Legned
    # plt.legend(frameon=False, loc="lower right", fontsize=13, bbox_to_anchor=(0.6, 1.0))
    # bbox_to_anchor=(0.5, 1.1))

    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    # plt.axvline(x=0, color='gray', linestyle='--')

    plt.savefig(fout, dpi=300, bbox_inches="tight")
    print("---> Written", fout)

    # Clear memory
    plt.close()

    return

def WedgeGeom(wedge_offset=0., base_length=5.12, side_height=2.37, peak_angle=146):

    # Calculate dimensions for the triangle
    triangle_height = 2.56 / (math.tan(math.radians(peak_angle / 2)))

    # [peak, right upper, left upper, left lower, right lower, peak] 
    coords_x = np.array([wedge_offset, triangle_height+wedge_offset, triangle_height+wedge_offset+side_height, triangle_height+wedge_offset+side_height, triangle_height+wedge_offset, wedge_offset])
    coords_z = np.array([0, base_length/2, base_length/2, -base_length/2, -base_length/2, 0])

    return coords_x, coords_z # returns inches!

def Thickness(t0, x, drel_thickness_dx):
    return t0 * (1 + drel_thickness_dx * x)

def RunEremeyFormula(wedge_offset, t0): 

    N = 100 
    drel_thickness_dx = -1/wedge_offset 
    # t0 = 2.51604 * wedge_offset # 2 * math.cos(math.radians(17))/math.sin(math.radians(17)) * wedge_offset
    x_ = np.linspace(wedge_offset, 3.15+wedge_offset, 2)
    z_ = np.array([Thickness(t0, x, drel_thickness_dx) for x in x_])

    print("---> drel_thickness_dx = ", drel_thickness_dx)

    return x_, z_ 

def main():

    # Wedge offset parameter
    wedge_offset = -0.00955*1e3 / 25.4

    # Wedge geometry 
    coords_x, coords_z = WedgeGeom(wedge_offset=wedge_offset)

    # Run using Eremey's method
    t0 = 2.51604 * wedge_offset * 2
    xa1_, za1_ = RunEremeyFormula(wedge_offset=wedge_offset, t0=t0)

    t0 = 2 * wedge_offset / math.tan(math.radians(17)) / 2
    xa2_, za2_ = RunEremeyFormula(wedge_offset=wedge_offset, t0=t0)

    # Plot(coords_x*25.4*1e-3, coords_z*25.4*1e-3, xa1_*25.4*1e-3, za1_*25.4*1e-3, xa2_*25.4*1e-3, za2_*25.4*1e-3, fout="../../Images/WedgeGeomA.png")
    Plot(coords_x, coords_z, xa1_, za1_, xa2_, za2_, fout="../../Images/WedgeGeom/WedgeGeom_B4C.png")
    # Plot(coords_x, coords_z, xa_, za_, fout="../../Images/WedgeGeomA.png")

    return

if __name__ == "__main__":
    main()

