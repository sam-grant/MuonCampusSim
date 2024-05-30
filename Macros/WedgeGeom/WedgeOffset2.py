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

def Plot(coords_z, coords_x, zline_, xline_, coords_x_shift, xline_shift_, title=None, xlabel="z [in]", ylabel="x [in]", fout="../../Images/WedgeGeom.png"):
    # Create figure and axes
    fig, ax = plt.subplots()

    # Draw wedges
    ax.plot(coords_z, coords_x, "b--") # lpha=0.3) 
    # ax.fill(coords_z, coords_x, 'b', alpha=0.3)  

    ax.plot(coords_z, coords_x_shift, "b-") 
    ax.fill(coords_z, coords_x_shift, 'b', alpha=0.3)  

    # Plot line
    # ax.plot(zline_, xline_, color="red", linestyle='-', linewidth=1.5) # , label=r"$x = x_{0} + z \frac{dx}{dz}$")
    ax.plot(zline_, xline_shift_, color="red", linestyle='-', linewidth=1.5)
    # ax.plot(z2_, x2_, color="red", linestyle='-', linewidth=1.5, label = r"$1/t_{0} \cdot dt/dx = -1 / \Delta x$" + "\n" + r"$t_{0} = 2 \cdot \Delta x \cdot \cot(17 \pi / 180)$") # label = r"$1/t_{0}dt/dx = -1/\Delta x$\n$t_{0} = 2 \cdot \Delta x \cdot \cot(17 \pi/180)$")

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

    ax.annotate('', xy=(0, coords_x[0]), xytext=(0, coords_x_shift[0]),
                arrowprops=dict(arrowstyle='<->', color='black'))

    ax.annotate('', xy=(4.0, coords_x[0]), xytext=(4.0, 0),
                arrowprops=dict(arrowstyle='<->', color='black'))

    ax.annotate('', xy=(4.0, 0), xytext=(4.0, coords_x_shift[0]),
                arrowprops=dict(arrowstyle='<->', color='black'))

    # Annotations
    x_prime_mid_x = 0.35
    x_prime_mid_y = (coords_x[0] + coords_x_shift[0] ) / 2 
    ax.text(x_prime_mid_x, x_prime_mid_y+0.1, r'$x^{\prime}$', fontsize=13, ha='right', va='center', color='black')

    L_mid_x = 4.35
    L_mid_y = (coords_x[0]) / 2 # ) / 2
    ax.text(L_mid_x, L_mid_y, r'$L$', fontsize=13, ha='right', va='center', color='black')

    delta_x_mid_x = 4.60
    L_mid_y = (coords_x_shift[0]) / 2 # ) / 2
    ax.text(delta_x_mid_x, L_mid_y, r'$\Delta x$', fontsize=13, ha='right', va='center', color='black')

    # ax.set_ylim(-0.5, 0.5)
    # ax.set_xlim(-3, 5)

    # Save the figure
    # plt.grid(True)

    # Legend
    plt.legend(frameon=False, loc="upper right", fontsize=13, bbox_to_anchor=(0.4, 1.3))
    # bboxline_to_anchor=(0.5, 1.1))

    plt.axhline(y=coords_x[0], color='gray', linestyle='--', linewidth=1)
    plt.axhline(y=0, color='grey', linestyle='-', linewidth=1)
    plt.axhline(y=coords_x_shift[0], color='gray', linestyle='--', linewidth=1)
    # plt.axvline(x=0, color='gray', linestyle='--')

    plt.savefig(fout, dpi=300) # , bboxline_inches="tight")
    print("---> Written", fout)

    # Clear memory
    plt.close()

    return

def GetWedgeLength(t0, peak_angle):
    return (0.5*t0) / (math.tan(math.radians(peak_angle/2))) # inches

def WedgeGeom(triangle_height, base_length, side_height):
    # [peak, right upper, left upper, left lower, right lower, peak] 
    coords_z = np.array([0, base_length/2, base_length/2, -base_length/2, -base_length/2, 0])
    # coords_x = np.array([0, triangle_height, triangle_height+side_height, triangle_height+side_height, triangle_height, 0])
    coords_x = np.array([-triangle_height, 0, side_height, side_height, 0, -triangle_height])
    # x-coords need to be negative
    return  coords_z, coords_x # returns inches!

def Thickness(t0, x, dthickness_dx):
    return t0 + x * dthickness_dx

# def RunEremeyFormula(wedge_offset, t0): 
#     N = 100 
#     dthickness_dx = -1/wedge_offset 
#     # t0 = 2.51604 * wedge_offset # 2 * math.cos(math.radians(17))/math.sin(math.radians(17)) * wedge_offset
#     xline_ = np.linspace(wedge_offset, 3.15+wedge_offset, 2)
#     zline_ = np.array([Thickness(t0, x, dthickness_dx) for x in xline_])

#     print("---> dthickness_dx = ", dthickness_dx)

#     return xline_, zline_ 

def GetSlope(t0, wedge_length):
    t0 = 0.5 * t0 # For isoceles wedge
    dthickness_dx = t0 / wedge_length
    x2_edge = 0.0 + 1 # +1 for illustration
    x1_edge = -wedge_length
    print("dthickness_dx", dthickness_dx)
    xline_ = np.linspace(x2_edge, x1_edge, 2)
    zline_ = np.array([Thickness(t0, x, dthickness_dx) for x in xline_])
    # zline_ = zline_ / 2 
    return zline_, xline_ 
    # x1_edge < 0 < x2_edge
    # x1_edge = 0
    # x2_edge = wedge_length in the +x direction (with t0 at x2_edge)
    # t = t_0 + x * dthickness_dx
    # where t = 0 at x = x1_edge
    # So dthickness_dx = -t0 / x1_edge = -t0 / wedge_length
    # Then we just adjust the offset in the positive x direction with xline_offset.

def main():
    # Parameters 
    t0 = 5.12 # inches 
    peak_angle = 146 # degrees
    side_height = 2.37 # inches 

    # Wedge length
    wedge_length = GetWedgeLength(t0=t0, peak_angle=peak_angle)
    # print(wedge_length)

    x_offset = -0.5 # -5 / 25.4 # mm -> inches
    x_offset -= wedge_length

    # Wedge geometry, including box part 
    coords_z, coords_x = WedgeGeom(triangle_height=wedge_length, base_length=t0, side_height=side_height) # * 25.4
    coords_x = -coords_x


    # print(len(coords_x), len(coords_z))
    # Convert to mm
    # coords_x, coords_z = coords_x * 25.4, coords_z * 25.4

    # Get slope
    zline_, xline_ = GetSlope(t0=t0, wedge_length=wedge_length)
    xline_ = -xline_


    # print(zline_, xline_)
    # Convert to mm

    # Apply xline_offset
    coords_x_shift = coords_x + x_offset
    xline_shift_ = xline_ + x_offset
    
    # Plot
    Plot(coords_z, coords_x, zline_, xline_, coords_x_shift, xline_shift_, fout="../../Images/WedgeGeom/WedgeGeom_B4C.1.png")

    return

if __name__ == "__main__":
    main()

