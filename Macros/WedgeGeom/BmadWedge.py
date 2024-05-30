""" 
Samuel Grant
May 2024
Illustrate Bmad wedge
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

def Plot(coords_x, coords_y, xline_, yline_, title=None, xlabel="x", ylabel="t", fout="../../Images/WedgeGeom.png"):
    # Create figure and axes
    fig, ax = plt.subplots()

    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=coords_x[0], color='black', linestyle='--', linewidth=1)


    # Draw wedge
    ax.plot(coords_x, coords_y, "b-") 
    ax.fill(coords_x, coords_y, 'b', alpha=0.3)  
   
    # Plot line
    ax.plot(xline_, yline_, color="red", linestyle='-', linewidth=1.5, label=r"$t = t_{0} + x \frac{dt}{dx}$" + "\n" + r"$\frac{dt}{dx} = \frac{t_{0}}{L}$")
    # ax.plot(z2_, x2_, color="red", linestyle='-', linewidth=1.5, label = r"$1/t_{0} \cdot dt/dx = -1 / \Delta x$" + "\n" + r"$t_{0} = 2 \cdot \Delta x \cdot \cot(17 \pi / 180)$") # label = r"$1/t_{0}dt/dx = -1/\Delta x$\n$t_{0} = 2 \cdot \Delta x \cdot \cot(17 \pi/180)$")

    # Set title, xlabel, and ylabel
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel(xlabel, fontsize=13, labelpad=10) 
    ax.set_ylabel(ylabel, fontsize=13, labelpad=10) 

    # Set font size of tick labels on x and y axes
    ax.tick_params(axis='x', labelsize=13)  # Set x-axis tick label font size
    ax.tick_params(axis='y', labelsize=13)  # Set y-axis tick label font size
    # Removing the axis numbering
    plt.xticks([0])
    plt.yticks([])

    ax.set_ylim(-2.0, 11.0)
    ax.set_xlim(-30, 2.0)
    
    ax.set_aspect('equal')
    # Legend
    plt.legend(frameon=False, loc="best", fontsize=13, bbox_to_anchor=(0.5, 0.5))

    # Adding the arrows and labels for t_0 and x_0
    # t_0 arrow
    # ax.annotate(r'$t_0$', xy=(coords_x[2], coords_y[1:2]), xytext=(coords_x[2] - 0.5, (coords_y[1] + coords_y[0]) / 2),
    #             arrowprops=dict(arrowstyle='<->', color='black'),
    #             fontsize=13, ha='right', va='center')

    ax.annotate('', xy=(0.5, 0), xytext=(0.5, coords_y[1]),
                arrowprops=dict(arrowstyle='<->', color='black'))

    ax.annotate('', xy=(coords_x[0], -0.5), xytext=(0, -0.5),
                arrowprops=dict(arrowstyle='<->', color='black'))

    # Annotate in the middle of the arrows
    # t_0 annotation
    t0_mid_x = 0.5
    t0_mid_y = (coords_y[1] + 0) / 2
    ax.text(t0_mid_x+1.2, t0_mid_y, r'$t_0$', fontsize=13, ha='right', va='center', color='black')

    # x_0 annotation
    x0_mid_x = (coords_x[0] + 0) / 2
    x0_mid_y = -0.5
    ax.text(x0_mid_x, x0_mid_y-0.2, r'$L$', fontsize=13, ha='center', va='top', color='black')

    # Adding labels next to the vertical lines
    plt.text(coords_x[0], plt.ylim()[1] * 1.05, 'x1_edge', horizontalalignment='center', color='black')
    plt.text(0, plt.ylim()[1] * 1.05, 'x2_edge', horizontalalignment='center', color='black')

    # # x_0 arrow
    # ax.annotate(r'$L$', xy=(coords_x[0], coords_y[0]), xytext=((coords_x[0] + coords_x[-1]) / 2, coords_y[0] - 0.5),
    #             arrowprops=dict(arrowstyle='<->', color='black'),
    #             fontsize=13, ha='center', va='top')


    # plt.tight_layout()

    plt.savefig(fout, dpi=300) #  layout="tight")
    print("---> Written", fout)

    # Clear memory
    plt.close()

    return

def GetWedgeLength(t0, peak_angle):
    return (0.5*t0) / (math.tan(peak_angle/2)) # mm

def WedgeGeom(triangle_height, base_length):
    # [tip, right upper, right lower, tip] 
    coords_x = np.array([-triangle_height, 0, 0, -triangle_height])
    # coords_x = np.array([0, triangle_height, triangle_height+side_height, triangle_height+side_height, triangle_height, 0])
    # coords_x = np.array([0, triangle_height, 0, 0])
    coords_y = np.array([0, base_length, 0, 0])# triangle_height, 0, 0])
    # x-coords need to be negative
    return  coords_x, coords_y

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
    dthickness_dx = t0 / wedge_length
    x2_edge = 0.0 
    x1_edge = -wedge_length
    # print("dthickness_dx", dthickness_dx)
    xline_ = np.linspace(x2_edge, x1_edge, 2)
    yline_ = np.array([Thickness(t0, x, dthickness_dx) for x in xline_])
    # zline_ = zline_ / 2 
    return xline_, yline_ 
    # x1_edge < 0 < x2_edge
    # x1_edge = 0
    # x2_edge = wedge_length in the +x direction (with t0 at x2_edge)
    # t = t_0 + x * dthickness_dx
    # where t = 0 at x = x1_edge
    # So dthickness_dx = -t0 / x1_edge = -t0 / wedge_length
    # Then we just adjust the offset in the positive x direction with xline_offset.

def main():
    # Parameters 
    t0 = 10 # mm 
    peak_angle = math.radians(20) # radians

    # Wedge length
    wedge_length = GetWedgeLength(t0=t0, peak_angle=peak_angle)
    print(wedge_length)

    x_offset = 0 # mm -> inches
    # x_offset -= wedge_length

    # Wedge geometry, including box part 
    coords_x, coords_y = WedgeGeom(triangle_height=wedge_length, base_length=t0) # * 25.4
    # coords_x = -coords_x
    # print(len(coords_x), len(coords_z))
    # Convert to mm
    # coords_x, coords_z = coords_x * 25.4, coords_z * 25.4

    # Get slope
    xline_, yline_ = GetSlope(t0=t0, wedge_length=wedge_length)
    # xline_ = -xline_
    # zline_ = -zline_

    # print(zline_, xline_)
    # Convert to mm

    # Apply xline_offset
    coords_x += x_offset
    xline_ += x_offset
    
    # Plot
    Plot(coords_x, coords_y, xline_, yline_, fout="../../Images/WedgeGeom/BmadWedge.png")

    return

if __name__ == "__main__":
    main()

