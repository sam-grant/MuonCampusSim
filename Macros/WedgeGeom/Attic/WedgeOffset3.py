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

def Plot(coords_z, coords_x, zline_, xline_, title=None, xlabel="z [mm]", ylabel="x [mm]", fout="../../Images/WedgeGeom.png"):

    # Create figure and axes
    fig, ax = plt.subplots()

    # # Draw wedges
    ax.plot(coords_z, coords_x, "b-") # lpha=0.3) 
    ax.fill(coords_z, coords_x, 'b', alpha=0.3)  

    # ax.plot(coords_z, coords_x_shift, "b-") 
    # ax.fill(coords_z, coords_x_shift, 'b', alpha=0.3)  

    # Plot line
    # ax.plot(zline_, xline_, color="red", linestyle='-', linewidth=1.5) # , label=r"$x = x_{0} + z \frac{dx}{dz}$")
    ax.plot(zline_, xline_, color="red", linestyle='-', linewidth=1.5, label=r"$t = t_{0} + x \frac{dt}{dx}$" + "\n" + r"$\frac{dt}{dx} = \frac{t_{0}}{L}$")
    # ax.plot(z2_, x2_, color="red", linestyle='-', linewidth=1.5, label = r"$1/t_{0} \cdot dt/dx = -1 / \Delta x$" + "\n" + r"$t_{0} = 2 \cdot \Delta x \cdot \cot(17 \pi / 180)$") # label = r"$1/t_{0}dt/dx = -1/\Delta x$\n$t_{0} = 2 \cdot \Delta x \cdot \cot(17 \pi/180)$")

    # Set title, xlabel, and ylabel
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel(xlabel, fontsize=13, labelpad=10) 
    ax.set_ylabel(ylabel, fontsize=13, labelpad=10) 

    # Set font size of tick labels on x and y axes
    ax.tick_params(axis='x', labelsize=13)  # Set x-axis tick label font size
    ax.tick_params(axis='y', labelsize=13)  # Set y-axis tick label font size

    # # Scientific notation
    # if ax.get_xlim()[1] > 9999 or ax.get_xlim()[1] < 9.999e-3:
    #     ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #     ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    #     ax.xaxis.offsetText.set_fontsize(13)
    # if ax.get_ylim()[1] > 9999 or ax.get_ylim()[1] < 9.999e-3:
    #     ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    #     ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    #     ax.yaxis.offsetText.set_fontsize(13)

    # ax.set_aspect('equal')

    # ax.annotate('', xy=(0, coords_x[0]), xytext=(0, coords_x_shift[0]),
    #             arrowprops=dict(arrowstyle='<->', color='black'))

    # ax.annotate('', xy=(4.0, coords_x[0]), xytext=(4.0, 0),
    #             arrowprops=dict(arrowstyle='<->', color='black'))

    # ax.annotate('', xy=(4.0, 0), xytext=(4.0, coords_x_shift[0]),
    #             arrowprops=dict(arrowstyle='<->', color='black'))

    # # Annotations
    # x_prime_mid_x = 0.35
    # x_prime_mid_y = (coords_x[0] + coords_x_shift[0] ) / 2 
    # ax.text(x_prime_mid_x, x_prime_mid_y+0.1, r'$x^{\prime}$', fontsize=13, ha='right', va='center', color='black')

    # L_mid_x = 4.35
    # L_mid_y = (coords_x[0]) / 2 # ) / 2
    # ax.text(L_mid_x, L_mid_y, r'$L$', fontsize=13, ha='right', va='center', color='black')

    # delta_x_mid_x = 4.60
    # L_mid_y = (coords_x_shift[0]) / 2 # ) / 2
    # ax.text(delta_x_mid_x, L_mid_y, r'$\Delta x$', fontsize=13, ha='right', va='center', color='black')

    ax.set_ylim(-28, 1)
    ax.set_xlim(-10, 135)

    # Save the figure
    # plt.grid(True)

    # delta x
    ax.annotate('', xy=(0, 0), xytext=(0, -5),
                arrowprops=dict(arrowstyle='<->', color='black'))

    x_offset_mid_x = 7.5
    x_offset_mix_y = -2.5
    ax.text(x_offset_mid_x, x_offset_mix_y, r'$\Delta x$', fontsize=13, ha='right', va='center', color='black')

    # t0
    ax.annotate('', xy=(0, coords_x[1]-0.5), xytext=(coords_z[1], coords_x[1]-0.5),
                arrowprops=dict(arrowstyle='<->', color='black'))

    t0_mid_x = coords_z[1]/2
    t0_mix_y = coords_x[1]-1.5
    ax.text(t0_mid_x, t0_mix_y, r'$t_{0}$', fontsize=13, ha='right', va='center', color='black')

    # t0
    ax.annotate('', xy=(0, coords_x[1]-0.5), xytext=(coords_z[1], coords_x[1]-0.5),
                arrowprops=dict(arrowstyle='<->', color='black'))

    t0_mid_x = coords_z[1]/2
    t0_mix_y = coords_x[1]-1.5
    ax.text(t0_mid_x, t0_mix_y, r'$t_{0}$', fontsize=13, ha='right', va='center', color='black')

    # L 
    ax.annotate('', xy=(-1.5, -5), xytext=(-1.5, coords_x[1]),
                arrowprops=dict(arrowstyle='<->', color='black'))

    L_mid_x = -3.0
    L_mix_y = (-5 + coords_x[1]) / 2
    ax.text(L_mid_x, L_mix_y, r'$L$', fontsize=13, ha='right', va='center', color='black')


# coords_z, coords_x

#     x_offset_mid_x = 22.5
#     x_offset_mix_y = -2.5
#     ax.text(coords_z, x_offset_mix_y, r'$\Delta x$', fontsize=13, ha='right', va='center', color='black')


    # Legend
    plt.legend(frameon=False, loc="center right", fontsize=13, bbox_to_anchor=(0.85, 0.55))
    # bboxline_to_anchor=(0.5, 1.1))

    plt.axhline(y=-5, color='gray', linestyle='--', linewidth=1)
    plt.axhline(y=0, color='grey', linestyle='--', linewidth=1)
    # plt.axhline(y=coords_x_shift[0], color='gray', linestyle='--', linewidth=1)
    # plt.axvline(x=0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.savefig(fout, dpi=300) # , bboxline_inches="tight")
    print("---> Written", fout)

    # Clear memory
    plt.close()

    return

def GetWedgeLength(t0, peak_angle):
    return (0.5*t0) / (math.tan(math.radians(peak_angle/2))) # inches

def WedgeGeom(triangle_height, base_length):
    # [tip, right upper, right lower, tip] 
    coords_x = np.array([0, base_length, 0, 0])
    coords_x = np.array([0, 0, 0, base_length])
    # coords_x = np.array([0, triangle_height, triangle_height+side_height, triangle_height+side_height, triangle_height, 0])
    # coords_x = np.array([0, triangle_height, 0, 0])
    coords_y = np.array([0, -triangle_height, -triangle_height, 0])# triangle_height, 0, 0])
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

def GetSlope(t0, wedge_length, x_offset):
    # t0 = 0.5 * t0 # For isoceles wedge
    dthickness_dx = t0 / wedge_length
    # +1 for illustration
    x1_edge = 0.0 # + x_offset
    x2_edge = -wedge_length # + x_offset 

    print(x1_edge, x2_edge)
    print("dthickness_dx", dthickness_dx)
    xline_ =  np.linspace(x1_edge, x2_edge, 2)
    zline_ = np.array([Thickness(t0, x, dthickness_dx) for x in xline_])
    # zline_ = zline_ / 2 
    return zline_, (xline_ + x_offset)
    # x1_edge < 0 < x2_edge
    # x1_edge = 0
    # x2_edge = wedge_length in the +x direction (with t0 at x2_edge)
    # t = t_0 + x * dthickness_dx
    # where t = 0 at x = x1_edge
    # So dthickness_dx = -t0 / x1_edge = -t0 / wedge_length
    # Then we just adjust the offset in the positive x direction with xline_offset.

def main():
    # Parameters 
    t0 = 5.12 * 24.5 # inches 
    peak_angle = 146 # degrees

    # Wedge length
    wedge_length = (3.15-2.37) * 24.5 # GetWedgeLength(t0=t0, peak_angle=peak_angle)
    # wedge_length *= 24.5 
    # print(wedge_length)
    # print(wedge_length)

    x_offset = -5 # -5 / 25.4 # mm -> inches
    # x_offset -= wedge_length

    # Wedge geometry, including box part 
    coords_z, coords_x = WedgeGeom(triangle_height=wedge_length, base_length=t0) 


    # print(len(coords_x), len(coords_z))
    # Convert to mm
    coords_x, coords_z = coords_x, coords_z

    # Get slope
    zline_, xline_ = GetSlope(t0=t0, wedge_length=wedge_length, x_offset=(x_offset))
    # zline_, xline_ = zline_* 25.4, xline_* 25.4
    
    # Apply rotation 
    # xline_ = np.flip(xline_) # .reverse() # (xline_)
    
    # xline_ = xline_
    # xline_ += x_offset
    coords_x += x_offset

    # coords_x = np.flip(coords_x)
    
    # Plot
    # Plot(coords_z, coords_x, zline_, xline_, title=r"B4C wedge with x-offset, rotated $180^{\circ}$ about y", fout="../../Images/WedgeGeom/WedgeGeom_B4C.2.png")
    Plot(coords_z, coords_x, zline_, xline_, title=r"B4C wedge with x-offset, no rotation", fout="../../Images/WedgeGeom/WedgeGeom_B4C.3.png")

    return

if __name__ == "__main__":
    main()

