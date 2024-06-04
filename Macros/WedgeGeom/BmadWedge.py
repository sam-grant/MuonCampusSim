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

def Plot(wedge_x, wedge_t, box_x, box_t, xline_, tline_, x_offset=0, delta_x=0, title=None, xlabel=r"$x_{\mathrm{body}}$", ylabel="t", fout="../../Images/WedgeGeom.png"):
    # Create figure and axes
    fig, ax = plt.subplots()

    # Draw wedge
    ax.plot(wedge_x, wedge_t, "b-") 
    ax.fill(wedge_x, wedge_t, 'b', alpha=0.3)  
   
    ax.plot(box_x, box_t, "b-") 
    ax.fill(box_x, box_t, 'b', alpha=0.3) 

    # Plot line
    ax.plot(xline_, tline_, color="red", linestyle='-', linewidth=1.5, label=r"$t = t_{0} + x \frac{dt}{dx}$" + "\n" + r"$\frac{dt}{dx} = -\frac{t_{0}}{L}$")

    # Set title, xlabel, and ylabel
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel(xlabel, fontsize=13, labelpad=10) 
    ax.set_ylabel(ylabel, fontsize=13, labelpad=10) 

    # Set font size of tick labels on x and y axes
    ax.tick_params(axis='x', labelsize=13)  
    ax.tick_params(axis='y', labelsize=13)  
    
    # Remove the axis numbering
    plt.xticks([0])
    plt.yticks([])

    ax.set_ylim(wedge_t[0]-2.0, wedge_t[1]+2.0)
    # ax.set_xlim(box_x[0]-2.65, wedge_x[0]+2.65) 
    ax.set_xlim(wedge_x[1]-5.0, wedge_x[0]+2.65) 

    ax.set_aspect('equal')
    
    # Legend
    plt.legend(frameon=False, loc="upper center", fontsize=13)# , bbox_to_anchor=(1.0, 1.0))

    # Annotations
    plt.axvline(x=x_offset, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=wedge_x[0], color='black', linestyle='--', linewidth=1)
    plt.axvline(x=box_x[0], color='black', linestyle='--', linewidth=1)
    
    # t0 line
    ax.annotate('', xy=(-.5+x_offset, 0), xytext=(-0.5+x_offset, wedge_t[1]),
                arrowprops=dict(arrowstyle='<->', color='black'))

    # ax.annotate('', xy=(wedge_x[0]+x_offset, 0), xytext=(wedge_x[0]+x_offset, wedge_t[1]),
    #             arrowprops=dict(arrowstyle='<->', color='black'))

    # L line
    ax.annotate('', xy=(x_offset, -0.5), xytext=(wedge_x[0], -0.5),
                arrowprops=dict(arrowstyle='<->', color='black'))



    # Annotate in the middle of the arrows
    # t_0 annotation
    t0_mid_x = x_offset - 0.80
    t0_mid_y = ((wedge_t[0] + wedge_t[1]) / 2) 

    ax.text(t0_mid_x, t0_mid_y-0.2, r'$t_0$', fontsize=13, ha='right', va='center', color='black')

    # L annotation
    x0_mid_x = ((wedge_x[0] + x_offset) / 2) 
    x0_mid_y = -0.5
    ax.text(x0_mid_x, x0_mid_y-0.2, r'$L$', fontsize=13, ha='center', va='top', color='black')

    # Adding labels next to the vertical lines
    plt.text(x_offset, plt.ylim()[1] * 1.05, 'x1_edge', horizontalalignment='center', color='black')
    plt.text(wedge_x[0], plt.ylim()[1] * 1.05, 'x2_edge', horizontalalignment='center', color='black')

    # Delta X 
    if x_offset != 0: 
        # delta_x = wedge_t[1] - x_offset
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.annotate('', xy=(0, wedge_t[1]), xytext=(delta_x, wedge_t[1]),
                    arrowprops=dict(arrowstyle='<->', color='black'))
        dx_mid_x = ((0 + delta_x) / 2) 
        dx_mid_y = wedge_t[1]-0.35
        ax.text(dx_mid_x, dx_mid_y, r'$\Delta x$', fontsize=13, ha='center', va='top', color='black')

    plt.tight_layout()
    plt.savefig(fout, dpi=300) #  layout="tight")
    print("---> Written", fout)

    # Clear memory
    plt.close()

    return

def GetWedgeLength(t0, peak_angle):
    return (0.5*t0) / (math.tan(peak_angle/2)) # mm

def WedgeGeom(triangle_height, base_length):
    # [tip, right upper, right lower, tip] 
    coords_x = np.array([triangle_height, 0, 0, triangle_height])
    coords_y = np.array([0, base_length, 0, 0])
    return  coords_x, coords_y

def BoxGeom(box_height, box_length):
    # [tip, right upper, right lower, tip] 
    coords_x = np.array([-box_height, 0, 0, -box_height, -box_height])
    coords_y = np.array([box_length, box_length, 0, 0, box_length])
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
    dthickness_dx = -t0 / wedge_length
    x1_edge = 0.0 
    x2_edge = wedge_length
    # print("dthickness_dx", dthickness_dx)
    xline_ = np.linspace(x1_edge, x2_edge, 2)
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
    box_length = (2.37) * 24.5
    # print(wedge_length)

    delta_x = 10 
    # x_offset = 10 # mm -> inches
    x_offset = delta_x - wedge_length

    # Wedge geometry
    wedge_x, wedge_t = WedgeGeom(triangle_height=wedge_length, base_length=t0) # * 25.4
    box_x, box_t = BoxGeom(box_height=box_length, box_length=t0)

    # Get slope
    xline_, tline_ = GetSlope(t0=t0, wedge_length=wedge_length)



    # Plot
    Plot(wedge_x, wedge_t, box_x, box_t, xline_, tline_, fout="../../Images/WedgeGeom/FullBmadWedge.png")


    # Apply xline_offset
    wedge_x += x_offset
    box_x += x_offset
    xline_ += x_offset

    # Plot(wedge_x, wedge_t, box_x, box_t, xline_, tline_, x_offset, xlabel=r"$x_{\mathrm{lab}} = x_{\mathrm{body}} - L$", fout="../../Images/WedgeGeom/FullBmadWedgeMinusL.png")

    Plot(wedge_x, wedge_t, box_x, box_t, xline_, tline_, x_offset, delta_x, xlabel=r"$x_{\mathrm{lab}} = x_{\mathrm{body}} + x_{\mathrm{offset}} = x_{\mathrm{body}} + (\Delta X - L)$", fout="../../Images/WedgeGeom/FullBmadWedgeWithOffset.png")


    return

if __name__ == "__main__":
    main()

