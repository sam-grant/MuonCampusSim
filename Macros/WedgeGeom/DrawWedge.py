# import matplotlib.pyplot as plt
# import numpy as np
# import math

# Function to plot the outer outline of the shape
# def plot_outer_outline(base_length, side_height, peak_angle):

#     # Calculate dimensions for the triangle
#     triangle_height = 2.56 / (math.tan(math.radians(peak_angle / 2)))

#     # Coordinates for the outer outline
#     x_outer = [0, base_length, base_length, base_length / 2, 0, base_length / 2]
#     y_outer = [0, 0, side_height, side_height + triangle_height, side_height, side_height]

#     x_outer = [0, base_length, base_length, base_length / 2, 0, base_length / 2]
#     y_outer = [0, 0, side_height, side_height + triangle_height, side_height, side_height]


#     # Plot the outer outline
#     plt.plot(x_outer, y_outer, 'b-')

#     # Set aspect ratio to equal
#     plt.gca().set_aspect('equal', adjustable='box')

#     # Set labels and title
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Outer Outline of the Shape')

#     # Show plot
#     plt.grid(True)
#     plt.show()

# # Define parameters
# base_length = 5.12
# side_height = 2.37
# peak_angle = 146

# # Plot the outer outline of the shape
# plot_outer_outline(base_length, side_height, peak_angle)



import matplotlib.pyplot as plt
import numpy as np
import math

# Function to plot an equilateral pentagon
def Wedge(wedge_offset=0., base_length=5.12, side_height=2.37, peak_angle=146):

    # Calculate dimensions for the triangle
    triangle_height = 2.56 / (math.tan(math.radians(peak_angle / 2)))

    # [left lower, left upper, peak, right upper, right lower, left lower]
    # coords_t = np.array([0-(base_length / 2), 0-(base_length / 2), 0, 0+(base_length / 2), 0+(base_length / 2), 0-(base_length / 2)])
    # coords_x = np.array([0, side_height, side_height+triangle_height, side_height, 0, 0])

    # coords_t = np.array([0-(base_length / 2), 0-(base_length / 2), 0, 0+(base_length / 2), 0+(base_length / 2), 0-(base_length / 2)])
    # coords_x = np.array([0, side_height, side_height+triangle_height, side_height, 0, 0])

    # [peak, right upper, left upper, left lower, right lower, peak] 
    coords_x = np.array([wedge_offset, triangle_height+wedge_offset, triangle_height+wedge_offset+side_height, triangle_height+wedge_offset+side_height, triangle_height+wedge_offset, wedge_offset])
    coords_z = np.array([0, base_length/2, base_length/2, -base_length/2, -base_length/2, 0])


    plt.plot(coords_x, coords_z, "b-")
    plt.fill(coords_x, coords_z, 'b', alpha=0.3)    # Fill triangle
    # Set aspect ratio to equal
    plt.gca().set_aspect('equal', adjustable='box')

    # Set labels and title
    plt.xlabel('x [in]')
    plt.ylabel('z [in]')
    plt.title('Wedge')
    # Show plot
    plt.grid(True)
    plt.show()

    return coords_x, coords_z

# Define parameters
offset = 5 / 25.4
base_length = 5.12
side_height = 2.37
peak_angle = 146


# Plot the equilateral pentagon
Wedge(offset, base_length, side_height, peak_angle)
