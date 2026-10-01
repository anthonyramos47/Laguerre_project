# This file is a template for creating a new python file in the hananLab/hanan directory

# Import the necessary libraries
import os
import sys
import argparse
import pickle
from pathlib import Path

# Obtain the path HananLab; this is the parent directory of the hananLab/hanan directory
# <Here you can manually add the path direction to the hananLab/hanan directory>
path = "/path/to/hanan"
sys.path.append(path)

# Verification of the path
print(path)

# Import the necessary libraries for visualization and computation
# Import the necessary libraries for visualization and computation
import igl
import polyscope as ps
import polyscope.imgui as psim
import numpy as np

import matplotlib.pyplot as plt
import splipy as sp
import json
from scipy.optimize import minimize
from scipy.interpolate import bisplev

# Import the necessary classes and functions from the hananLab/hanan directory

# Geometry classes
from geometry.mesh import Mesh
from geometry.utils import *
from utils.bsplines_functions import *

# Optimization classes
from energies.BS_approx import BS_approx
from optimization.Optimizer import Optimizer


# Here you can add the directory where you want to save the results
dir_path = os.getcwd()
print(dir_path)

# Define Bsplines Surface directory
surface_dir = os.path.join(dir_path, "data", "Bsplines_Surfaces")

print("surface dir:", surface_dir)

experiment_dir = os.path.join(dir_path, "experiments")


# Create the parser
parser = argparse.ArgumentParser(description="Curvature Vis")
# Add an argument
parser.add_argument('file_name', type=str, help='File name to load')

bspline_surf_name = parser.parse_args().file_name

# Define the path to the B-spline surface
bspline_surf_path = os.path.join(surface_dir, bspline_surf_name + ".json")
print("bspline_surf_path:", bspline_surf_path)

# Sample in each direction
sample = 50

# Load the B-spline surface
control_points, knots_u, knots_v, order_u, order_v = read_bspline_json(bspline_surf_path)


# Create the B-splines basis
basis_u = sp.BSplineBasis(order_u, knots_u) 
basis_v = sp.BSplineBasis(order_v, knots_v) 

# Create the B-spline surface
bsp1 = sp.Surface(basis_u, basis_v, control_points)


# Sample the grid points to evaluate the B-spline surface
u_vals, v_vals = sample_grid(sample, sample, deltaum=0.01, deltauM=0.01, deltavm = 0.01, deltavM = 0.01)
# Compute the curvature
_, H, _ = curvatures_par(bsp1, u_vals, v_vals)
H = H.flatten()

# Count positive H 
num_pos =  len(np.where(H > 0)[0])

# Count negative H
num_neg = len(np.where(H < 0)[0])


cp_flat = control_points.flatten()

# epsilon 
epsilon = 1/(0.5*np.linalg.norm([np.min(cp_flat, axis=0), np.max(cp_flat, axis=0)]))

print("Epsilon: ", epsilon)
print("Radius/2: ", 0.5*np.linalg.norm([np.min(cp_flat, axis=0), np.max(cp_flat, axis=0)]))

original_cp_flat = cp_flat.copy()


dummy = 0.2*np.ones_like(H)

x0 = np.hstack((cp_flat, dummy))

if num_pos > num_neg:

    def objective_function(cp_flat, original_cp_flat, weight):
        cp_pts = cp_flat[:len(original_cp_flat)]
        # Objective function to minimize
        

        # For example, weighted distance from the original control points
        return weight * np.sum((cp_pts - original_cp_flat) ** 2)

    
    def constraint_function(cp_flat):
        #size = len(original_cp_flat)
        cp_pts = cp_flat[:len(original_cp_flat)]
        dummy  = cp_flat[len(original_cp_flat):]

        # Reconstruct the control points from the flattened array
        control_points_reshaped = cp_pts.reshape(control_points.shape)

        # Recreate the B-spline surface with the new control points
        bsp1_new = sp.Surface(basis_u, basis_v, control_points_reshaped)
        
        # Compute the curvature with the new control points
        _, H, _ = curvatures_par(bsp1_new, u_vals, v_vals)

        
        # Ensure H is positive
        return (H.flatten() - dummy**2 - epsilon) # Subtract a small value to ensure positivity

    # Define the constraint in the form expected by scipy.optimize.minimize
    constraints = {'type': 'ineq', 
                   'fun': constraint_function
                   }
    # Run the optimization

    result = minimize(objective_function, x0, args=( original_cp_flat, 1e-7), constraints=constraints) 

    # Check if the optimization was successful
    if result.success:
        optimized_control_points = result.x[:len(original_cp_flat)].reshape(control_points.shape)
        print(result)
        print("Optimization successful.")
        print("Optimized control points:", optimized_control_points)
    else:
        print("Optimization failed:", result.message)

else:
    def objective_function(cp_flat, original_cp_flat, weight):
        size = len(original_cp_flat)
         # Objective function to minimize
        # For example, weighted distance from the original control points
        return weight * np.sum((cp_flat[:size] - original_cp_flat) ** 2)

    
    def constraint_function(cp_flat):
        size = len(original_cp_flat)

        # Reconstruct the control points from the flattened array
        control_points_reshaped = cp_flat[:size].reshape(control_points.shape)

        # Recreate the B-spline surface with the new control points
        bsp1_new = sp.Surface(basis_u, basis_v, control_points_reshaped)
        
        # Compute the curvature with the new control points
        _, H, _ = curvatures_par(bsp1_new, u_vals, v_vals)
        
        # Ensure H is positive
        return (-H.flatten() - cp_flat[size:]**2 - epsilon) # Subtract a small value to ensure positivity

    # Define the constraint in the form expected by scipy.optimize.minimize
    constraints = {'type': 'ineq', 
                   'fun': constraint_function
                   }
    # Run the optimization

    result = minimize(objective_function, x0, args=( original_cp_flat, 1e-7), constraints=constraints) 

    # Check if the optimization was successful
    if result.success:
        optimized_control_points = result.x[:len(original_cp_flat)].reshape(control_points.shape)
        print(result)
        print("Optimization successful.")
        print("Optimized control points:", optimized_control_points)
    else:
        print("Optimization failed:", result.message)



#optimized_control_points = result.x.reshape(control_points.shape)
# Create the B-spline surface
bsp1 = sp.Surface(basis_u, basis_v, optimized_control_points)

# Evaluate the B-spline surface at the u and v points
delta_u_m = 0.05 
delta_u_M = 0.05  
delta_v_m = 0.05
delta_v_M = 0.05


def curvature_sel():
    global bsp1, sample, delta_u_m, delta_u_M, delta_v_m, delta_v_M, u_vals, v_vals, surf

    _, delta_u_m = psim.SliderFloat("deltaumin", delta_u_m, 0.05, 1)
    _, delta_u_M = psim.SliderFloat("deltaumax", delta_u_M, 0.05, 1)
    _, delta_v_m = psim.SliderFloat("deltavmin", delta_v_m, 0.05, 1)
    _, delta_v_M = psim.SliderFloat("deltavmax", delta_v_M, 0.05, 1)
  

    if psim.Button("Draw Surface"):
        # Sample the grid points to evaluate the B-spline surface
        u_vals, v_vals = sample_grid(sample, sample, deltaum=delta_u_m, deltauM=delta_u_M, deltavm = delta_v_m, deltavM = delta_v_M)

        # Evaluate the B-spline surface at the u and v points
        V,F =Bspline_to_mesh(bsp1, u_vals, v_vals)

        # Create the surface mesh
        surf = ps.register_surface_mesh("mesh", V, F)

        # Compute the curvature
        K, H, _ = curvatures_par(bsp1, u_vals, v_vals)

        # Flatten the curvature values
        H = H.flatten()
        K = K.flatten()
        valid = np.zeros_like(H)

        # Define number of positive and negative values
        idx = np.where(H < 0)[0]

        valid[idx] = 1
        
        surf.add_scalar_quantity("Mean Curvature", H, enabled=True)
        surf.add_scalar_quantity("Gaussian Curvature", K, enabled=False)
        surf.add_scalar_quantity("Near Vanishing Curv", valid, enabled=True)


    if psim.Button("Save"):
         # Variable to save
        save_data = {
                    'surf': bsp1
                    }
        
        save_file_path = os.path.join(experiment_dir, bspline_surf_name  + "_init.pickle")

        # Save the variable to a file
        with open(save_file_path, 'wb') as file:
            pickle.dump(save_data, file)

        ps.warning("Results saved in: " + save_file_path)

ps.init()

ps.set_user_callback(curvature_sel)
ps.show()
