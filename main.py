import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from types import SimpleNamespace
import datetime
import time as timer

from unicycle_nominal_controller import nominal_control, reference_path, saturation
from unicycle_zocbf import ZOCBF
from unicycle_model import UnicycleModel

import argparse
import sys

# Create the parser
parser = argparse.ArgumentParser(description="accepting command-line arguments.")

# Add arguments
parser.add_argument("--use_filter", type=bool, help="Use safety filter?",default=False)

# Parse arguments
args = parser.parse_args()

# use safety filter? By default it is False.
use_filter = args.use_filter
# Does attack exist?
exist_attack = True


x0 = np.array([-10,0,-0.1])
T = 0.01
l = 25 # steps
total_time = 20 # seconds
input_bounds = np.array([[-5, 5], [-2, 2]])

s = 2
sensors = {'x_pos':0,'y_pos':1,'radius':2,'bearing':3,'heading':4}
attacked_index = [0,1] # index at your choice

# unicycle model
model_param = SimpleNamespace(sample_time = T,input_bounds = input_bounds,data_length = l+1, integration_method = 'RK45',
                              attack_count = s,sensor_dict = sensors,consistency_err = 1e-2)
unicyle = UnicycleModel(parameters=model_param)

# environment parameters
# reference trajectory x = p0 + vx*t, y = a*sin(bx+c) + d 
reference_param = SimpleNamespace(a = 1.8, b = np.pi/5, c = 0.0, d = 1, p0 = -10, vx = 1.0)

# zocbf parameters
zocbf_param = SimpleNamespace(radius = 0.4,amplitude=reference_param.a,frequency= reference_param.b,
                              phase_shift = reference_param.c, y_shift = reference_param.d, 
                              gamma = 1*T, eps = 0.01,input_bounds = input_bounds)

zocbf = ZOCBF(unicyle.dt_dynamics,zocbf_param)
## set ad hoc safety indicating function 
def safe_func(self,x):
    return 3**2 - (x[1])**2

zocbf.h = safe_func.__get__(zocbf,ZOCBF)



# one-step safety filter test
u_nom = nominal_control(0,x0)
zocbf.u_guess = u_nom
u_safe = zocbf.safety_filter(0,x0,u_nom)
x_next = unicyle.dt_dynamics(0,x0,u_safe)

print(f'linear velocity correction:  {u_nom[0]} -> {u_safe[0]}')
print(f'angular velocity correction: {u_nom[1]} -> {u_safe[1]}')

print(x_next)

# close the loop
# sim with safety filter
state = x0
states = [state]

if exist_attack:
    fake_state = np.array([-10,0,0.1])
    fake_states = [fake_state]
    measurement = unicyle.measure(state)
    measurements = [measurement]

nom_controls = []
act_controls = []
controls_filtered = []

h_values = []

timesteps = int(total_time / T)
u_prev = np.zeros(2)

print('\n--------------------start closed-loop simulation------------------')
print(f'safety filter: {use_filter}, attacks exist: {exist_attack},total iteration: {timesteps}')
if exist_attack:
    print(f'number of attacks: {s}, attacked index:{attacked_index}')
    print(f'x0:{x0}, fake x0:{fake_state}')
print('------------------------------------------------------------------\n')

tic = timer.time()
for i in range(timesteps):
    time = i*T

    # generate nominal input in different cases
    x_ref = reference_path(time,reference_param)

    if exist_attack:
        if use_filter:
            # print(f'\n timestep:{i},state:{state},fake_state:{fake_state}')
            x_set = unicyle.get_plausible_state(i,np.array(act_controls),np.array(measurements))
        else:
            # for testing naive approach
            # x_set = measurement[[0,1,4]]
            x_set = np.array([state, fake_state])
            # print(f'x_set:{x_set}')

        x_est = unicyle.get_average_state(x_set)
        if x_est.size != 3:
            raise ValueError(f"Error at time {i}: x_est has unexpected size:{x_est}")
        
        # to generate a good reference control
        # x_set_omni = np.array([state, fake_state])
        # x_est_omni = unicyle.get_average_state(x_set_omni)
        u_nom = nominal_control(time, fake_state,xg = x_ref,kr = 1.5,ka = 1.5)
    else:
        x_set = state # for uniform in safety filter
        u_nom = nominal_control(time, state,xg = x_ref,kr = 1.0,ka = 1.0)
    u_nom = saturation(u_nom,input_bounds)

    # generate actual input in different cases
    if use_filter:
        # safety filter, ZOCBF
        zocbf.u_guess = u_prev
        control_actual = zocbf.safety_filter(time, x_set, u_nom)
        # control_actual = u_nom
        # print(f'x_set:{x_set}')
        # next state with safe control input
        controls_filtered.append(control_actual)
        u_prev = control_actual # update u_prev
    else:
        control_actual = u_nom
    
    # generate real and fake next states and corresponding measurement
    state_next = unicyle.dt_dynamics(time,state,control_actual)
    if exist_attack:
        # fake state update
        fake_state_next = unicyle.dt_dynamics(time,fake_state,control_actual)
        measurement = unicyle.measure(state_next,fake_state_next,attacked_index)
        measurements.append(measurement)
        fake_states.append(fake_state)
        fake_state = fake_state_next

    # logging
    h_val = zocbf.h(state) # for plotting
    states.append(state_next)
    nom_controls.append(u_nom)
    act_controls.append(control_actual)
    h_values.append(float(h_val)) 

    # ground-true
    state = state_next

    if (i + 1) % (timesteps/10) == 0:
        toc = timer.time()
        print(f"\nProgress: {i + 1}/{timesteps} iterations completed ({(i + 1)/timesteps*100:.1f}%) Elapsed time: {toc-tic:.1f} seconds.")
        if exist_attack:
            print(f'measurement:{measurement}')
            print(f'x_set:{x_set},\n state:{state},\n fake_state:{fake_state}')


# Convert lists to numpy arrays for easy indexing
states = np.array(states)  # Shape (timesteps, state_dim)
if exist_attack:
    fake_states = np.array(fake_states)
nom_controls = np.array(nom_controls)  # Shape (timesteps, control_dim)
act_controls = np.array(act_controls)  # Shape (timesteps, control_dim)
controls_filtered = np.array(controls_filtered)  # Shape (timesteps, control_dim)
h_values = np.array(h_values)  # Shape (timesteps,)

time_axis = np.linspace(0, timesteps * T, timesteps)  # Time vector


# # Define the header
# header = "xp yp theta v w"

# # Save to a text file
# timestamp = datetime.datetime.now().strftime('%m%d_%H%M')
# data = np.hstack([states[:-1],act_controls])
# np.savetxt(f"data/x_p_y_p_theta_{timestamp}.txt", data, header=header, fmt="%.6f")




def plt_levelset(func, grid_region):
    xlim = grid_region[0]
    ylim = grid_region[1]
    # Define grid
    x_vals = np.linspace(xlim[0], xlim[1], 200)
    y_vals = np.linspace(ylim[0], ylim[1], 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Apply the function to the grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(X[i, j], Y[i, j])
    
    # Plot the level set
    plt.contour(X, Y, Z, levels=[0], colors='r', linewidths=2, linestyles='dashed')
    # Add a label for the safety boundary
    plt.text(xlim[0] + 0.3* (xlim[1] - xlim[0]), 3, 'Safety Boundary', 
             color='r', fontsize=14, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Plot Trajectory
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(states[:, 0], states[:, 1], label='Ground-true Trajectory')
if exist_attack:
    plt.plot(fake_states[:, 0], fake_states[:, 1], label='Fake Trajectory')

ax = plt.gca()
# Retrieve the automatically determined x and y limits
x_limits = ax.get_xlim()
y_limits = ax.get_ylim()
grid_region = [x_limits, y_limits]

plt_levelset(lambda x,y: safe_func(None,[x,y]),grid_region)
# lane_x = np.arange(-20,0,0.01)
# lane_y = reference_param.a*np.sin(reference_param.b*lane_x+ reference_param.c) + reference_param.d
# plt.fill_between(lane_x, lane_y-zocbf_param.radius, lane_y+zocbf_param.radius, 
#                  alpha=0.4, label="Safe region")

plt.scatter(states[0, 0], states[0, 1], color='green', marker='o', label='Start')
plt.scatter(states[-1, 0], states[-1, 1], color='red', marker='x', label='End')
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Unicycle Trajectory")
plt.legend()
plt.grid()

# Plot Inputs
plt.subplot(1, 2, 2)
plt.plot(time_axis, nom_controls[:, 0], label='Nominal $u_1$')
plt.plot(time_axis, nom_controls[:, 1], label='Nominal $u_2$')
if use_filter:
    plt.plot(time_axis, controls_filtered[:, 0], label='Filtered $u_1$', linestyle='dashed')
    plt.plot(time_axis, controls_filtered[:, 1], label='Filtered $u_2$', linestyle='dashed')
plt.xlabel("Time (s)")
plt.ylabel("Control Inputs")
plt.title("Control Inputs vs Filtered Inputs")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Plot h-values
plt.figure(figsize=(8, 4))
plt.plot(time_axis, h_values, label='h-values')
plt.axhline(y=0, color='r', linestyle='--', label='h = 0 boundary')
plt.xlabel("Time (s)")
plt.ylabel("h-value")
plt.title("Safety Constraint h over Time")
plt.legend()
plt.grid()
plt.show()


## saving the data
# np.savez('data_nossf_states_a1_8b5',states = states, fake_states = fake_states,h = h_values,time_axis=time_axis)

# np.savez('data_ssf_states_a1_8b5',states = states, fake_states = fake_states,h = h_values,time_axis=time_axis,
#          nom_controls= nom_controls, controls_filtered = controls_filtered, reference_param = reference_param)
