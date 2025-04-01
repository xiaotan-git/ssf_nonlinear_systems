import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages

# Set global matplotlib parameters for consistent styling
def set_figure_defaults():
    plt.rcParams['lines.linewidth'] = 2    # linewidth
    plt.rcParams['lines.markersize'] = 2   # marker size

    plt.rcParams['axes.linewidth'] = 1     # linewidth
    plt.rcParams['axes.labelsize'] = 11    # axes font size
    plt.rcParams['xtick.labelsize'] = 11   # x-tick font size
    plt.rcParams['ytick.labelsize'] = 11   # y-tick font size

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 10          # size for text
    plt.rcParams['legend.fontsize'] = 11    # legend font size
    plt.rcParams['legend.title_fontsize'] = 11  # legend title font size

    plt.rcParams['axes.formatter.use_locale'] = False
    plt.rcParams['legend.handlelength'] = 1.2
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # (if needed)
fig_width = 6
fig_height = fig_width / 1.8

# plt.rcParams.update({
#     'font.size': 14,
#     'font.family': 'serif',
#     'axes.labelsize': 14,
#     'axes.titlesize': 14,
#     'xtick.labelsize': 14,
#     'ytick.labelsize': 14,
#     'legend.fontsize': 14,
#     'figure.titlesize': 14
# })
# # fixing TYPE 3 font problem 
# plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['ps.fonttype'] = 42
# plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
# plt.rc('text', usetex=True )

set_figure_defaults()

# Load data
load_ssf = np.load("data_ssf_states_a1_8b5.npz",allow_pickle=True)
ssf_states, ssf_fake_states, ssf_h, nom_controls, controls_filtered, time_axis = load_ssf['states'], load_ssf['fake_states'], load_ssf['h'], load_ssf['nom_controls'],load_ssf['controls_filtered'],load_ssf['time_axis']

load_nossf = np.load('data_nossf_states_a1_8b5.npz')
nossf_states, nossf_fake_states, nossf_h, time_axis = load_nossf['states'], load_nossf['fake_states'], load_nossf['h'], load_nossf['time_axis']


reference_param = load_ssf['reference_param']

from pprint import pprint
print(reference_param)
# pprint(vars(reference_param))

def safe_func(x):
    return 3**2 - (x[1])**2


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
fig1 = plt.figure(figsize=(fig_width, fig_height))

# Plot with enhanced colors and line widths
plt.plot(nossf_states[:, 0], nossf_states[:, 1], linewidth=2, color='navy', 
            label='Ground-truth Trajectory w/o SSF')
plt.plot(nossf_fake_states[:, 0], nossf_fake_states[:, 1], linewidth=2, color='cornflowerblue', 
            linestyle='--', label='Fake Trajectory w/o SSF')
plt.plot(ssf_states[:, 0], ssf_states[:, 1], linewidth=2, color='darkgreen', 
            label='Ground-truth Trajectory w/ SSF')
plt.plot(ssf_fake_states[:, 0], ssf_fake_states[:, 1], linewidth=2, color='lightgreen', 
            linestyle='--', label='Fake Trajectory w/ SSF')

ax = plt.gca()
# Retrieve the automatically determined x and y limits
plt.ylim(-7,4)
x_limits = ax.get_xlim()
y_limits = ax.get_ylim()
grid_region = [x_limits, y_limits]
plt_levelset(lambda x, y: safe_func([x, y]), grid_region)

# Add a more visible starting point
plt.scatter(nossf_states[0, 0], nossf_states[0, 1], color='green', marker='o', s=100, 
            edgecolor='black', label='Start', zorder=5)
plt.scatter(nossf_states[-1, 0], nossf_states[-1, 1], color='navy', marker='x', s=100, 
            edgecolor='black', label='End', zorder=5)
plt.scatter(ssf_states[-1, 0], ssf_states[-1, 1], color='darkgreen', marker='x', s=100, 
            edgecolor='black', label='End', zorder=5)

plt.xlabel("X Position", fontweight='bold')
plt.ylabel("Y Position", fontweight='bold')
plt.legend(loc='best', ncol=2, framealpha=0.9, edgecolor='gray')
plt.grid(linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the first figure
with PdfPages('figures/trajectory_plot.pdf') as pdf:
    plt.savefig(pdf, format='pdf', bbox_inches='tight', pad_inches=0.1)

# Close the first figure to avoid memory issues
plt.show()

# Plot h-values
fig2 = plt.figure(figsize=(fig_width, fig_height))

plt.plot(time_axis, nossf_h, linestyle='--', linewidth=2, color='navy', label='h-values w/o SSF')
plt.plot(time_axis, ssf_h, linewidth=2, color='darkgreen', label='h-values w/ SSF')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Safety Threshold')

# Fill areas below the safety threshold to highlight violations
# plt.fill_between(time_axis, 0, nossf_h, where=(nossf_h < 0), color='navy', alpha=0.3)
# plt.fill_between(time_axis, 0, ssf_h, where=(ssf_h < 0), color='darkgreen', alpha=0.3)

plt.xlabel("Time (s)", fontweight='bold')
plt.ylabel("h value", fontweight='bold')
plt.legend(loc='best', framealpha=0.9, edgecolor='gray')
plt.grid(linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the second figure
with PdfPages('figures/h_values_plot.pdf') as pdf:
    plt.savefig(pdf, format='pdf', bbox_inches='tight', pad_inches=0.1)

# Display the plots
plt.show()

# Plot control inputs
fig3 = plt.figure(figsize=(fig_width, fig_height))

plt.plot(time_axis, nom_controls[:, 0], linewidth=2, color='navy', label='Nominal $v$')
plt.plot(time_axis, nom_controls[:, 1], linewidth=2, color='cornflowerblue', label='Nominal $\mu$')

plt.plot(time_axis, controls_filtered[:, 0], linewidth=2, color='darkgreen', label='Filtered $v$', linestyle='dashed')
plt.plot(time_axis, controls_filtered[:, 1], linewidth=2, color='lightgreen', label='Filtered $\mu$', linestyle='dashed')

# Fill areas below the safety threshold to highlight violations
# plt.fill_between(time_axis, 0, nossf_h, where=(nossf_h < 0), color='navy', alpha=0.3)
# plt.fill_between(time_axis, 0, ssf_h, where=(ssf_h < 0), color='darkgreen', alpha=0.3)

plt.xlabel("Time (s)", fontweight='bold')
plt.ylabel("Control Inputs", fontweight='bold')
plt.legend(loc='best', framealpha=0.9, edgecolor='gray')
plt.grid(linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the second figure
with PdfPages('figures/u_plot.pdf') as pdf:
    plt.savefig(pdf, format='pdf', bbox_inches='tight', pad_inches=0.1)

# Display the plots
plt.show()