# Week1: Plot the average GRF/COP from the force plate and insole
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import interpolate

# Load data
data = loadmat('C:/Users/XuanL/Documents/MATLAB/Gait Analysis/ABLE1_112024_1.0.mat', struct_as_record=False,
               squeeze_me=True)

# Get gait data
gait_insole = data['gait_insole']
gait_trackers = data['gait_trackers']


def process_gait_cycles(data, strikes, offs):
    """
    Extract and normalize stance phases

    Parameters:
    data -- array containing measurement data
    strikes -- heel strike indices
    offs -- toe off indices

    Returns:
    norm_data -- normalized stance phase data (100 points per cycle)
    avg_data -- average of normalized data
    std_data -- standard deviation of normalized data
    """
    num_cycles = min(len(strikes), len(offs))
    stance_data = []

    # Extract stance phase data
    for i in range(num_cycles):
        if strikes[i] < offs[i]:
            stance_data.append(data[strikes[i]:offs[i], :])

    num_valid = len(stance_data)

    if num_valid == 0:
        raise ValueError('No valid stance phases found')

    # Normalize to 100 points
    norm_data = np.zeros((100, data.shape[1], num_valid))
    for i in range(num_valid):
        for j in range(data.shape[1]):
            x_old = np.linspace(0, 1, stance_data[i].shape[0])
            x_new = np.linspace(0, 1, 100)
            f = interpolate.interp1d(x_old, stance_data[i][:, j])
            norm_data[:, j, i] = f(x_new)

    # Calculate average and standard deviation
    avg_data = np.mean(norm_data, axis=2)
    std_data = np.std(norm_data, axis=2)

    return norm_data, avg_data, std_data


# Process Force Plate GRF
# Right foot GRF
norm_right_grf, avg_right_grf, std_right_grf = process_gait_cycles(
    gait_trackers.force_plate_ds_r[:, 0:3], gait_trackers.strike_r, gait_trackers.off_r)

# Left foot GRF
norm_left_grf, avg_left_grf, std_left_grf = process_gait_cycles(
    gait_trackers.force_plate_ds_l[:, 0:3], gait_trackers.strike_l, gait_trackers.off_l)

# Process Force Plate GRM
# Right foot GRM
norm_right_grm, avg_right_grm, std_right_grm = process_gait_cycles(
    gait_trackers.force_plate_ds_r[:, 3:6], gait_trackers.strike_r, gait_trackers.off_r)

# Left foot GRM
norm_left_grm, avg_left_grm, std_left_grm = process_gait_cycles(
    gait_trackers.force_plate_ds_l[:, 3:6], gait_trackers.strike_l, gait_trackers.off_l)

# Process Force Plate COP
# Right foot COP
norm_right_cop_fp, avg_right_cop_fp, std_right_cop_fp = process_gait_cycles(
    gait_trackers.force_plate_ds_r[:, 6:8], gait_trackers.strike_r, gait_trackers.off_r)

# Left foot COP
norm_left_cop_fp, avg_left_cop_fp, std_left_cop_fp = process_gait_cycles(
    gait_trackers.force_plate_ds_l[:, 6:8], gait_trackers.strike_l, gait_trackers.off_l)

# Process Insole COP
# Right foot insole COP
right_cop_data = np.column_stack((gait_insole.cop_x_r, gait_insole.cop_y_r))
norm_right_cop_insole, avg_right_cop_insole, std_right_cop_insole = process_gait_cycles(
    right_cop_data, gait_insole.strike_r, gait_insole.off_r)

# Left foot insole COP
left_cop_data = np.column_stack((gait_insole.cop_x_l, gait_insole.cop_y_l))
norm_left_cop_insole, avg_left_cop_insole, std_left_cop_insole = process_gait_cycles(
    left_cop_data, gait_insole.strike_l, gait_insole.off_l)

# Right foot insole pp
norm_right_insole_pp, avg_right_insole_pp, std_right_insole_pp = process_gait_cycles(
    gait_insole.pp_r.reshape(-1, 1), gait_insole.strike_r, gait_insole.off_r)

# Left foot insole pp
norm_left_insole_pp, avg_left_insole_pp, std_left_insole_pp = process_gait_cycles(
    gait_insole.pp_l.reshape(-1, 1), gait_insole.strike_l, gait_insole.off_l)

#Used in WNN
norm_right_insole, avg_right_insole, std_right_insole = process_gait_cycles(
    gait_insole.insole_r, gait_insole.strike_r, gait_insole.off_r)


def plot_graph():
    # Common x-axis for all plots
    x_axis = np.arange(100)

    # Plot Force Plate-COP
    plt.figure(figsize=(12, 10))
    plt.suptitle('COP from Force Plate')

    # Right foot COP X
    plt.subplot(2, 2, 1)
    plt.plot(x_axis, avg_right_cop_fp[:, 0], 'r', linewidth=2)
    plt.fill_between(x_axis,
                     avg_right_cop_fp[:, 0] + std_right_cop_fp[:, 0],
                     avg_right_cop_fp[:, 0] - std_right_cop_fp[:, 0],
                     color='r', alpha=0.2)
    plt.title('Right Foot COP X (Medial-Lateral)')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)

    # Right foot COP Y
    plt.subplot(2, 2, 2)
    plt.plot(x_axis, avg_right_cop_fp[:, 1], 'r', linewidth=2)
    plt.fill_between(x_axis,
                     avg_right_cop_fp[:, 1] + std_right_cop_fp[:, 1],
                     avg_right_cop_fp[:, 1] - std_right_cop_fp[:, 1],
                     color='r', alpha=0.2)
    plt.title('Right Foot COP Y (Anterior-Posterior)')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)

    # Left foot COP X
    plt.subplot(2, 2, 3)
    plt.plot(x_axis, avg_left_cop_fp[:, 0], 'b', linewidth=2)
    plt.fill_between(x_axis,
                     avg_left_cop_fp[:, 0] + std_left_cop_fp[:, 0],
                     avg_left_cop_fp[:, 0] - std_left_cop_fp[:, 0],
                     color='b', alpha=0.2)
    plt.title('Left Foot COP X (Medial-Lateral)')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)

    # Left foot COP Y
    plt.subplot(2, 2, 4)
    plt.plot(x_axis, avg_left_cop_fp[:, 1], 'b', linewidth=2)
    plt.fill_between(x_axis,
                     avg_left_cop_fp[:, 1] + std_left_cop_fp[:, 1],
                     avg_left_cop_fp[:, 1] - std_left_cop_fp[:, 1],
                     color='b', alpha=0.2)
    plt.title('Left Foot COP Y (Anterior-Posterior)')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)
    plt.show()

    # Plot COP from Insoles
    plt.figure(figsize=(12, 10))
    plt.suptitle('COP from Insoles')

    # Right foot insole COP X
    plt.subplot(2, 2, 1)
    plt.plot(x_axis, avg_right_cop_insole[:, 0], 'r', linewidth=2)
    plt.fill_between(x_axis,
                     avg_right_cop_insole[:, 0] + std_right_cop_insole[:, 0],
                     avg_right_cop_insole[:, 0] - std_right_cop_insole[:, 0],
                     color='r', alpha=0.2)
    plt.title('Right Foot Insole COP X (Anterior-Posterior)')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)

    # Right foot insole COP Y
    plt.subplot(2, 2, 2)
    plt.plot(x_axis, avg_right_cop_insole[:, 1], 'r', linewidth=2)
    plt.fill_between(x_axis,
                     avg_right_cop_insole[:, 1] + std_right_cop_insole[:, 1],
                     avg_right_cop_insole[:, 1] - std_right_cop_insole[:, 1],
                     color='r', alpha=0.2)
    plt.title('Right Foot Insole COP Y (Medial-Lateral)')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)

    # Left foot insole COP X
    plt.subplot(2, 2, 3)
    plt.plot(x_axis, avg_left_cop_insole[:, 0], 'b', linewidth=2)
    plt.fill_between(x_axis,
                     avg_left_cop_insole[:, 0] + std_left_cop_insole[:, 0],
                     avg_left_cop_insole[:, 0] - std_left_cop_insole[:, 0],
                     color='b', alpha=0.2)
    plt.title('Left Foot Insole COP X (Anterior-Posterior)')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)

    # Left foot insole COP Y
    plt.subplot(2, 2, 4)
    plt.plot(x_axis, avg_left_cop_insole[:, 1], 'b', linewidth=2)
    plt.fill_between(x_axis,
                     avg_left_cop_insole[:, 1] + std_left_cop_insole[:, 1],
                     avg_left_cop_insole[:, 1] - std_left_cop_insole[:, 1],
                     color='b', alpha=0.2)
    plt.title('Left Foot Insole COP Y (Medial-Lateral)')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)
    plt.show()

    # Plot Ground Reaction Forces
    plt.figure(figsize=(12, 15))
    plt.suptitle('Ground Reaction Forces')

    # X direction (Medial-Lateral)
    plt.subplot(3, 2, 1)
    plt.plot(x_axis, avg_right_grf[:, 0], 'r', linewidth=2)
    plt.fill_between(x_axis,
                     avg_right_grf[:, 0] + std_right_grf[:, 0],
                     avg_right_grf[:, 0] - std_right_grf[:, 0],
                     color='r', alpha=0.2)
    plt.title('Right Foot GRF X (Medial-Lateral)')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Force (N)')
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(x_axis, avg_left_grf[:, 0], 'b', linewidth=2)
    plt.fill_between(x_axis,
                     avg_left_grf[:, 0] + std_left_grf[:, 0],
                     avg_left_grf[:, 0] - std_left_grf[:, 0],
                     color='b', alpha=0.2)
    plt.title('Left Foot GRF X (Medial-Lateral)')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Force (N)')
    plt.grid(True)

    # Y direction (Anterior-Posterior)
    plt.subplot(3, 2, 3)
    plt.plot(x_axis, avg_right_grf[:, 1], 'r', linewidth=2)
    plt.fill_between(x_axis,
                     avg_right_grf[:, 1] + std_right_grf[:, 1],
                     avg_right_grf[:, 1] - std_right_grf[:, 1],
                     color='r', alpha=0.2)
    plt.title('Right Foot GRF Y (Anterior-Posterior)')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Force (N)')
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(x_axis, avg_left_grf[:, 1], 'b', linewidth=2)
    plt.fill_between(x_axis,
                     avg_left_grf[:, 1] + std_left_grf[:, 1],
                     avg_left_grf[:, 1] - std_left_grf[:, 1],
                     color='b', alpha=0.2)
    plt.title('Left Foot GRF Y (Anterior-Posterior)')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Force (N)')
    plt.grid(True)

    # Z direction (Vertical)
    plt.subplot(3, 2, 5)
    plt.plot(x_axis, avg_right_grf[:, 2], 'r', linewidth=2)
    plt.fill_between(x_axis,
                     avg_right_grf[:, 2] + std_right_grf[:, 2],
                     avg_right_grf[:, 2] - std_right_grf[:, 2],
                     color='r', alpha=0.2)
    plt.title('Right Foot GRF Z (Vertical)')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Force (N)')
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.plot(x_axis, avg_left_grf[:, 2], 'b', linewidth=2)
    plt.fill_between(x_axis,
                     avg_left_grf[:, 2] + std_left_grf[:, 2],
                     avg_left_grf[:, 2] - std_left_grf[:, 2],
                     color='b', alpha=0.2)
    plt.title('Left Foot GRF Z (Vertical)')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Force (N)')
    plt.grid(True)
    plt.show()

    # Plot Ground Reaction Moments
    plt.figure(figsize=(12, 15))
    plt.suptitle('Ground Reaction Moments')

    # X direction (Medial-Lateral)
    plt.subplot(3, 2, 1)
    plt.plot(x_axis, avg_right_grm[:, 0], 'r', linewidth=2)
    plt.fill_between(x_axis,
                     avg_right_grm[:, 0] + std_right_grm[:, 0],
                     avg_right_grm[:, 0] - std_right_grm[:, 0],
                     color='r', alpha=0.2)
    plt.title('Right Foot GRM X (Medial-Lateral)')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Moment (N*m)')
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(x_axis, avg_left_grm[:, 0], 'b', linewidth=2)
    plt.fill_between(x_axis,
                     avg_left_grm[:, 0] + std_left_grm[:, 0],
                     avg_left_grm[:, 0] - std_left_grm[:, 0],
                     color='b', alpha=0.2)
    plt.title('Left Foot GRM X (Medial-Lateral)')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Moment (N*m)')
    plt.grid(True)

    # Y direction (Anterior-Posterior)
    plt.subplot(3, 2, 3)
    plt.plot(x_axis, avg_right_grm[:, 1], 'r', linewidth=2)
    plt.fill_between(x_axis,
                     avg_right_grm[:, 1] + std_right_grm[:, 1],
                     avg_right_grm[:, 1] - std_right_grm[:, 1],
                     color='r', alpha=0.2)
    plt.title('Right Foot GRM Y (Anterior-Posterior)')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Moment (N*m)')
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(x_axis, avg_left_grm[:, 1], 'b', linewidth=2)
    plt.fill_between(x_axis,
                     avg_left_grm[:, 1] + std_left_grm[:, 1],
                     avg_left_grm[:, 1] - std_left_grm[:, 1],
                     color='b', alpha=0.2)
    plt.title('Left Foot GRM Y (Anterior-Posterior)')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Moment (N*m)')
    plt.grid(True)

    # Z direction (Vertical)
    plt.subplot(3, 2, 5)
    plt.plot(x_axis, avg_right_grm[:, 2], 'r', linewidth=2)
    plt.fill_between(x_axis,
                     avg_right_grm[:, 2] + std_right_grm[:, 2],
                     avg_right_grm[:, 2] - std_right_grm[:, 2],
                     color='r', alpha=0.2)
    plt.title('Right Foot GRM Z (Vertical)')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Moment (N*m)')
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.plot(x_axis, avg_left_grm[:, 2], 'b', linewidth=2)
    plt.fill_between(x_axis,
                     avg_left_grm[:, 2] + std_left_grm[:, 2],
                     avg_left_grm[:, 2] - std_left_grm[:, 2],
                     color='b', alpha=0.2)
    plt.title('Left Foot GRM Z (Vertical)')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Moment (N*m)')
    plt.grid(True)
    plt.show()

    plt.subplot(2,1,1)
    plt.plot(x_axis, avg_right_insole_pp, 'r', linewidth=2)

    plt.title('Right Foot Insole PP')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('ADC Value')
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(x_axis, avg_left_insole_pp, 'b', linewidth=2)

    plt.title('Left Foot Insole PP')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('ADC Value')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    norm_right_grf=norm_right_grf.reshape(10000,3)
    plt.figure(figsize=(10, 6))
    plt.plot(norm_right_grf[:, 0], label='Actual GRF (X)')
    plt.plot(norm_right_grf[:, 1], label='Predicted GRF (X)')
    plt.xlabel('Sample')
    plt.ylabel('GRF (X)')
    plt.title('Actual vs. Predicted GRF (X)')
    plt.legend()
    plt.show()
    plot_graph()