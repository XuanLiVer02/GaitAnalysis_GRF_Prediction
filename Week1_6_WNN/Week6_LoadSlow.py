import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

# Load data
data = loadmat('C:/Users/XuanL/Documents/MATLAB/Gait Analysis/ABLE1_112024_0.75.mat', struct_as_record=False,
               squeeze_me=True)

# Get gait data
gait_insole = data['gait_insole']
gait_trackers = data['gait_trackers']

def lowpass_filter(data, cutoff=10, fs=100, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    # Butterworth Filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Zero-phase filtering
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

grf_r = lowpass_filter(gait_trackers.force_plate_ds_r[:, 0:3], cutoff=10, fs=100, order=2)
grm_r = lowpass_filter(gait_trackers.force_plate_ds_r[:, 3:6], cutoff=10, fs=100, order=2)
grf_l = lowpass_filter(gait_trackers.force_plate_ds_l[:, 0:3], cutoff=10, fs=100, order=2)
grm_l = lowpass_filter(gait_trackers.force_plate_ds_l[:, 3:6], cutoff=10, fs=100, order=2)

def pcami_cycle(X_insole, Y_target, n_components=5, top_k_features=10):
    pca = PCA(n_components=top_k_features)
    scaler = StandardScaler()
    X_insole = scaler.fit_transform(X_insole)
    X_pca = pca.fit_transform(X_insole)
    # 单个主成分解释方差比例
    explained = pca.explained_variance_ratio_  # shape: (50,)
    # 累积解释方差
    cumulative = np.cumsum(explained)
    print(f"Cumulative Variance: {cumulative[-1] * 100:.2f}%")
    mi_scores = []
    for i in range(X_pca.shape[1]):
        mi = sum(mutual_info_regression(X_pca[:, [i]], Y_target[:, j])[0] for j in range(Y_target.shape[1]))
        mi_scores.append(mi)
    mi_scores = np.array(mi_scores)
    selected = np.argsort(mi_scores)[-n_components:]  # indices of top-N MI scores
    selected.sort()

    print("Top MI score:", mi_scores[selected[-1]])
    return X_pca[:, selected]

def process_gait_cycles(insole_data, force_data, strikes, offs, strikes_force, offs_force, pcami=False):
    """
    Extract and normalize stance phases for insole and force plate

    Parameters:
    insole_data -- array of insole measurements (time x channels)
    force_data -- array of force plate measurements (time x channels)
    strikes -- heel strike indices (对应insole和force都一样时间轴)
    offs -- toe off indices

    Returns:
    norm_insole -- normalized insole stance phase (100 points per cycle)
    avg_insole -- average of normalized insole data
    std_insole -- standard deviation of normalized insole data
    norm_force -- normalized force plate stance phase (100 points per cycle)
    avg_force -- average of normalized force plate data
    std_force -- standard deviation of normalized force plate data
    """
    if pcami:
        insole_for_pca = insole_data

        t_force = np.arange(force_data.shape[0]) * 0.01  # 100Hz
        t_insole = np.arange(insole_data.shape[0]) * 0.025  # 40Hz

        force_for_pca = np.zeros((len(t_insole), force_data.shape[1]))

        for j in range(force_data.shape[1]):
            f = interpolate.interp1d(t_force, force_data[:, j], kind='linear', fill_value='extrapolate')
            force_for_pca[:, j] = f(t_insole)

        n_components = 5
        top_k_features = 10
        X_reduced = pcami_cycle(insole_for_pca, force_for_pca, n_components=n_components, top_k_features=top_k_features)
        print('X_reduced', X_reduced)  # 4920,5
        insole_data = X_reduced
    num_cycles = min(len(strikes), len(offs), len(strikes_force), len(offs_force))
    insole_stance = []
    force_stance = []

    for i in range(num_cycles):
        if strikes[i] < offs[i]:
            insole_stance.append(insole_data[strikes[i]:offs[i], :])
    for i in range(num_cycles):
        if strikes_force[i] < offs_force[i]:
            force_stance.append(force_data[strikes_force[i]:offs_force[i], :])
#    insole_for_pca = np.vstack(insole_stance)
#    force_for_pca = np.vstack(force_stance)
#    print('123', insole_for_pca.shape, force_for_pca.shape)
#    if pcami:
#        n_components = 5
#        top_k_features = 10
#        X_reduced = pcami_cycle(insole_for_pca, force_for_pca, n_components=n_components, top_k_features=top_k_features)
#        split_indices = np.cumsum(num_cycles)
#        X_splits = np.split(X_reduced, split_indices[:-1])
#        insole_stance = X_splits

    num_valid = len(insole_stance)

    if num_valid == 0:
        raise ValueError('No valid stance phases found')

    num_points = 100
    raw_insole = np.zeros((num_points, insole_data.shape[1], num_valid))
    raw_force = np.zeros((num_points, force_data.shape[1], num_valid))

    for i in range(num_valid):  # Every step
        insole_cycle = insole_stance[i]     # 30, num_features
        force_cycle = force_stance[i]   #75,3
        """
        t_insole = np.arange(insole_cycle.shape[0]) * 0.025
        t_force = np.arange(force_cycle.shape[0]) * 0.01
        standard_time_insole = np.linspace(0, t_insole[-1], num_points)
        standard_time_force = np.linspace(0, t_force[-1], num_points)
        for j in range(insole_cycle.shape[1]):
            f_insole = interpolate.interp1d(t_insole, insole_cycle[:, j], kind='cubic', fill_value='extrapolate')
            raw_insole[:, j, i] = f_insole(standard_time_insole)
        for j in range(force_cycle.shape[1]):
            f_force = interpolate.interp1d(t_force, force_cycle[:, j], kind='cubic', fill_value='extrapolate')
            raw_force[:, j, i] = f_force(standard_time_force)
        """
        t_insole = np.arange(insole_cycle.shape[0]) * 0.025
        t_force_full = np.arange(force_cycle.shape[0]) * 0.01

        # Find those < t_insole[-1] in t_force
        valid_force_idx = t_force_full <= t_insole[-1]
        t_force = t_force_full[valid_force_idx]
        force_cycle = force_cycle[valid_force_idx, :]  # Cutoff force_cycle

        standard_time_insole = np.linspace(0, t_insole[-1], num_points)
        standard_time_force = np.linspace(0, t_force[-1], num_points)
        for j in range(insole_cycle.shape[1]):
            f_insole = interpolate.interp1d(t_insole, insole_cycle[:, j], kind='cubic', fill_value='extrapolate')
            raw_insole[:, j, i] = f_insole(standard_time_insole)

        for j in range(force_cycle.shape[1]):
            f_force = interpolate.interp1d(t_force, force_cycle[:, j], kind='cubic', fill_value='extrapolate')
            raw_force[:, j, i] = f_force(standard_time_force)

    avg_insole = np.mean(raw_insole, axis=2)
    std_insole = np.std(raw_insole, axis=2)

    avg_force = np.mean(raw_force, axis=2)
    std_force = np.std(raw_force, axis=2)

    return raw_insole, avg_insole, std_insole, raw_force, avg_force, std_force

# Process Force Plate GRF
# Right foot GRF
_, _, _, norm_right_grf_slow, avg_right_grf_slow, std_right_grf_slow = process_gait_cycles(
    gait_insole.pp_r.reshape(-1, 1),
    grf_r,
    gait_insole.strike_r,
    gait_insole.off_r,
    gait_trackers.strike_r,
    gait_trackers.off_r)

# Left foot GRF
_, _, _, norm_left_grf_slow, avg_left_grf_slow, std_left_grf_slow = process_gait_cycles(
    gait_insole.pp_l.reshape(-1, 1),
    grf_l,
    gait_insole.strike_l,
    gait_insole.off_l,
    gait_trackers.strike_l,
    gait_trackers.off_l)

# Process Force Plate GRM
# Right foot GRM
_, _, _, norm_right_grm_slow, avg_right_grm_slow, std_right_grm_slow = process_gait_cycles(
    gait_insole.pp_r.reshape(-1, 1),
    grm_r,
    gait_insole.strike_r,
    gait_insole.off_r,
    gait_trackers.strike_r,
    gait_trackers.off_r
)

# Left foot GRM
_, _, _, norm_left_grm_slow, avg_left_grm_slow, std_left_grm_slow = process_gait_cycles(
    gait_insole.pp_l.reshape(-1, 1),
    grm_l,
    gait_insole.strike_l,
    gait_insole.off_l,
    gait_trackers.strike_l,
    gait_trackers.off_l
)

# Process Force Plate COP
# Right foot COP
_, _, _, norm_right_cop_fp_slow, avg_right_cop_fp_slow, std_right_cop_fp_slow = process_gait_cycles(
    gait_insole.pp_r.reshape(-1, 1),
    gait_trackers.force_plate_ds_r[:, 6:8],
    gait_insole.strike_r,
    gait_insole.off_r,
    gait_trackers.strike_r,
    gait_trackers.off_r
)

# Left foot COP
_, _, _, norm_left_cop_fp_slow, avg_left_cop_fp_slow, std_left_cop_fp_slow = process_gait_cycles(
    gait_insole.pp_l.reshape(-1, 1),
    gait_trackers.force_plate_ds_l[:, 6:8],
    gait_insole.strike_l,
    gait_insole.off_l,
    gait_trackers.strike_l,
    gait_trackers.off_l
)

# Process Insole COP
# Right foot insole COP
right_cop_data = np.column_stack((gait_insole.cop_x_r, gait_insole.cop_y_r))
norm_right_cop_insole_slow, avg_right_cop_insole_slow, std_right_cop_insole_slow, _, _, _ = process_gait_cycles(
    right_cop_data,
    gait_trackers.force_plate_ds_r[:, 6:8],  # force plate COP
    gait_insole.strike_r,
    gait_insole.off_r,
    gait_trackers.strike_r,
    gait_trackers.off_r
)

# Left foot insole COP
left_cop_data = np.column_stack((gait_insole.cop_x_l, gait_insole.cop_y_l))
norm_left_cop_insole_slow, avg_left_cop_insole_slow, std_left_cop_insole_slow, _, _, _ = process_gait_cycles(
    left_cop_data,
    gait_trackers.force_plate_ds_l[:, 6:8],  # force plate COP
    gait_insole.strike_l,
    gait_insole.off_l,
    gait_trackers.strike_l,
    gait_trackers.off_l
)

# Right foot insole pp
norm_right_insole_pp_slow, avg_right_insole_pp_slow, std_right_insole_pp_slow, _, _, _ = process_gait_cycles(
    gait_insole.pp_r.reshape(-1, 1),
    grf_r,  # Force plate vertical GRF（Z方向）可以作为reference
    gait_insole.strike_r,
    gait_insole.off_r,
    gait_trackers.strike_r,
    gait_trackers.off_r
)

# Left foot insole pp
norm_left_insole_pp_slow, avg_left_insole_pp_slow, std_left_insole_pp_slow, _, _, _ = process_gait_cycles(
    gait_insole.pp_l.reshape(-1, 1),
    grf_l,
    gait_insole.strike_l,
    gait_insole.off_l,
    gait_trackers.strike_l,
    gait_trackers.off_l
)

# Used in WNN
norm_right_insole_slow, avg_right_insole_slow, std_right_insole_slow, _, _, _ = process_gait_cycles(
    gait_insole.insole_r,
    grf_r,
    gait_insole.strike_r,
    gait_insole.off_r,
    gait_trackers.strike_r,
    gait_trackers.off_r,
    pcami=True)
norm_left_insole_slow, avg_left_insole_slow, std_v_insole_slow, _, _, _ = process_gait_cycles(
    gait_insole.insole_l,
    grf_l,
    gait_insole.strike_l,
    gait_insole.off_l,
    gait_trackers.strike_l,
    gait_trackers.off_l,
    pcami=True)


# New features for week 4
norm_right_fti_slow, avg_right_fti_slow, std_right_fti_slow, _, _, _ = process_gait_cycles(
    gait_insole.fti_r.reshape(-1, 1),
    grf_r,
    gait_insole.strike_r,
    gait_insole.off_r,
    gait_trackers.strike_r,
    gait_trackers.off_r,
)

norm_left_fti_slow, avg_left_fti_slow, std_left_fti_slow, _, _, _ = process_gait_cycles(
    gait_insole.fti_l.reshape(-1, 1),
    grf_l,
    gait_insole.strike_l,
    gait_insole.off_l,
    gait_trackers.strike_l,
    gait_trackers.off_l,
)

norm_right_ft_slow, avg_right_ft_slow, std_right_ft_slow, _, _, _ = process_gait_cycles(
    gait_insole.foot_trace_r.reshape(-1, 1),
    grf_r,
    gait_insole.strike_r,
    gait_insole.off_r,
    gait_trackers.strike_r,
    gait_trackers.off_r,
)
norm_left_ft_slow, avg_left_ft_slow, std_left_ft_slow, _, _, _ = process_gait_cycles(
    gait_insole.foot_trace_l.reshape(-1, 1),
    grf_l,
    gait_insole.strike_l,
    gait_insole.off_l,
    gait_trackers.strike_l,
    gait_trackers.off_l,
)

norm_right_contact_slow, avg_right_contact_slow, std_right_contact_slow, _, _, _ = process_gait_cycles(
    gait_insole.cont_area_r.reshape(-1, 1),
    grf_r,
    gait_insole.strike_r,
    gait_insole.off_r,
    gait_trackers.strike_r,
    gait_trackers.off_r,
)
norm_left_contact_slow, avg_left_contact_slow, std_left_contact_slow, _, _, _ = process_gait_cycles(
    gait_insole.cont_area_l.reshape(-1, 1),
    grf_l,
    gait_insole.strike_l,
    gait_insole.off_l,
    gait_trackers.strike_l,
    gait_trackers.off_l,
)

right_pp_pos = np.column_stack((gait_insole.pp_x_r, gait_insole.pp_y_r))
norm_right_pp_pos_slow, avg_right_pp_pos_slow, std_right_pp_pos_slow, _, _, _ = process_gait_cycles(
    right_pp_pos,
    grf_r,
    gait_insole.strike_r,
    gait_insole.off_r,
    gait_trackers.strike_r,
    gait_trackers.off_r,
)

left_pp_pos = np.column_stack((gait_insole.pp_x_l, gait_insole.pp_y_l))
norm_left_pp_pos_slow, avg_left_pp_pos_slow, std_left_pp_pos_slow, _, _, _ = process_gait_cycles(
    left_pp_pos,
    grf_l,
    gait_insole.strike_l,
    gait_insole.off_l,
    gait_trackers.strike_l,
    gait_trackers.off_l,
)
# Used in CNN
norm_right_insole_slow_all, avg_right_insole_slow_all, std_right_insole_slow_all, _, _, _ = process_gait_cycles(
    gait_insole.insole_r,
    grf_r,
    gait_insole.strike_r,
    gait_insole.off_r,
    gait_trackers.strike_r,
    gait_trackers.off_r,)
norm_left_insole_slow_all, avg_left_insole_slow_all, std_v_insole_slow_all, _, _, _ = process_gait_cycles(
    gait_insole.insole_l,
    grf_l,
    gait_insole.strike_l,
    gait_insole.off_l,
    gait_trackers.strike_l,
    gait_trackers.off_l,)

def plot_graph():
    x_axis = np.linspace(0, 1, 100)

    # Plot Force Plate COP
    plt.figure(figsize=(12, 10))
    plt.suptitle('New COP from Force Plate')

    plt.subplot(2, 2, 1)
    plt.plot(x_axis, avg_right_cop_fp_slow[:, 0], 'r', linewidth=2)
    plt.fill_between(x_axis, avg_right_cop_fp_slow[:, 0] + std_right_cop_fp_slow[:, 0],
                     avg_right_cop_fp_slow[:, 0] - std_right_cop_fp_slow[:, 0], color='r', alpha=0.2)
    plt.title('Right Foot COP X')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(x_axis, avg_right_cop_fp_slow[:, 1], 'r', linewidth=2)
    plt.fill_between(x_axis, avg_right_cop_fp_slow[:, 1] + std_right_cop_fp_slow[:, 1],
                     avg_right_cop_fp_slow[:, 1] - std_right_cop_fp_slow[:, 1], color='r', alpha=0.2)
    plt.title('Right Foot COP Y')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(x_axis, avg_left_cop_fp_slow[:, 0], 'b', linewidth=2)
    plt.fill_between(x_axis, avg_left_cop_fp_slow[:, 0] + std_left_cop_fp_slow[:, 0],
                     avg_left_cop_fp_slow[:, 0] - std_left_cop_fp_slow[:, 0], color='b', alpha=0.2)
    plt.title('Left Foot COP X')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(x_axis, avg_left_cop_fp_slow[:, 1], 'b', linewidth=2)
    plt.fill_between(x_axis, avg_left_cop_fp_slow[:, 1] + std_left_cop_fp_slow[:, 1],
                     avg_left_cop_fp_slow[:, 1] - std_left_cop_fp_slow[:, 1], color='b', alpha=0.2)
    plt.title('Left Foot COP Y')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)

    plt.show()

    # Plot Insole COP
    plt.figure(figsize=(12, 10))
    plt.suptitle('New COP from Insole')

    plt.subplot(2, 2, 1)
    plt.plot(x_axis, avg_right_cop_insole_slow[:, 0], 'r', linewidth=2)
    plt.fill_between(x_axis, avg_right_cop_insole_slow[:, 0] + std_right_cop_insole_slow[:, 0],
                     avg_right_cop_insole_slow[:, 0] - std_right_cop_insole_slow[:, 0], color='r', alpha=0.2)
    plt.title('Right Insole COP X')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(x_axis, avg_right_cop_insole_slow[:, 1], 'r', linewidth=2)
    plt.fill_between(x_axis, avg_right_cop_insole_slow[:, 1] + std_right_cop_insole_slow[:, 1],
                     avg_right_cop_insole_slow[:, 1] - std_right_cop_insole_slow[:, 1], color='r', alpha=0.2)
    plt.title('Right Insole COP Y')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(x_axis, avg_left_cop_insole_slow[:, 0], 'b', linewidth=2)
    plt.fill_between(x_axis, avg_left_cop_insole_slow[:, 0] + std_left_cop_insole_slow[:, 0],
                     avg_left_cop_insole_slow[:, 0] - std_left_cop_insole_slow[:, 0], color='b', alpha=0.2)
    plt.title('Left Insole COP X')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(x_axis, avg_left_cop_insole_slow[:, 1], 'b', linewidth=2)
    plt.fill_between(x_axis, avg_left_cop_insole_slow[:, 1] + std_left_cop_insole_slow[:, 1],
                     avg_left_cop_insole_slow[:, 1] - std_left_cop_insole_slow[:, 1], color='b', alpha=0.2)
    plt.title('Left Insole COP Y')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)

    plt.show()

    # Plot GRF
    plt.figure(figsize=(12, 15))
    plt.suptitle('New Ground Reaction Forces')

    plt.subplot(3, 2, 1)
    plt.plot(x_axis, avg_right_grf_slow[:, 0], 'r', linewidth=2)
    plt.fill_between(x_axis, avg_right_grf_slow[:, 0] + std_right_grf_slow[:, 0],
                     avg_right_grf_slow[:, 0] - std_right_grf_slow[:, 0], color='r', alpha=0.2)
    plt.title('Right GRF X')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Force (N)')
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(x_axis, avg_left_grf_slow[:, 0], 'b', linewidth=2)
    plt.fill_between(x_axis, avg_left_grf_slow[:, 0] + std_left_grf_slow[:, 0],
                     avg_left_grf_slow[:, 0] - std_left_grf_slow[:, 0], color='b', alpha=0.2)
    plt.title('Left GRF X')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Force (N)')
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(x_axis, avg_right_grf_slow[:, 1], 'r', linewidth=2)
    plt.fill_between(x_axis, avg_right_grf_slow[:, 1] + std_right_grf_slow[:, 1],
                     avg_right_grf_slow[:, 1] - std_right_grf_slow[:, 1], color='r', alpha=0.2)
    plt.title('Right GRF Y')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Force (N)')
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(x_axis, avg_left_grf_slow[:, 1], 'b', linewidth=2)
    plt.fill_between(x_axis, avg_left_grf_slow[:, 1] + std_left_grf_slow[:, 1],
                     avg_left_grf_slow[:, 1] - std_left_grf_slow[:, 1], color='b', alpha=0.2)
    plt.title('Left GRF Y')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Force (N)')
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(x_axis, avg_right_grf_slow[:, 2], 'r', linewidth=2)
    plt.fill_between(x_axis, avg_right_grf_slow[:, 2] + std_right_grf_slow[:, 2],
                     avg_right_grf_slow[:, 2] - std_right_grf_slow[:, 2], color='r', alpha=0.2)
    plt.title('Right GRF Z')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Force (N)')
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.plot(x_axis, avg_left_grf_slow[:, 2], 'b', linewidth=2)
    plt.fill_between(x_axis, avg_left_grf_slow[:, 2] + std_left_grf_slow[:, 2],
                     avg_left_grf_slow[:, 2] - std_left_grf_slow[:, 2], color='b', alpha=0.2)
    plt.title('Left GRF Z')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Force (N)')
    plt.grid(True)

    plt.show()

    # Plot GRM
    plt.figure(figsize=(12, 15))
    plt.suptitle('New Ground Reaction Moments')

    plt.subplot(3, 2, 1)
    plt.plot(x_axis, avg_right_grm_slow[:, 0], 'r', linewidth=2)
    plt.fill_between(x_axis, avg_right_grm_slow[:, 0] + std_right_grm_slow[:, 0],
                     avg_right_grm_slow[:, 0] - std_right_grm_slow[:, 0], color='r', alpha=0.2)
    plt.title('Right GRM X')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Moment (N*m)')
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(x_axis, avg_left_grm_slow[:, 0], 'b', linewidth=2)
    plt.fill_between(x_axis, avg_left_grm_slow[:, 0] + std_left_grm_slow[:, 0],
                     avg_left_grm_slow[:, 0] - std_left_grm_slow[:, 0], color='b', alpha=0.2)
    plt.title('Left GRM X')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Moment (N*m)')
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(x_axis, avg_right_grm_slow[:, 1], 'r', linewidth=2)
    plt.fill_between(x_axis, avg_right_grm_slow[:, 1] + std_right_grm_slow[:, 1],
                     avg_right_grm_slow[:, 1] - std_right_grm_slow[:, 1], color='r', alpha=0.2)
    plt.title('Right GRM Y')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Moment (N*m)')
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(x_axis, avg_left_grm_slow[:, 1], 'b', linewidth=2)
    plt.fill_between(x_axis, avg_left_grm_slow[:, 1] + std_left_grm_slow[:, 1],
                     avg_left_grm_slow[:, 1] - std_left_grm_slow[:, 1], color='b', alpha=0.2)
    plt.title('Left GRM Y')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Moment (N*m)')
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(x_axis, avg_right_grm_slow[:, 2], 'r', linewidth=2)
    plt.fill_between(x_axis, avg_right_grm_slow[:, 2] + std_right_grm_slow[:, 2],
                     avg_right_grm_slow[:, 2] - std_right_grm_slow[:, 2], color='r', alpha=0.2)
    plt.title('Right GRM Z')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Moment (N*m)')
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.plot(x_axis, avg_left_grm_slow[:, 2], 'b', linewidth=2)
    plt.fill_between(x_axis, avg_left_grm_slow[:, 2] + std_left_grm_slow[:, 2],
                     avg_left_grm_slow[:, 2] - std_left_grm_slow[:, 2], color='b', alpha=0.2)
    plt.title('Left GRM Z')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Moment (N*m)')
    plt.grid(True)

    plt.show()

if __name__ == '__main__':
    plot_graph()
