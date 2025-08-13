import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

# Load data
data = loadmat('C:/Users/XuanL/Documents/MATLAB/Gait Analysis/ABLE1_112024_1.0.mat', struct_as_record=False,
               squeeze_me=True)
# Get gait data
gait_insole = data['gait_insole']
gait_trackers = data['gait_trackers']

def extract_stance_phase(data, strikes, offs):
    """
    Extract stance phase segments from data.

    Parameters:
    data -- (frames, channels) 原始信号
    strikes -- heel strike indices
    offs -- toe off indices

    Returns:
    stance_data -- all stance data concatenated (n_frames, channels)
    stance_list -- list of every cycle (every element is (n_frames_in_cycle, channels))
    """
    num_cycles = min(len(strikes), len(offs))
    stance_list = []

    for i in range(num_cycles):
        if strikes[i] < offs[i]:
            stance_list.append(data[strikes[i]:offs[i], :])

    if len(stance_list) == 0:
        raise ValueError('No valid stance phases found')

    # concatenate all stance data
    stance_data = np.vstack(stance_list)

    return stance_data, stance_list   # stance data: all data eg.(4920, 1024), stance_list: list of cycles eg. (103, 1024)

def pca_mi_reduction(X_insole, Y_target, n_components=5, top_k_features=10):
    """
    Perform PCA-MI reduction: select directions in X_insole that best preserve mutual information with Y_target.

    Parameters:
    X_insole -- (N_samples, N_insole_features)
    Y_target -- (N_samples, N_target_features)
    n_components -- Number of principal components
    top_k_features -- After PCA, how many features for MI

    Returns:
    X_reduced -- (N_samples, n_components)
    pca_model
    selected_components -- selected indices of PCA components based on mutual information
    """

    pca = PCA(n_components=min(top_k_features, X_insole.shape[1]))
    X_pca = pca.fit_transform(X_insole)

    mi_scores = []
    for i in range(X_pca.shape[1]):
        mi = 0
        for j in range(Y_target.shape[1]):
            # mutual_info_regression need 1D input
            mi += mutual_info_regression(X_pca[:, [i]], Y_target[:, j], discrete_features=False)[0]
        mi_scores.append(mi)

    mi_scores = np.array(mi_scores)

    selected_indices = np.argsort(mi_scores)[-n_components:]  # Maximum mutual information indices
    selected_indices.sort()  # low to high for PCA component order

    X_reduced = X_pca[:, selected_indices]

    return X_reduced, pca, selected_indices

def split_reduced_to_cycles(X_reduced, original_stance_list):
    """
    Split reduced data back to cycles based on original stance lengths.

    Parameters:
    X_reduced -- (N_total_samples, n_components)
    original_stance_list -- Original stance list, each element is (n_frames_in_cycle, n_features)

    Returns:
    reduced_stance_list -- each element: (n_frames_in_cycle, n_components)
    """
    reduced_stance_list = []
    start_idx = 0
    for cycle in original_stance_list:
        length = cycle.shape[0]
        reduced_stance_list.append(X_reduced[start_idx:start_idx + length, :])
        start_idx += length

    return reduced_stance_list

def interpolate_gait(insole_stance, stance_list, force_stance, force_list):
    num_points = 100
    num_valid = len(stance_list)
    norm_insole = np.zeros((num_points, insole_stance.shape[1], num_valid))
    norm_force = np.zeros((num_points, force_stance.shape[1], num_valid))

    for i in range(num_valid):
        insole_cycle = stance_list[i]
        force_cycle = force_list[i]

        t_insole = np.linspace(0, 1, insole_cycle.shape[0])
        t_force = np.linspace(0, 1, force_cycle.shape[0])

        combined_times = np.unique(np.concatenate((t_insole, t_force)))
        combined_times.sort()

        insole_interp = np.zeros((len(combined_times), insole_cycle.shape[1]))
        for j in range(insole_cycle.shape[1]):
            f_nearest = interpolate.interp1d(t_insole, insole_cycle[:, j], kind='nearest', fill_value='extrapolate')
            insole_interp[:, j] = f_nearest(combined_times)

        force_interp = np.zeros((len(combined_times), force_cycle.shape[1]))
        for j in range(force_cycle.shape[1]):
            f_linear = interpolate.interp1d(t_force, force_cycle[:, j], kind='linear', fill_value='extrapolate')
            force_interp[:, j] = f_linear(combined_times)

        standard_time = np.linspace(0, 1, num_points)

        for j in range(insole_cycle.shape[1]):
            f_final_insole = interpolate.interp1d(combined_times, insole_interp[:, j], kind='linear', fill_value='extrapolate')
            norm_insole[:, j, i] = f_final_insole(standard_time)

        for j in range(force_cycle.shape[1]):
            f_final_force = interpolate.interp1d(combined_times, force_interp[:, j], kind='linear', fill_value='extrapolate')
            norm_force[:, j, i] = f_final_force(standard_time)

    avg_insole = np.mean(norm_insole, axis=2)
    std_insole = np.std(norm_insole, axis=2)

    avg_force = np.mean(norm_force, axis=2)
    std_force = np.std(norm_force, axis=2)

    return norm_insole, avg_insole, std_insole, norm_force, avg_force, std_force

# insole_r & grf_r
insole_pp_r_data, insole_pp_r_list = extract_stance_phase(gait_insole.insole_r, gait_insole.strike_r, gait_insole.off_r)
fp_grf_r_data, fp_grf_r_list = extract_stance_phase(gait_trackers.force_plate_ds_r[:, 0:3], gait_trackers.strike_r, gait_trackers.off_r)
print(insole_pp_r_data.shape, insole_pp_r_list[0].shape)  # (2963, 1024)  (34, 1024)*103
print(fp_grf_r_data.shape, fp_grf_r_list[0].shape)  # (2963, 3)  (34, 3)*103

insole_r, _, _ = pca_mi_reduction(insole_pp_r_data, fp_grf_r_data)
insole_pp_r_list = split_reduced_to_cycles(insole_r, insole_pp_r_list)
norm_right_insole_pp, avg_right_insole_pp, std_right_insole_pp, _, _, _ = interpolate_gait(insole_pp_r_data, insole_pp_r_list, fp_grf_r_data, fp_grf_r_list)
print(norm_right_insole_pp.shape)

# insole_l & grf_l
insole_pp_l_data, insole_pp_l_list = extract_stance_phase(gait_insole.insole_l, gait_insole.strike_l, gait_insole.off_l)
fp_grf_l_data, fp_grf_l_list = extract_stance_phase(gait_trackers.force_plate_ds_l[:, 0:3], gait_trackers.strike_l, gait_trackers.off_l)
insole_l, _, _ = pca_mi_reduction(insole_pp_l_data, fp_grf_l_data)
insole_pp_l_list = split_reduced_to_cycles(insole_l, insole_pp_l_list)
norm_left_insole_pp, avg_left_insole_pp, std_left_insole_pp, _, _, _ = interpolate_gait(insole_pp_l_data, insole_pp_l_list, fp_grf_l_data, fp_grf_l_list)

_, _, _, norm_right_grf, avg_right_grf, std_right_grf = interpolate_gait(fp_grf_r_data, fp_grf_r_list, fp_grf_r_data, fp_grf_r_list)
_, _, _, norm_left_grf, avg_left_grf, std_left_grf = interpolate_gait(fp_grf_l_data, fp_grf_l_list, fp_grf_l_data, fp_grf_l_list)

x_axis = np.arange(100)
# Plot GRF
plt.figure(figsize=(12, 15))
plt.suptitle('New Ground Reaction Forces')

plt.subplot(3, 2, 1)
plt.plot(x_axis, avg_right_grf[:, 0], 'r', linewidth=2)
plt.fill_between(x_axis, avg_right_grf[:, 0] + std_right_grf[:, 0],
                 avg_right_grf[:, 0] - std_right_grf[:, 0], color='r', alpha=0.2)
plt.title('Right GRF X')
plt.xlabel('Stance Phase (%)')
plt.ylabel('Force (N)')
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(x_axis, avg_left_grf[:, 0], 'b', linewidth=2)
plt.fill_between(x_axis, avg_left_grf[:, 0] + std_left_grf[:, 0],
                 avg_left_grf[:, 0] - std_left_grf[:, 0], color='b', alpha=0.2)
plt.title('Left GRF X')
plt.xlabel('Stance Phase (%)')
plt.ylabel('Force (N)')
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(x_axis, avg_right_grf[:, 1], 'r', linewidth=2)
plt.fill_between(x_axis, avg_right_grf[:, 1] + std_right_grf[:, 1],
                 avg_right_grf[:, 1] - std_right_grf[:, 1], color='r', alpha=0.2)
plt.title('Right GRF Y')
plt.xlabel('Stance Phase (%)')
plt.ylabel('Force (N)')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(x_axis, avg_left_grf[:, 1], 'b', linewidth=2)
plt.fill_between(x_axis, avg_left_grf[:, 1] + std_left_grf[:, 1],
                 avg_left_grf[:, 1] - std_left_grf[:, 1], color='b', alpha=0.2)
plt.title('Left GRF Y')
plt.xlabel('Stance Phase (%)')
plt.ylabel('Force (N)')
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(x_axis, avg_right_grf[:, 2], 'r', linewidth=2)
plt.fill_between(x_axis, avg_right_grf[:, 2] + std_right_grf[:, 2],
                 avg_right_grf[:, 2] - std_right_grf[:, 2], color='r', alpha=0.2)
plt.title('Right GRF Z')
plt.xlabel('Stance Phase (%)')
plt.ylabel('Force (N)')
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(x_axis, avg_left_grf[:, 2], 'b', linewidth=2)
plt.fill_between(x_axis, avg_left_grf[:, 2] + std_left_grf[:, 2],
                 avg_left_grf[:, 2] - std_left_grf[:, 2], color='b', alpha=0.2)
plt.title('Left GRF Z')
plt.xlabel('Stance Phase (%)')
plt.ylabel('Force (N)')
plt.grid(True)

plt.show()