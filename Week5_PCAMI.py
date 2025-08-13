import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

# Load data
data = loadmat('C:/Users/XuanL/Documents/MATLAB/Gait Analysis/ABLE1_112024_1.0.mat', struct_as_record=False,
               squeeze_me=True)

# Get gait data
gait_insole = data['gait_insole']
gait_trackers = data['gait_trackers']

def lowpass_filter(data, cutoff=10, fs=100, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    # Butterworth Filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Zero phase filtering
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data
grf_r = lowpass_filter(gait_trackers.force_plate_ds_r[:, 0:3], cutoff=10, fs=100, order=2)
grm_r = lowpass_filter(gait_trackers.force_plate_ds_r[:, 0:3], cutoff=10, fs=100, order=2)
grf_l = lowpass_filter(gait_trackers.force_plate_ds_l[:, 0:3], cutoff=10, fs=100, order=2)
grm_l = lowpass_filter(gait_trackers.force_plate_ds_l[:, 0:3], cutoff=10, fs=100, order=2)

def pcami_cycle(X_insole, Y_target, n_components=5, top_k_features=10):
    pca = PCA(n_components=top_k_features)
    scaler = StandardScaler()
    X_insole = scaler.fit_transform(X_insole)
    X_pca = pca.fit_transform(X_insole)

    explained = pca.explained_variance_ratio_  # shape: (50,)
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
#        X_splits = np.split(X_reduced, split_indices[:-1])  # 得到每个周期降维后的数据
#        insole_stance = X_splits
    num_valid = len(insole_stance)

    if num_valid == 0:
        raise ValueError('No valid stance phases found')

    num_points = 100
    raw_insole = np.zeros((num_points, insole_data.shape[1], num_valid))
    raw_force = np.zeros((num_points, force_data.shape[1], num_valid))

    for i in range(num_valid):  # Every step

        insole_cycle = insole_stance[i] # eg. (29, 1024)
        force_cycle = force_stance[i]

        t_insole = np.arange(insole_cycle.shape[0]) * 0.025 # eg. 29个点均匀分布在0,1
        t_force = np.arange(force_cycle.shape[0]) * 0.01

        combined_times = np.unique(np.concatenate((t_insole, t_force))) # eg. 105
        combined_times.sort()

        insole_interp = np.zeros((len(combined_times), insole_cycle.shape[1]))  #eg. 105, 1024
        for j in range(insole_cycle.shape[1]):
            f_nearest = interpolate.interp1d(t_insole, insole_cycle[:, j], kind='nearest', fill_value='extrapolate')
            insole_interp[:, j] = f_nearest(combined_times)

        force_interp = np.zeros((len(combined_times), force_cycle.shape[1]))
        for j in range(force_cycle.shape[1]):
            f_linear = interpolate.interp1d(t_force, force_cycle[:, j], kind='linear', fill_value='extrapolate')
            force_interp[:, j] = f_linear(combined_times)

        standard_time = np.linspace(combined_times[0], combined_times[-1], num_points)
        for j in range(insole_cycle.shape[1]):
            f_final_insole = interpolate.interp1d(combined_times, insole_interp[:, j], kind='linear', fill_value='extrapolate')
            raw_insole[:, j, i] = f_final_insole(standard_time)

        for j in range(force_cycle.shape[1]):
            f_final_force = interpolate.interp1d(combined_times, force_interp[:, j], kind='linear', fill_value='extrapolate')
            raw_force[:, j, i] = f_final_force(standard_time)
    if pcami:
        X_insole = raw_insole.reshape(num_points * num_valid, -1)  # [100*N, channels]
        #print('raw_insole', raw_insole)     #100,1024,100
        Y_target = raw_force.reshape(num_points * num_valid, -1)  # [100*N, force_channels]
        n_components = 5
        top_k_features = 10
        X_reduced = pcami_cycle(X_insole, Y_target, n_components=n_components, top_k_features=top_k_features)
        print('X_reduced', X_reduced)  # 10000,5
        norm_insole = X_reduced.reshape(num_points, n_components, num_valid)  # 5 是 n_components
    else:
        norm_insole = raw_insole

    avg_insole = np.mean(norm_insole, axis=2)
    std_insole = np.std(norm_insole, axis=2)

    avg_force = np.mean(raw_force, axis=2)
    std_force = np.std(raw_force, axis=2)

    return norm_insole, avg_insole, std_insole, raw_force, avg_force, std_force

# Process Force Plate GRF
# Right foot GRF
norm_right_grf, avg_right_grf, std_right_grf, _, _, _ = process_gait_cycles(
    grf_r,
    grf_r,
    gait_trackers.strike_r,
    gait_trackers.off_r,
    gait_trackers.strike_r,
    gait_trackers.off_r)
print(norm_right_grf.shape)
# Left foot GRF
norm_left_grf, avg_left_grf, std_left_grf, _, _, _ = process_gait_cycles(
    grf_l,
    grf_l,
    gait_trackers.strike_l,
    gait_trackers.off_l,
    gait_trackers.strike_l,
    gait_trackers.off_l)

# Process Force Plate GRM
# Right foot GRM
norm_right_grm, avg_right_grm, std_right_grm, _, _, _ = process_gait_cycles(
    grm_r,
    grm_r,
    gait_trackers.strike_r,
    gait_trackers.off_r,
    gait_trackers.strike_r,
    gait_trackers.off_r
)

# Left foot GRM
norm_left_grm, avg_left_grm, std_left_grm, _, _, _ = process_gait_cycles(
    grm_l,
    grm_l,
    gait_trackers.strike_l,
    gait_trackers.off_l,
    gait_trackers.strike_l,
    gait_trackers.off_l
)

# Process Force Plate COP
# Right foot COP
norm_right_cop_fp, avg_right_cop_fp, std_right_cop_fp, _, _, _ = process_gait_cycles(
    gait_trackers.force_plate_ds_r[:, 6:8],
    gait_trackers.force_plate_ds_r[:, 6:8],
    gait_trackers.strike_r,
    gait_trackers.off_r,
    gait_trackers.strike_r,
    gait_trackers.off_r
)

# Left foot COP
norm_left_cop_fp, avg_left_cop_fp, std_left_cop_fp, _, _, _ = process_gait_cycles(
    gait_trackers.force_plate_ds_l[:, 6:8],
    gait_trackers.force_plate_ds_l[:, 6:8],
    gait_trackers.strike_l,
    gait_trackers.off_l,
    gait_trackers.strike_l,
    gait_trackers.off_l
)

# Process Insole COP
# Right foot insole COP
right_cop_data = np.column_stack((gait_insole.cop_x_r, gait_insole.cop_y_r))
norm_right_cop_insole, avg_right_cop_insole, std_right_cop_insole, _, _, _ = process_gait_cycles(
    right_cop_data,
    gait_trackers.force_plate_ds_r[:, 6:8],  # force plate COP
    gait_insole.strike_r,
    gait_insole.off_r,
    gait_trackers.strike_r,
    gait_trackers.off_r
)

# Left foot insole COP
left_cop_data = np.column_stack((gait_insole.cop_x_l, gait_insole.cop_y_l))
norm_left_cop_insole, avg_left_cop_insole, std_left_cop_insole, _, _, _ = process_gait_cycles(
    left_cop_data,
    gait_trackers.force_plate_ds_l[:, 6:8],  # force plate COP
    gait_insole.strike_l,
    gait_insole.off_l,
    gait_trackers.strike_l,
    gait_trackers.off_l
)

# Right foot insole pp
norm_right_insole_pp, avg_right_insole_pp, std_right_insole_pp, _, _, _ = process_gait_cycles(
    gait_insole.pp_r.reshape(-1, 1),
    grf_r,  # Force plate vertical GRF（Z方向）可以作为reference
    gait_insole.strike_r,
    gait_insole.off_r,
    gait_trackers.strike_r,
    gait_trackers.off_r
)

# Left foot insole pp
norm_left_insole_pp, avg_left_insole_pp, std_left_insole_pp, _, _, _ = process_gait_cycles(
    gait_insole.pp_l.reshape(-1, 1),
    grf_l,
    gait_insole.strike_l,
    gait_insole.off_l,
    gait_trackers.strike_l,
    gait_trackers.off_l
)

# Used in WNN
norm_right_insole, avg_right_insole, std_right_insole, _, _, _ = process_gait_cycles(
    gait_insole.insole_r,
    grf_r,
    gait_insole.strike_r,
    gait_insole.off_r,
    gait_trackers.strike_r,
    gait_trackers.off_r,
    pcami=True)
norm_left_insole, avg_left_insole, std_v_insole, _, _, _ = process_gait_cycles(
    gait_insole.insole_l,
    grf_l,
    gait_insole.strike_l,
    gait_insole.off_l,
    gait_trackers.strike_l,
    gait_trackers.off_l,
    pcami=True)
print(norm_right_insole.shape, norm_left_insole.shape)

# New features for week 4
norm_right_fti, avg_right_fti, std_right_fti, _, _, _ = process_gait_cycles(
    gait_insole.fti_r.reshape(-1, 1),
    grf_r,
    gait_insole.strike_r,
    gait_insole.off_r,
    gait_trackers.strike_r,
    gait_trackers.off_r,
)
print(norm_right_fti.shape) #100,1,103
norm_left_fti, avg_left_fti, std_left_fti, _, _, _ = process_gait_cycles(
    gait_insole.fti_l.reshape(-1, 1),
    grf_l,
    gait_insole.strike_l,
    gait_insole.off_l,
    gait_trackers.strike_l,
    gait_trackers.off_l,
)

norm_right_ft, avg_right_ft, std_right_ft, _, _, _ = process_gait_cycles(
    gait_insole.foot_trace_r.reshape(-1, 1),
    grf_r,
    gait_insole.strike_r,
    gait_insole.off_r,
    gait_trackers.strike_r,
    gait_trackers.off_r,
)
norm_left_ft, avg_left_ft, std_left_ft, _, _, _ = process_gait_cycles(
    gait_insole.foot_trace_l.reshape(-1, 1),
    grf_l,
    gait_insole.strike_l,
    gait_insole.off_l,
    gait_trackers.strike_l,
    gait_trackers.off_l,
)

norm_right_contact, avg_right_contact, std_right_contact, _, _, _ = process_gait_cycles(
    gait_insole.cont_area_r.reshape(-1, 1),
    grf_r,
    gait_insole.strike_r,
    gait_insole.off_r,
    gait_trackers.strike_r,
    gait_trackers.off_r,
)
norm_left_contact, avg_left_contact, std_left_contact, _, _, _ = process_gait_cycles(
    gait_insole.cont_area_l.reshape(-1, 1),
    grf_l,
    gait_insole.strike_l,
    gait_insole.off_l,
    gait_trackers.strike_l,
    gait_trackers.off_l,
)
print(norm_right_contact.shape)

def plot_graph():
    x_axis = np.arange(100)

    # Plot Force Plate COP
    plt.figure(figsize=(12, 10))
    plt.suptitle('New COP from Force Plate')

    plt.subplot(2, 2, 1)
    plt.plot(x_axis, avg_right_cop_fp[:, 0], 'r', linewidth=2)
    plt.fill_between(x_axis, avg_right_cop_fp[:, 0] + std_right_cop_fp[:, 0],
                     avg_right_cop_fp[:, 0] - std_right_cop_fp[:, 0], color='r', alpha=0.2)
    plt.title('Right Foot COP X')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(x_axis, avg_right_cop_fp[:, 1], 'r', linewidth=2)
    plt.fill_between(x_axis, avg_right_cop_fp[:, 1] + std_right_cop_fp[:, 1],
                     avg_right_cop_fp[:, 1] - std_right_cop_fp[:, 1], color='r', alpha=0.2)
    plt.title('Right Foot COP Y')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(x_axis, avg_left_cop_fp[:, 0], 'b', linewidth=2)
    plt.fill_between(x_axis, avg_left_cop_fp[:, 0] + std_left_cop_fp[:, 0],
                     avg_left_cop_fp[:, 0] - std_left_cop_fp[:, 0], color='b', alpha=0.2)
    plt.title('Left Foot COP X')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(x_axis, avg_left_cop_fp[:, 1], 'b', linewidth=2)
    plt.fill_between(x_axis, avg_left_cop_fp[:, 1] + std_left_cop_fp[:, 1],
                     avg_left_cop_fp[:, 1] - std_left_cop_fp[:, 1], color='b', alpha=0.2)
    plt.title('Left Foot COP Y')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)

    plt.show()

    # Plot Insole COP
    plt.figure(figsize=(12, 10))
    plt.suptitle('New COP from Insole')

    plt.subplot(2, 2, 1)
    plt.plot(x_axis, avg_right_cop_insole[:, 0], 'r', linewidth=2)
    plt.fill_between(x_axis, avg_right_cop_insole[:, 0] + std_right_cop_insole[:, 0],
                     avg_right_cop_insole[:, 0] - std_right_cop_insole[:, 0], color='r', alpha=0.2)
    plt.title('Right Insole COP X')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(x_axis, avg_right_cop_insole[:, 1], 'r', linewidth=2)
    plt.fill_between(x_axis, avg_right_cop_insole[:, 1] + std_right_cop_insole[:, 1],
                     avg_right_cop_insole[:, 1] - std_right_cop_insole[:, 1], color='r', alpha=0.2)
    plt.title('Right Insole COP Y')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(x_axis, avg_left_cop_insole[:, 0], 'b', linewidth=2)
    plt.fill_between(x_axis, avg_left_cop_insole[:, 0] + std_left_cop_insole[:, 0],
                     avg_left_cop_insole[:, 0] - std_left_cop_insole[:, 0], color='b', alpha=0.2)
    plt.title('Left Insole COP X')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(x_axis, avg_left_cop_insole[:, 1], 'b', linewidth=2)
    plt.fill_between(x_axis, avg_left_cop_insole[:, 1] + std_left_cop_insole[:, 1],
                     avg_left_cop_insole[:, 1] - std_left_cop_insole[:, 1], color='b', alpha=0.2)
    plt.title('Left Insole COP Y')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Position (mm)')
    plt.grid(True)

    plt.show()

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

    # Plot GRM
    plt.figure(figsize=(12, 15))
    plt.suptitle('New Ground Reaction Moments')

    plt.subplot(3, 2, 1)
    plt.plot(x_axis, avg_right_grm[:, 0], 'r', linewidth=2)
    plt.fill_between(x_axis, avg_right_grm[:, 0] + std_right_grm[:, 0],
                     avg_right_grm[:, 0] - std_right_grm[:, 0], color='r', alpha=0.2)
    plt.title('Right GRM X')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Moment (N*m)')
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(x_axis, avg_left_grm[:, 0], 'b', linewidth=2)
    plt.fill_between(x_axis, avg_left_grm[:, 0] + std_left_grm[:, 0],
                     avg_left_grm[:, 0] - std_left_grm[:, 0], color='b', alpha=0.2)
    plt.title('Left GRM X')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Moment (N*m)')
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(x_axis, avg_right_grm[:, 1], 'r', linewidth=2)
    plt.fill_between(x_axis, avg_right_grm[:, 1] + std_right_grm[:, 1],
                     avg_right_grm[:, 1] - std_right_grm[:, 1], color='r', alpha=0.2)
    plt.title('Right GRM Y')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Moment (N*m)')
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(x_axis, avg_left_grm[:, 1], 'b', linewidth=2)
    plt.fill_between(x_axis, avg_left_grm[:, 1] + std_left_grm[:, 1],
                     avg_left_grm[:, 1] - std_left_grm[:, 1], color='b', alpha=0.2)
    plt.title('Left GRM Y')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Moment (N*m)')
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(x_axis, avg_right_grm[:, 2], 'r', linewidth=2)
    plt.fill_between(x_axis, avg_right_grm[:, 2] + std_right_grm[:, 2],
                     avg_right_grm[:, 2] - std_right_grm[:, 2], color='r', alpha=0.2)
    plt.title('Right GRM Z')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Moment (N*m)')
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.plot(x_axis, avg_left_grm[:, 2], 'b', linewidth=2)
    plt.fill_between(x_axis, avg_left_grm[:, 2] + std_left_grm[:, 2],
                     avg_left_grm[:, 2] - std_left_grm[:, 2], color='b', alpha=0.2)
    plt.title('Left GRM Z')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('Moment (N*m)')
    plt.grid(True)

    plt.show()

    # Plot Insole PP
    plt.figure(figsize=(8, 6))
    plt.suptitle('New Insole PP')

    plt.subplot(2,1,1)
    plt.plot(x_axis, avg_right_insole_pp[:,0], 'r', linewidth=2)
    plt.title('Right Foot Insole PP')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('ADC Value')
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(x_axis, avg_left_insole_pp[:,0], 'b', linewidth=2)
    plt.title('Left Foot Insole PP')
    plt.xlabel('Stance Phase (%)')
    plt.ylabel('ADC Value')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_graph()
