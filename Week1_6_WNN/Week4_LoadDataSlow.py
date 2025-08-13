import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
# Load data
data_slow = loadmat('C:/Users/XuanL/Documents/MATLAB/Gait Analysis/ABLE1_112024_0.75.mat', struct_as_record=False,
                    squeeze_me=True)

# Get gait data
gait_insole_slow = data_slow['gait_insole']
gait_trackers_slow = data_slow['gait_trackers']

def pcami_cycle(X_insole, Y_target, n_components=5, top_k_features=10):
    pca = PCA(n_components=min(top_k_features, X_insole.shape[1]))
    print(X_insole.shape[1])
    scaler = StandardScaler()
    X_insole = scaler.fit_transform(X_insole)
    X_pca = pca.fit_transform(X_insole)

    mi_scores = []
    for i in range(X_pca.shape[1]):
        mi = sum(mutual_info_regression(X_pca[:, [i]], Y_target[:, j])[0] for j in range(Y_target.shape[1]))
        mi_scores.append(mi)

    selected = np.argsort(mi_scores)[-n_components:]
    selected.sort()

    return X_pca[:, selected]

def process_gait_cycles(insole_data, force_data, strikes, offs, pcami=False):
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
    num_cycles = min(len(strikes), len(offs))
    insole_stance = []
    force_stance = []

    # Extract stance phases
    for i in range(num_cycles):
        if strikes[i] < offs[i]:
            insole_stance.append(insole_data[strikes[i]:offs[i], :])
            force_stance.append(force_data[strikes[i]:offs[i], :])

    num_valid = len(insole_stance)

    if num_valid == 0:
        raise ValueError('No valid stance phases found')

    num_points = 100
    raw_insole = np.zeros((num_points, insole_data.shape[1], num_valid))
    raw_force = np.zeros((num_points, force_data.shape[1], num_valid))

    for i in range(num_valid):

        insole_cycle = insole_stance[i]
        force_cycle = force_stance[i]

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
            raw_insole[:, j, i] = f_final_insole(standard_time)

        for j in range(force_cycle.shape[1]):
            f_final_force = interpolate.interp1d(combined_times, force_interp[:, j], kind='linear', fill_value='extrapolate')
            raw_force[:, j, i] = f_final_force(standard_time)

    if pcami:
        X_insole = raw_insole.reshape(num_points * num_valid, -1)  # [100*N, channels]
        Y_target = raw_force.reshape(num_points * num_valid, -1)  # [100*N, force_channels]

        n_components = 7
        top_k_features = 50
        X_reduced = pcami_cycle(X_insole, Y_target, n_components=n_components, top_k_features=top_k_features)

        norm_insole = X_reduced.reshape(num_points, n_components, num_valid)  # 5 = n_components
    else:
        norm_insole = raw_insole

    avg_insole = np.mean(norm_insole, axis=2)
    std_insole = np.std(norm_insole, axis=2)

    avg_force = np.mean(raw_force, axis=2)
    std_force = np.std(raw_force, axis=2)

    return norm_insole, avg_insole, std_insole, raw_force, avg_force, std_force

# Process Force Plate GRF
# Right foot GRF
norm_right_grf_slow, avg_right_grf_slow, std_right_grf_slow, _, _, _ = process_gait_cycles(
    gait_trackers_slow.force_plate_ds_r[:, 0:3],
    gait_trackers_slow.force_plate_ds_r[:, 0:3],
    gait_trackers_slow.strike_r,
    gait_trackers_slow.off_r)
print(norm_right_grf_slow.shape)

# Left foot GRF
norm_left_grf_slow, avg_left_grf_slow, std_left_grf_slow, _, _, _ = process_gait_cycles(
    gait_trackers_slow.force_plate_ds_l[:, 0:3],
    gait_trackers_slow.force_plate_ds_l[:, 0:3],
    gait_trackers_slow.strike_l,
    gait_trackers_slow.off_l)

# Process Force Plate GRM
# Right foot GRM
norm_right_grm_slow, avg_right_grm_slow, std_right_grm_slow, _, _, _ = process_gait_cycles(
    gait_trackers_slow.force_plate_ds_r[:, 3:6],
    gait_trackers_slow.force_plate_ds_r[:, 3:6],
    gait_trackers_slow.strike_r,
    gait_trackers_slow.off_r)

# Left foot GRM
norm_left_grm_slow, avg_left_grm_slow, std_left_grm_slow, _, _, _ = process_gait_cycles(
    gait_trackers_slow.force_plate_ds_l[:, 3:6],
    gait_trackers_slow.force_plate_ds_l[:, 3:6],
    gait_trackers_slow.strike_l,
    gait_trackers_slow.off_l)

# Process Force Plate COP
# Right foot COP
norm_right_cop_fp_slow, avg_right_cop_fp_slow, std_right_cop_fp_slow, _, _, _ = process_gait_cycles(
    gait_trackers_slow.force_plate_ds_r[:, 6:8],
    gait_trackers_slow.force_plate_ds_r[:, 6:8],
    gait_trackers_slow.strike_r,
    gait_trackers_slow.off_r)

# Left foot COP
norm_left_cop_fp_slow, avg_left_cop_fp_slow, std_left_cop_fp_slow, _, _, _ = process_gait_cycles(
    gait_trackers_slow.force_plate_ds_l[:, 6:8],
    gait_trackers_slow.force_plate_ds_l[:, 6:8],
    gait_trackers_slow.strike_l,
    gait_trackers_slow.off_l)

# Process Insole COP
# Right foot insole COP
right_cop_data_slow = np.column_stack((gait_insole_slow.cop_x_r, gait_insole_slow.cop_y_r))
norm_right_cop_insole_slow, avg_right_cop_insole_slow, std_right_cop_insole_slow, _, _, _ = process_gait_cycles(
    right_cop_data_slow,
    gait_trackers_slow.force_plate_ds_r[:, 6:8],
    gait_insole_slow.strike_r,
    gait_insole_slow.off_r)

# Left foot insole COP
left_cop_data_slow = np.column_stack((gait_insole_slow.cop_x_l, gait_insole_slow.cop_y_l))
norm_left_cop_insole_slow, avg_left_cop_insole_slow, std_left_cop_insole_slow, _, _, _ = process_gait_cycles(
    left_cop_data_slow,
    gait_trackers_slow.force_plate_ds_l[:, 6:8],
    gait_insole_slow.strike_l,
    gait_insole_slow.off_l)

# Right foot insole pp
norm_right_insole_pp_slow, avg_right_insole_pp_slow, std_right_insole_pp_slow, _, _, _ = process_gait_cycles(
    gait_insole_slow.pp_r.reshape(-1, 1),
    gait_trackers_slow.force_plate_ds_r[:, 0:3],
    gait_insole_slow.strike_r,
    gait_insole_slow.off_r)

# Left foot insole pp
norm_left_insole_pp_slow, avg_left_insole_pp_slow, std_left_insole_pp_slow, _, _, _ = process_gait_cycles(
    gait_insole_slow.pp_l.reshape(-1, 1),
    gait_trackers_slow.force_plate_ds_l[:, 0:3],
    gait_insole_slow.strike_l,
    gait_insole_slow.off_l)

# Used in WNN
norm_right_insole_slow, avg_right_insole_slow, std_right_insole_slow, _, _, _ = process_gait_cycles(
    gait_insole_slow.insole_r,
    gait_trackers_slow.force_plate_ds_r[:, 0:3],
    gait_insole_slow.strike_r,
    gait_insole_slow.off_r,
    pcami=True)
norm_left_insole_slow, avg_left_insole_slow, std_v_insole_slow, _, _, _ = process_gait_cycles(
    gait_insole_slow.insole_l,
    gait_trackers_slow.force_plate_ds_l[:, 0:3],
    gait_insole_slow.strike_l,
    gait_insole_slow.off_l,
    pcami=True)
print(norm_right_insole_slow.shape, norm_left_insole_slow.shape)

# New features for week 4
norm_right_fti_slow, avg_right_fti_slow, std_right_fti_slow, _, _, _ = process_gait_cycles(
    gait_insole_slow.fti_r.reshape(-1, 1),
    gait_trackers_slow.force_plate_ds_r[:, 0:3],
    gait_insole_slow.strike_r,
    gait_insole_slow.off_r)
print(norm_right_fti_slow.shape)

norm_left_fti_slow, avg_left_fti_slow, std_left_fti_slow, _, _, _ = process_gait_cycles(
    gait_insole_slow.fti_l.reshape(-1, 1),
    gait_trackers_slow.force_plate_ds_l[:, 0:3],
    gait_insole_slow.strike_l,
    gait_insole_slow.off_l)

norm_right_ft_slow, avg_right_ft_slow, std_right_ft_slow, _, _, _ = process_gait_cycles(
    gait_insole_slow.foot_trace_r.reshape(-1, 1),
    gait_trackers_slow.force_plate_ds_r[:, 0:3],
    gait_insole_slow.strike_r,
    gait_insole_slow.off_r)

norm_left_ft_slow, avg_left_ft_slow, std_left_ft_slow, _, _, _ = process_gait_cycles(
    gait_insole_slow.foot_trace_l.reshape(-1, 1),
    gait_trackers_slow.force_plate_ds_l[:, 0:3],
    gait_insole_slow.strike_l,
    gait_insole_slow.off_l)

norm_right_contact_slow, avg_right_contact_slow, std_right_contact_slow, _, _, _ = process_gait_cycles(
    gait_insole_slow.cont_area_r.reshape(-1, 1),
    gait_trackers_slow.force_plate_ds_r[:, 0:3],
    gait_insole_slow.strike_r,
    gait_insole_slow.off_r)

norm_left_contact_slow, avg_left_contact_slow, std_left_contact_slow, _, _, _ = process_gait_cycles(
    gait_insole_slow.cont_area_l.reshape(-1, 1),
    gait_trackers_slow.force_plate_ds_l[:, 0:3],
    gait_insole_slow.strike_l,
    gait_insole_slow.off_l)

print(norm_right_contact_slow.shape)