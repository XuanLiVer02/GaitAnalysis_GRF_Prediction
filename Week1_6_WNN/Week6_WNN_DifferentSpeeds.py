from Week6_NewInterpolation import *
from Week6_LoadFast import *
from Week6_LoadSlow import *
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

steps = 100
steps_slow = 92
steps_fast = 119
norm_right_cop_insole_all = np.concatenate([norm_right_cop_insole[:, :, :steps], norm_right_cop_insole_slow[:, :, :steps_slow], norm_right_cop_insole_fast[:, :, :steps_fast]], axis=2)
print('norm_right_insole_fast', norm_right_insole_fast.shape)
norm_right_insole_all = np.concatenate([norm_right_insole[:, :, :steps], norm_right_insole_slow[:, :, :steps_slow], norm_right_insole_fast[:, :, :steps_fast]], axis=2)
norm_right_insole_pp_all = np.concatenate([norm_right_insole_pp[:, :, :steps], norm_right_insole_pp_slow[:, :, :steps_slow], norm_right_insole_pp_fast[:, :, :steps_fast]], axis=2)
norm_right_fti_all = np.concatenate([norm_right_fti[:, :, :steps], norm_right_fti_slow[:, :, :steps_slow], norm_right_fti_fast[:, :, :steps_fast]], axis=2)
norm_right_ft_all = np.concatenate([norm_right_ft[:, :, :steps], norm_right_ft_slow[:, :, :steps_slow], norm_right_ft_fast[:, :, :steps_fast]], axis=2)
norm_right_contact_all = np.concatenate([norm_right_contact[:, :, :steps], norm_right_contact_slow[:, :, :steps_slow], norm_right_contact_fast[:, :, :steps_fast]], axis=2)
norm_right_grf_all = np.concatenate([norm_right_grf[:, :, :steps], norm_right_grf_slow[:, :, :steps_slow], norm_right_grf_fast[:, :, :steps_fast]], axis=2)
norm_right_grm_all = np.concatenate([norm_right_grm[:, :, :steps], norm_right_grm_slow[:, :, :steps_slow], norm_right_grm_fast[:, :, :steps_fast]], axis=2)
norm_right_pp_pos_all = np.concatenate([norm_right_pp_pos[:, :, :steps], norm_right_pp_pos_slow[:, :, :steps_slow], norm_right_pp_pos_fast[:, :, :steps_fast]], axis=2)
print('norm_right_insole_pp:', norm_right_insole_pp.shape)   #(100, 1, 100)
print(norm_right_cop_insole_all.shape)  #(100, 2, 100)
print(norm_right_grf_all.shape)         #(100, 3, 100)
print(norm_right_insole_pp_all.shape)   #(100, 1, 100)
print(norm_right_grf_all.shape)
#####
# num_cycles: how many steps (100)
# time_steps: how many time point each step (100)
#####
num_cycles = norm_right_cop_insole_all.shape[2]  # 100
time_steps = norm_right_cop_insole_all.shape[0]  # 100
print(num_cycles), print(time_steps)
# Input & Output Empty arrays
X = np.zeros((num_cycles, time_steps, 5))
Y = np.zeros((num_cycles, time_steps, 6))
Test = np.zeros((norm_right_grf_all.shape[2], norm_right_grf_all.shape[0], 6))

for i in range(num_cycles):
    insole_r = norm_right_insole_all[:, :, i]
    insole_pp = norm_right_insole_pp_all[:, :, i]  # shape: (100, 1)
    cop = norm_right_cop_insole_all[:, :, i]  # shape: (100, 2)
    fti = norm_right_fti_all[:, :, i]  # shape: (100, 1)
    ft = norm_right_ft_all[:, :, i]  # shape: (100, 1)
    cont = norm_right_contact_all[:, :, i]  # shape: (100, 1)
    pp_pos = norm_right_pp_pos_all[:, :, i]  # shape: (100, 2)
    # shape: (100, 3)
    grf = norm_right_grf_all[:, :, i]
    grm = norm_right_grm_all[:, :, i]
    # Input: insole_pp + cop
    #X[i] = np.concatenate((insole_r, cop, fti, cont, pp_pos), axis=1)
    X[i] = insole_r
    Y[i] = np.concatenate((grf, grm), axis=1)
# X: [100, 100, 8]
# Y: [100, 100, 3]
print(X)
def normalize_data_seq(data):
    min_val = np.min(data, axis=(0, 1), keepdims=True)
    max_val = np.max(data, axis=(0, 1), keepdims=True)
    normalized_data = 2 * (data - min_val) / (max_val - min_val) - 1
    return normalized_data, min_val, max_val
def normalize_data_seq_with_given_min_max(data, min, max):
    normalized_data = 2 * (data - min) / (max - min) - 1
    return normalized_data
def denormalize_data_seq(normalized_data, min_val, max_val):
    return (normalized_data + 1) / 2 * (max_val - min_val) + min_val

X_normalized, X_min, X_max = normalize_data_seq(X)  # X shape: [100, 100, 3]
Y_normalized, Y_min, Y_max = normalize_data_seq(Y)

"""
X_pool, X_final_test, y_pool, y_final_test = train_test_split(
    X_normalized, Y_normalized, test_size=0.2, random_state=42, shuffle=True
)
"""
X_pool = X_normalized
y_pool = Y_normalized

device = torch.device('cuda')

class MorletWavelet(nn.Module):
    def __init__(self):
        super(MorletWavelet, self).__init__()

    def forward(self, x):
        return torch.cos(1.6 * x) * torch.exp(-x ** 2 / 2)

class WaveletNN(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super(WaveletNN, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden

        self.omega = nn.Parameter(torch.randn(n_hidden, n_inputs))
        self.shift = nn.Parameter(torch.randn(n_hidden, n_inputs))
        self.dilation = nn.Parameter(torch.abs(torch.randn(n_hidden, n_inputs)) + 1e-6)

        self.output_layer = nn.Linear(n_hidden, n_outputs)

        self.wavelet = MorletWavelet()

    def forward(self, x):
        # x: [batch_size, time_steps, n_inputs]
        B, T, N = x.shape  # B=batch_size, T=time_steps, N=input_dim

        x_flat = x.reshape(-1, N)  # [B*T, N]       8000,3

        x_exp = x_flat.unsqueeze(1).expand(-1, self.n_hidden, -1)  # [B*T, H, N]    8000,H,3
        omega = self.omega.unsqueeze(0)                            # [1, H, N]      H,3
        shift = self.shift.unsqueeze(0)
        dilation = self.dilation.unsqueeze(0)

        wave_input = (x_exp * omega - shift) / dilation            # [B*T, H, N]    8000,H,3
        wave_output = self.wavelet(wave_input)                     # [B*T, H, N]
        H = torch.prod(wave_output, dim=2)                         # [B*T, H]

        out = self.output_layer(H)                                 # [B*T, n_outputs]
        out = out.view(B, T, -1)                                   # [B, T, n_outputs]
        return out

k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

n_inputs_model = X_pool.shape[2]
n_hidden_model = 41
n_outputs_model = y_pool.shape[2]
lr_model = 0.025
epochs_model = 2500

feature_names = ["GRF-X", "GRF-Y", "GRF-Z", "GRM-X", "GRM-Y", "GRM-Z"]
all_folds_metrics_by_feature = {name: {"r": [], "rmse": [], "nrmse": []} for name in feature_names}
fold_val_losses_final = []

def get_metrics(y_true, y_pred):
    y_true_flat = np.asarray(y_true).flatten()
    y_pred_flat = np.asarray(y_pred).flatten()

    r, _ = pearsonr(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    nrmse = rmse / (np.max(y_true_flat) - np.min(y_true_flat))
    return r, rmse, nrmse

for fold, (train_idx, val_idx) in enumerate(kf.split(X_pool)):
    print(f"\n--- Fold {fold+1}/{k_folds} ---")
    X_train_fold, y_train_fold = X_pool[train_idx], y_pool[train_idx]
    X_val_fold, y_val_fold = X_pool[val_idx], y_pool[val_idx]

    X_train_fold_t = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
    y_train_fold_t = torch.tensor(y_train_fold, dtype=torch.float32).to(device)
    X_val_fold_t = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
    y_val_fold_t = torch.tensor(y_val_fold, dtype=torch.float32).to(device)

    model = WaveletNN(n_inputs=n_inputs_model, n_hidden=n_hidden_model, n_outputs=n_outputs_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_model)
    loss_fn = nn.MSELoss()

    last_val_loss_this_fold = float('inf')
    for epoch in range(epochs_model):
        model.train()
        optimizer.zero_grad()
        y_pred_train = model(X_train_fold_t)
        loss_train = loss_fn(y_pred_train, y_train_fold_t)
        loss_train.backward()
        optimizer.step()
        if epoch % (epochs_model // 10) == 0 or epoch == epochs_model - 1 :
            model.eval()
            with torch.no_grad():
                y_pred_val_epoch = model(X_val_fold_t)
                loss_val_epoch = loss_fn(y_pred_val_epoch, y_val_fold_t)
                last_val_loss_this_fold = loss_val_epoch.item()
            print(f"Fold {fold + 1}, Epoch {epoch}, Train Loss: {loss_train.item():.6f}, Val Loss: {last_val_loss_this_fold:.6f}")

    fold_val_losses_final.append(last_val_loss_this_fold)

    model.eval()
    with torch.no_grad():
        y_pred_val_fold_norm = model(X_val_fold_t)

    y_pred_val_fold_denorm = denormalize_data_seq(y_pred_val_fold_norm.cpu().numpy(), Y_min, Y_max)
    y_val_fold_denorm = denormalize_data_seq(y_val_fold, Y_min, Y_max)

    print(f"Fold {fold+1} Metrics by Feature:")
    for feature_j in range(n_outputs_model):
        feature_name = feature_names[feature_j]
        r_vals_feature = []
        rmse_vals_feature = []
        nrmse_vals_feature = []
        for sample_i in range(y_pred_val_fold_denorm.shape[0]):
            true_seq = y_val_fold_denorm[sample_i, :, feature_j]
            pred_seq = y_pred_val_fold_denorm[sample_i, :, feature_j]
            r, rmse, nrmse = get_metrics(true_seq, pred_seq)
            r_vals_feature.append(r)
            rmse_vals_feature.append(rmse)
            nrmse_vals_feature.append(nrmse)

        avg_r_feature_fold = np.nanmean(r_vals_feature)
        avg_rmse_feature_fold = np.nanmean(rmse_vals_feature)
        avg_nrmse_feature_fold = np.nanmean(nrmse_vals_feature)

        all_folds_metrics_by_feature[feature_name]["r"].append(avg_r_feature_fold)
        all_folds_metrics_by_feature[feature_name]["rmse"].append(avg_rmse_feature_fold)
        all_folds_metrics_by_feature[feature_name]["nrmse"].append(avg_nrmse_feature_fold)
        print(f"  {feature_name}: Avg R={avg_r_feature_fold:.4f}, Avg RMSE={avg_rmse_feature_fold:.4f}, Avg NRMSE={avg_nrmse_feature_fold:.4f}")

print(f"\n--- K-Fold Cross-Validation Summary (Metrics Averaged Across Folds per Feature) ---")
mean_val_loss_cv = np.nanmean(fold_val_losses_final)
std_val_loss_cv = np.nanstd(fold_val_losses_final)
print(f"Mean Validation Loss across {k_folds} folds: {mean_val_loss_cv:.6f} (Std: {std_val_loss_cv:.6f})")

for feature_name in feature_names:
    avg_r = np.nanmean(all_folds_metrics_by_feature[feature_name]["r"])
    std_r = np.nanstd(all_folds_metrics_by_feature[feature_name]["r"])
    avg_rmse = np.nanmean(all_folds_metrics_by_feature[feature_name]["rmse"])
    std_rmse = np.nanstd(all_folds_metrics_by_feature[feature_name]["rmse"])
    avg_nrmse = np.nanmean(all_folds_metrics_by_feature[feature_name]["nrmse"])
    std_nrmse = np.nanstd(all_folds_metrics_by_feature[feature_name]["nrmse"])
    print(f"  {feature_name}:")
    print(f"    Avg R     : {avg_r:.4f} (Std: {std_r:.4f})")
    print(f"    Avg RMSE  : {avg_rmse:.4f} (Std: {std_rmse:.4f})")
    print(f"    Avg NRMSE : {avg_nrmse:.4f} (Std: {std_nrmse:.4f})")

"""
print("\n--- Training Final Model on Full Pool (X_pool, y_pool) ---")
X_pool_t = torch.tensor(X_pool, dtype=torch.float32).to(device)
y_pool_t = torch.tensor(y_pool, dtype=torch.float32).to(device)
X_final_test_t = torch.tensor(X_final_test, dtype=torch.float32).to(device)
y_final_test_t = torch.tensor(y_final_test, dtype=torch.float32).to(device)

final_model = WaveletNN(n_inputs=n_inputs_model, n_hidden=n_hidden_model, n_outputs=n_outputs_model).to(device)
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=lr_model)
final_loss_fn = nn.MSELoss()

for epoch in range(epochs_model):
    final_model.train()
    final_optimizer.zero_grad()
    y_pred_final_train = final_model(X_pool_t)
    loss_final_train = final_loss_fn(y_pred_final_train, y_pool_t)
    loss_final_train.backward()
    final_optimizer.step()
    if epoch % (epochs_model // 10) == 0 or epoch == epochs_model -1: #减少打印
        print(f"Final Model Training - Epoch {epoch}, Loss: {loss_final_train.item():.6f}")

final_model.eval()
with torch.no_grad():
    y_pred_on_final_test_norm = final_model(X_final_test_t)
    final_test_loss_val = final_loss_fn(y_pred_on_final_test_norm, y_final_test_t)
    print(f"\n--- Final Model Evaluation on Hold-out Test Set ---")
    print(f"Loss on X_final_test: {final_test_loss_val.item():.6f}")

y_pred_val_fold_denorm = denormalize_data_seq(y_pred_on_final_test_norm.cpu().numpy(), Y_min, Y_max)
y_val_fold_denorm = denormalize_data_seq(y_final_test, Y_min, Y_max)

print(f"\n  Metrics on Hold-out Test Set (Averaged Across Samples per Feature):")
for feature_j in range(n_outputs_model):
    feature_name = feature_names[feature_j]
    r_vals_feature_final = []
    rmse_vals_feature_final = []
    nrmse_vals_feature_final = []
    for sample_i in range(y_pred_val_fold_denorm.shape[0]):
        true_seq = y_val_fold_denorm[sample_i, :, feature_j]
        pred_seq = y_pred_val_fold_denorm[sample_i, :, feature_j]
        r, rmse, nrmse = get_metrics(true_seq, pred_seq)
        if not np.isnan(r): r_vals_feature_final.append(r)
        if not np.isnan(rmse): rmse_vals_feature_final.append(rmse)
        if not np.isnan(nrmse): nrmse_vals_feature_final.append(nrmse)

    avg_r_feature_final = np.nanmean(r_vals_feature_final) if r_vals_feature_final else np.nan
    avg_rmse_feature_final = np.nanmean(rmse_vals_feature_final) if rmse_vals_feature_final else np.nan
    avg_nrmse_feature_final = np.nanmean(nrmse_vals_feature_final) if nrmse_vals_feature_final else np.nan
    print(f"    {feature_name}: Avg R={avg_r_feature_final:.4f}, Avg RMSE={avg_rmse_feature_final:.4f}, Avg NRMSE={avg_nrmse_feature_final:.4f}")
"""

for sample_idx in range(50):
    grf_x_mean_plot = y_pred_val_fold_denorm[sample_idx, :, 0]
    grf_real_x_mean_plot = y_val_fold_denorm[sample_idx, :, 0]
    grf_y_mean_plot = y_pred_val_fold_denorm[sample_idx, :, 1]
    grf_real_y_mean_plot = y_val_fold_denorm[sample_idx, :, 1]
    grf_z_mean_plot = y_pred_val_fold_denorm[sample_idx, :, 2]
    grf_real_z_mean_plot = y_val_fold_denorm[sample_idx, :, 2]

    grm_x_mean_plot = y_pred_val_fold_denorm[sample_idx, :, 3] / 10000
    grm_real_x_mean_plot = y_val_fold_denorm[sample_idx, :, 3] / 10000
    grm_y_mean_plot = y_pred_val_fold_denorm[sample_idx, :, 4] / 10000
    grm_real_y_mean_plot = y_val_fold_denorm[sample_idx, :, 4] / 10000
    grm_z_mean_plot = y_pred_val_fold_denorm[sample_idx, :, 5] / 10000
    grm_real_z_mean_plot = y_val_fold_denorm[sample_idx, :, 5] / 10000

    x_axis_plot = np.arange(grf_real_x_mean_plot.shape[0])
    fig, axs = plt.subplots(2, 3, figsize=(21, 13), sharex=True)
    axs = axs.flatten()

    plot_details = [
        (grf_x_mean_plot, grf_real_x_mean_plot, y_val_fold_denorm[sample_idx, :, 0], feature_names[0]),
        (grf_y_mean_plot, grf_real_y_mean_plot, y_val_fold_denorm[sample_idx, :, 1], feature_names[1]),
        (grf_z_mean_plot, grf_real_z_mean_plot, y_val_fold_denorm[sample_idx, :, 2], feature_names[2]),
        (grm_x_mean_plot, grm_real_x_mean_plot, y_val_fold_denorm[sample_idx, :, 3] / 10000, feature_names[3]),
        (grm_y_mean_plot, grm_real_y_mean_plot, y_val_fold_denorm[sample_idx, :, 4] / 10000, feature_names[4]),
        (grm_z_mean_plot, grm_real_z_mean_plot, y_val_fold_denorm[sample_idx, :, 5] / 10000, feature_names[5]),
    ]

    for i, (pred_mean, real_mean, real_all_samples_feature, title) in enumerate(plot_details):
        ax = axs[i]
        ax.plot(x_axis_plot, pred_mean, label='Predicted WNN', color='red', linewidth=1.5)
        ax.plot(x_axis_plot, real_mean, label='Measured (Force plate)', color='blue', linewidth=1.5)

        ax.text(0.5, 0.94, title, fontsize=32, fontweight='bold',
                ha='center', va='top', transform=ax.transAxes)
        ax.set_xlabel("Stance phase (%)", fontsize=32)
        if i == 0:
            ax.set_ylabel("Ground Reaction Force (N)", fontsize=28, labelpad=10)
        if i == 3:
            ax.set_ylabel("Ground Reaction Moment ($10^4$ Nm)", fontsize=28, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=30)
        ax.grid(True)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=30)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.08, wspace=0.2, hspace=0.15)
    plt.show()