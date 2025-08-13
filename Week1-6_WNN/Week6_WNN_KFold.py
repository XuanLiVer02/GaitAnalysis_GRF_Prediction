from Week6_NewInterpolation import *
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

steps = 100
norm_right_cop_insole = norm_right_cop_insole[:, :, :steps]
norm_right_insole = norm_right_insole[:, :, :steps]
norm_right_insole_pp = norm_right_insole_pp[:, :, :steps]
norm_right_fti = norm_right_fti[:, :, :steps]
norm_right_ft = norm_right_ft[:, :, :steps]
norm_right_contact = norm_right_contact[:, :, :steps]
norm_right_pp_pos = norm_right_pp_pos[:, :, :steps]
print('norm_right_insole_pp:', norm_right_insole_pp.shape)   #(100, 1, 100)
print(norm_right_cop_insole.shape)  #(100, 2, 100)
print(norm_right_grf.shape)         #(100, 3, 100)
print(norm_right_insole_pp.shape)   #(100, 1, 100)
print(norm_right_grf.shape)
#####
# num_cycles: how many steps (100)
# time_steps: how many time point each step (100)
#####
num_cycles = norm_right_cop_insole.shape[2]  # 100
time_steps = norm_right_cop_insole.shape[0]  # 100
print(num_cycles), print(time_steps)
# Input & Output Empty arrays
X = np.zeros((num_cycles, time_steps, 5))
Y = np.zeros((num_cycles, time_steps, 6))

for i in range(num_cycles):
    insole_r = norm_right_insole[:, :, i]
    insole_pp = norm_right_insole_pp[:, :, i]   # shape: (100, 1)
    cop = norm_right_cop_insole[:, :, i]    # shape: (100, 2)
    fti = norm_right_fti[:, :, i]   # shape: (100, 1)
    #ft = norm_right_ft[:, :, i]   # shape: (100, 1)
    cont = norm_right_contact[:, :, i]   # shape: (100, 1)
    pp_pos = norm_right_pp_pos[:, :, i]   # shape: (100, 1)
    # shape: (100, 3)
    grf = norm_right_grf[:, :, i]
    grm = norm_right_grm[:, :, i]
    # Input: insole_pp + cop
    #X[i] = np.concatenate((insole_r, cop, cont, fti, pp_pos), axis=1)
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

def denormalize_data_seq(normalized_data, min_val, max_val):
    return (normalized_data + 1) / 2 * (max_val - min_val) + min_val

X_normalized, X_min, X_max = normalize_data_seq(X)  # X shape: [100, 100, 3]
Y_normalized, Y_min, Y_max = normalize_data_seq(Y)

X_pool, X_final_test, y_pool, y_final_test = train_test_split(
    X_normalized, Y_normalized, test_size=0.2, random_state=42, shuffle=True
)

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
n_hidden_model = 61
n_outputs_model = y_pool.shape[2]
lr_model = 0.02
epochs_model = 3000

# 例如: all_folds_metrics_by_feature['GRF-X']['r'] = [r_fold1, r_fold2, ...]
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

y_pred_final_denorm = denormalize_data_seq(y_pred_on_final_test_norm.cpu().numpy(), Y_min, Y_max)
y_final_test_denorm = denormalize_data_seq(y_final_test, Y_min, Y_max)

print(f"\n  Metrics on Hold-out Test Set (Averaged Across Samples per Feature):")
for feature_j in range(n_outputs_model):
    feature_name = feature_names[feature_j]
    r_vals_feature_final = []
    rmse_vals_feature_final = []
    nrmse_vals_feature_final = []
    for sample_i in range(y_pred_final_denorm.shape[0]):
        true_seq = y_final_test_denorm[sample_i, :, feature_j]
        pred_seq = y_pred_final_denorm[sample_i, :, feature_j]
        r, rmse, nrmse = get_metrics(true_seq, pred_seq)
        if not np.isnan(r): r_vals_feature_final.append(r)
        if not np.isnan(rmse): rmse_vals_feature_final.append(rmse)
        if not np.isnan(nrmse): nrmse_vals_feature_final.append(nrmse)

    avg_r_feature_final = np.nanmean(r_vals_feature_final) if r_vals_feature_final else np.nan
    avg_rmse_feature_final = np.nanmean(rmse_vals_feature_final) if rmse_vals_feature_final else np.nan
    avg_nrmse_feature_final = np.nanmean(nrmse_vals_feature_final) if nrmse_vals_feature_final else np.nan
    print(f"    {feature_name}: Avg R={avg_r_feature_final:.4f}, Avg RMSE={avg_rmse_feature_final:.4f}, Avg NRMSE={avg_nrmse_feature_final:.4f}")


if y_pred_final_denorm.shape[0] > 0 :
    grf_x_mean_plot = y_pred_final_denorm[0, :, 0]
    grf_x_real = y_final_test_denorm[0,:,0]
    grf_real_x_mean_plot = y_final_test_denorm[:,:,0].mean(axis=0)
    grf_y_mean_plot = y_pred_final_denorm[0, :, 1]
    grf_y_real = y_final_test_denorm[0,:,1]
    grf_real_y_mean_plot = y_final_test_denorm[:,:,1].mean(axis=0)
    grf_z_mean_plot = y_pred_final_denorm[0, :, 2]
    grf_z_real = y_final_test_denorm[0,:,2]
    grf_real_z_mean_plot = y_final_test_denorm[:,:,2].mean(axis=0)

    grm_x_mean_plot = y_pred_final_denorm[0, :, 3]
    grm_x_real = y_final_test_denorm[0,:,3]
    grm_real_x_mean_plot = y_final_test_denorm[:,:,3].mean(axis=0)
    grm_y_mean_plot = y_pred_final_denorm[0, :, 4]
    grm_y_real = y_final_test_denorm[0,:,4]
    grm_real_y_mean_plot = y_final_test_denorm[:,:,4].mean(axis=0)
    grm_z_mean_plot = y_pred_final_denorm[0, :, 5]
    grm_z_real = y_final_test_denorm[0,:,5]
    grm_real_z_mean_plot = y_final_test_denorm[:,:,5].mean(axis=0)

    x_axis_plot = np.arange(grf_real_x_mean_plot.shape[0])
    plt.figure(figsize=(12, 10))
    plt.suptitle('Final Model: Predicted/Real GRF/GRM (Averaged over Hold-out Test Set)', fontsize=16)

    plot_details = [
        (grf_x_mean_plot, grf_real_x_mean_plot, grf_x_real, y_final_test_denorm[:, :, 0], feature_names[0]),
        (grm_x_mean_plot, grm_real_x_mean_plot, grm_x_real, y_final_test_denorm[:, :, 3], feature_names[3]),
        (grf_y_mean_plot, grf_real_y_mean_plot, grf_y_real ,y_final_test_denorm[:, :, 1], feature_names[1]),
        (grm_y_mean_plot, grm_real_y_mean_plot, grm_y_real, y_final_test_denorm[:, :, 4], feature_names[4]),
        (grf_z_mean_plot, grf_real_z_mean_plot, grf_z_real, y_final_test_denorm[:, :, 2], feature_names[2]),
        (grm_z_mean_plot, grm_real_z_mean_plot, grm_z_real, y_final_test_denorm[:, :, 5], feature_names[5]),
    ]

    for i, (pred_mean, real_mean, real_sample, real_all_samples_feature, title) in enumerate(plot_details):
        plt.subplot(3, 2, i + 1)
        plt.plot(x_axis_plot, pred_mean, label=f'Predicted {title}', color='red', linewidth=1.5)
        plt.plot(x_axis_plot, real_mean, label=f'Real {title}', color='blue', linewidth=1.5)
        plt.plot(x_axis_plot, real_sample, color='green', linestyle='--', label=f'Real {title} Mean')
        plt.fill_between(x_axis_plot, real_mean + np.std(real_all_samples_feature, axis=0),
                         real_mean - np.std(real_all_samples_feature, axis=0), color='blue', alpha=0.2, label=f'Real {title} StdDev')
        plt.title(f"Average Predicted {title}")
        plt.xlabel("Time Step")
        plt.ylabel(title)
        plt.grid(True)
        plt.legend(fontsize='small')
    plt.show()