from model import VisionTransformer
from transfer_model import transfer_ViT
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, TensorDataset
from Gait.Week8_Interpolation import *

steps = 100
norm_right_cop_insole = norm_right_cop_insole[:, :, :steps]
norm_right_insole = norm_right_insole_all[:, :, :, :steps]

#####
# num_cycles: how many steps (100)
# time_steps: how many time point each step (100)
#####
num_cycles = norm_right_cop_insole.shape[2]  # 100
time_steps = norm_right_cop_insole.shape[0]  # 100

# Input & Output Empty arrays
X = np.zeros((num_cycles, time_steps, 64, 16))
Y = np.zeros((num_cycles, time_steps, 6))

for i in range(num_cycles):
    insole_r = norm_right_insole[:, :, :, i]
    cop = norm_right_cop_insole[:, :, i]    # shape: (100, 2)
    grf = norm_right_grf[:, :, i]
    grm = norm_right_grm[:, :, i]
    #X[i] = np.concatenate((insole_r, insole_pp, cop), axis=1)
    X[i] = insole_r
    Y[i] = np.concatenate((grf, grm), axis=1)
# X: [100, 100, 64, 16]
print(X.shape)
def calculate_stats_and_standardize(data):
    mean_val = np.mean(data, axis=(0, 1), keepdims=True)
    std_val = np.std(data, axis=(0, 1), keepdims=True)
    standardized_data = (data - mean_val) / (std_val + 1e-8)    # avoid zero value
    return standardized_data, mean_val, std_val

def apply_standardization(data, mean_val, std_val):
    return (data - mean_val) / (std_val + 1e-8)

def destandardize_data(standardized_data, mean_val, std_val):
    return standardized_data * (std_val + 1e-8) + mean_val

X_normalized, X_mean, X_std = calculate_stats_and_standardize(X)
Y_normalized, Y_mean, Y_std = calculate_stats_and_standardize(Y)

X_pool = X_normalized
y_pool = Y_normalized
print("X_pool shape:", X_pool.shape)
print("y_pool shape:", y_pool.shape)
device = torch.device('cuda')

train_loss_list = []
val_loss_list = []

def train(model, train_loader, val_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = evaluate(model, val_loader, criterion)
        train_loss_list.append(avg_train_loss)
        val_loss_list.append(avg_val_loss)
        if epoch % (epochs // 20) == 0 or epoch == epochs - 1:
            print(f"Final Model Training - Epoch {epoch}, Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
    return total_loss / len(loader)

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

k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
epochs_model = 200

for fold, (train_idx, val_idx) in enumerate(kf.split(X_pool)):
    print(f"\n--- Fold {fold+1}/{k_folds} ---")
    X_train_fold, y_train_fold = X_pool[train_idx], y_pool[train_idx]
    X_val_fold, y_val_fold = X_pool[val_idx], y_pool[val_idx]

    X_train_fold_t = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
    y_train_fold_t = torch.tensor(y_train_fold, dtype=torch.float32).to(device)
    X_val_fold_t = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
    y_val_fold_t = torch.tensor(y_val_fold, dtype=torch.float32).to(device)

    pre_model = VisionTransformer(output_features=1, num_layers=6, time_steps=101).to(device)
    pre_model.load_state_dict(torch.load('early_stop_ViT.pth', map_location=device))
    model = transfer_ViT(output_features=6).to(device)
    model.patch_embedding.load_state_dict(pre_model.patch_embedding.state_dict())
    model.positional_encoding.load_state_dict(pre_model.positional_encoding.state_dict())
    model.transformer_encoder.load_state_dict(pre_model.transformer_encoder.state_dict())
    for name, param in model.named_parameters():
        if any(nd in name for nd in ['patch_embedding', 'positional_encoding', 'transformer_encoder']):
            param.requires_grad = False

    param_groups = [
        {"params": model.mlp.parameters(), "lr": 1e-3},
        {"params": model.fc_out.parameters(), "lr": 1e-3}
    ]

    optimizer = torch.optim.Adam(param_groups, weight_decay=1e-5)
    criterion = nn.MSELoss()
    loss_fn = nn.MSELoss()

    last_val_loss_this_fold = float('inf')
    train_loader = DataLoader(TensorDataset(X_train_fold_t, y_train_fold_t), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_fold_t, y_val_fold_t), batch_size=64)
    train(model, train_loader, val_loader, criterion, optimizer, epochs=epochs_model)
    fold_val_losses_final.append(last_val_loss_this_fold)

    model.eval()
    with torch.no_grad():
        y_pred_val_fold_norm = model(X_val_fold_t)
    y_pred_val_fold_norm = y_pred_val_fold_norm.cpu().numpy()
    y_val_fold = y_val_fold_t.cpu().numpy()
    y_pred_val_fold_denorm = destandardize_data(y_pred_val_fold_norm, Y_mean, Y_std)
    y_val_fold_denorm = destandardize_data(y_val_fold, Y_mean, Y_std)

    print(f"Fold {fold+1} Metrics by Feature:")
    for feature_j in range(y_pool.shape[2]):
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

final_train_dataset = TensorDataset(X_pool_t, y_pool_t)
final_train_loader = DataLoader(final_train_dataset, batch_size=50, shuffle=True)  # 使用相同的 batch_size

final_model = transfer_ViT(output_features=6).to(device)
final_model.load_state_dict(torch.load('early_stop_ViT.pth', map_location=device))
final_optimizer = torch.optim.Adam(final_model.parameters(param_groups))
final_loss_fn = nn.MSELoss()

for epoch in range(epochs_model):
    final_model.train()
    total_epoch_loss = 0
    for X_batch, y_batch in final_train_loader:
        final_optimizer.zero_grad()
        y_pred_batch = final_model(X_batch)
        loss = final_loss_fn(y_pred_batch, y_batch)
        loss.backward()
        final_optimizer.step()
        total_epoch_loss += loss.item()

    avg_epoch_loss = total_epoch_loss / len(final_train_loader)
    if epoch % (epochs_model // 10) == 0 or epoch == epochs_model - 1:
        print(f"Final Model Training - Epoch {epoch}, Average Loss: {avg_epoch_loss:.6f}")

if y_pred_val_fold_norm.shape[0] > 0 :
    grf_x_mean_plot = y_pred_val_fold_denorm[1, :, 0]
    grf_x_real = y_val_fold_denorm[1, :, 0]
    grf_real_x_mean_plot = y_val_fold_denorm[:, :, 0].mean(axis=0)

    x_axis_plot = np.arange(grf_real_x_mean_plot.shape[0])
    plt.figure(figsize=(12, 10))
    plt.suptitle('Final Model: Predicted/Real GRF/GRM (Averaged over Hold-out Test Set)', fontsize=16)

    plot_details = [
        (grf_x_mean_plot, grf_real_x_mean_plot, grf_x_real, y_val_fold_denorm[:, :, 0], feature_names[0])
    ]

    for i, (pred_mean, real_mean, real_sample, real_all_samples_feature, title) in enumerate(plot_details):
        plt.subplot(3, 2, i + 1)
        plt.plot(x_axis_plot, pred_mean, label=f'Predicted {title}', color='red', linewidth=1.5)
        plt.plot(x_axis_plot, real_mean, label=f'Real {title}', color='blue', linewidth=1.5)
        plt.plot(x_axis_plot, real_sample, color='green', linestyle='--', label=f'Real {title} Mean')
        plt.fill_between(x_axis_plot, real_mean + np.std(real_all_samples_feature, axis=0),
                         real_mean - np.std(real_all_samples_feature, axis=0), color='blue', alpha=0.2,
                         label=f'Real {title} StdDev')
        plt.title(f"Average Predicted {title}")
        plt.xlabel("Time Step")
        plt.ylabel(title)
        plt.grid(True)
        plt.legend(fontsize='small')
    plt.savefig('transfer_results.png')