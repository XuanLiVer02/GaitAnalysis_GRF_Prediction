from model import VisionTransformer
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from Gait.Week12_UNB.load_data_batch import X_all, Y_all

def calculate_stats_and_standardize(data):
    mean_val = np.mean(data, axis=(0, 1), keepdims=True)
    std_val = np.std(data, axis=(0, 1), keepdims=True)
    standardized_data = (data - mean_val) / (std_val + 1e-8)
    return standardized_data, mean_val, std_val

def apply_standardization(data, mean_val, std_val):
    return (data - mean_val) / (std_val + 1e-8)

def destandardize_data(standardized_data, mean_val, std_val):
    return standardized_data * (std_val + 1e-8) + mean_val

X_normalized, X_mean, X_std = calculate_stats_and_standardize(X_all)
Y_normalized, Y_mean, Y_std = calculate_stats_and_standardize(Y_all)
X_pool = X_normalized
y_pool = Y_normalized
X_train, X_test, y_train, y_test = train_test_split(X_pool, y_pool, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
print("X_pool shape:", X_pool.shape)
print("y_pool shape:", y_pool.shape)

train_loss_list = []
val_loss_list = []

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    best_val_loss = float('inf')
    counter = 0
    patience = 20
    min_delta = 1e-4
    model_path = 'early_stop_ViT.pth'

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

        scheduler.step(avg_val_loss)

        if epoch % (epochs // 20) == 0 or epoch == epochs - 1:
            print(f"Final Model Training - Epoch {epoch}, Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping logic
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best val loss: {best_val_loss:.6f}")
                break
    # Restore best model
    model.load_state_dict(torch.load(model_path))
    return train_loss_list, val_loss_list

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

lr_model = 1e-4
epochs_model = 200

print("\n--- Training Final Model on Full Pool (X_pool, y_pool) ---")

batch_size = 50
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

final_model = VisionTransformer(output_features=1, num_layers=6).to(device)
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=lr_model, weight_decay=1e-5)
final_criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, mode='min', factor=0.5, patience=15, verbose=True)

train(final_model, train_loader, val_loader, final_criterion, final_optimizer, scheduler, epochs=epochs_model)
torch.save(final_model.state_dict(), 'ViT.pth')

plt.figure(figsize=(12, 6))
plt.plot(range(1, len(train_loss_list)+1), train_loss_list, label='Train Loss', color='blue')
plt.plot(range(1, len(train_loss_list)+1), val_loss_list, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig("training_result.png")

test_loss = 0.0
final_model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred_batch = final_model(X_batch)
        loss = final_criterion(y_pred_batch, y_batch)
        test_loss += loss.item()
        all_preds.append(y_pred_batch.cpu())
        all_targets.append(y_batch.cpu())

    final_test_loss_val = test_loss / len(test_loader)

    print(f"\n--- Final Model Evaluation on Hold-out Test Set ---")
    print(f"Loss on X_final_test: {final_test_loss_val:.6f}")

    y_pred_on_final_test_norm = torch.cat(all_preds, dim=0).numpy()
    y_final_test = torch.cat(all_targets, dim=0).numpy()

    y_pred_final_denorm = destandardize_data(y_pred_on_final_test_norm, Y_mean, Y_std)
    y_final_test_denorm = destandardize_data(y_final_test, Y_mean, Y_std)

print(f"\n  Metrics on Hold-out Test Set (Averaged Across Samples per Feature):")
for feature_j in range(y_pool.shape[2]):
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

grf_x_mean_plot = y_pred_final_denorm[1, :, 0]
grf_x_real = y_final_test_denorm[1, :, 0]
grf_real_x_mean_plot = y_final_test_denorm[:, :, 0].mean(axis=0)

x_axis_plot = np.arange(grf_real_x_mean_plot.shape[0])
plt.figure(figsize=(12, 10))
plt.suptitle('Final Model: Predicted/Real GRF/GRM (Averaged over Hold-out Test Set)', fontsize=16)
plot_details = [
    (grf_x_mean_plot, grf_real_x_mean_plot, grf_x_real, y_final_test_denorm[:, :, 0], feature_names[0])
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
# plt.show()
plt.savefig("GRF_output.png")
plt.close('all')