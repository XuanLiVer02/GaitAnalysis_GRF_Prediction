from Week6_NewInterpolation import *
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset

steps = 100
norm_right_cop_insole = norm_right_cop_insole[:, :, :steps]
#norm_right_insole = norm_right_insole[:, :, :steps]
norm_right_insole = norm_right_insole_all[:, :, :steps]
norm_right_insole_pp = norm_right_insole_pp[:, :, :steps]
norm_right_fti = norm_right_fti[:, :, :steps]
norm_right_ft = norm_right_ft[:, :, :steps]
norm_right_contact = norm_right_contact[:, :, :steps]
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
X = np.zeros((num_cycles, time_steps, 1024))
Y = np.zeros((num_cycles, time_steps, 6))

for i in range(num_cycles):
    insole_r = norm_right_insole[:, :, i]
    insole_pp = norm_right_insole_pp[:, :, i]   # shape: (100, 1)
    cop = norm_right_cop_insole[:, :, i]    # shape: (100, 2)
    fti = norm_right_fti[:, :, i]   # shape: (100, 1)
    #ft = norm_right_ft[:, :, i]   # shape: (100, 1)
    cont = norm_right_contact[:, :, i]   # shape: (100, 1)
    # shape: (100, 3)
    grf = norm_right_grf[:, :, i]
    grm = norm_right_grm[:, :, i]
    #X[i] = np.concatenate((insole_r, insole_pp, cop), axis=1)
    X[i] = insole_r
    Y[i] = np.concatenate((grf, grm), axis=1)
# X: [100, 100, 5]      # num_cycles, num_points, num_features
# Y: [100, 100, 3]

def normalize_data_seq(data):
    min_val = np.min(data, axis=(0, 2), keepdims=True)
    max_val = np.max(data, axis=(0, 2), keepdims=True)
    normalized_data = 2 * (data - min_val) / (max_val - min_val + 1e-8) - 1
    return normalized_data, min_val, max_val
def normalize_data_seq_with_given_min_max(data, min, max):
    normalized_data = 2 * (data - min) / (max - min + 1e-8) - 1
    normalized_data = np.clip(normalized_data, -1, 1)
    return normalized_data
def denormalize_data_seq(normalized_data, min_val, max_val):
    return (normalized_data + 1) / 2 * (max_val - min_val + 1e-8) + min_val

X_transpose = np.transpose(X, (0, 2, 1))  # shape: (num_cycles, num_features, num_points)
Y_transpose = np.transpose(Y, (0, 2, 1))  # shape: (num_cycles, 6, num_points)  20,6,100

X_train, X_test, y_train, y_test = train_test_split(X_transpose, Y_transpose, test_size=0.2, random_state=42, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, shuffle=True)

X_train, X_min, X_max = normalize_data_seq(X_train)     # X_min, X_max shape: (1, 1024, 1)  X_train.shape 60,1024,100
X_val = normalize_data_seq_with_given_min_max(X_val, X_min, X_max)      # (20, 1024, 100)
X_test = normalize_data_seq_with_given_min_max(X_test, X_min, X_max)
y_train, Y_min, Y_max = normalize_data_seq(y_train)
y_val = normalize_data_seq_with_given_min_max(y_val, Y_min, Y_max)
y_test = normalize_data_seq_with_given_min_max(y_test, Y_min, Y_max)
Y_min = np.transpose(Y_min, (0, 2, 1))
Y_max = np.transpose(Y_max, (0, 2, 1))  #1,1,6

device = torch.device('cuda')
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# DataLoader
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

class Gait1DCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Gait1DCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, out_channels, kernel_size=5, padding=2),
        )
    def forward(self, x):
        return self.net(x)

num_features = X_transpose.shape[1]
num_outputs = y_train.shape[1]
model = Gait1DCNN(in_channels=num_features, out_channels=num_outputs)
model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
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
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
    return total_loss / len(loader)

train(model, train_loader, val_loader, criterion, optimizer, epochs=500)
test_loss = evaluate(model, test_loader, criterion)
print(f"Test Loss: {test_loss:.4f}")

plt.figure()
plt.plot(train_loss_list, color = 'red', label='Training Loss')
plt.plot(val_loss_list, color = 'blue', label='Validation Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()

model.eval()
with torch.no_grad():
    y_pred = model(X_test)

y_pred = y_pred.cpu().detach().numpy()
y_test = y_test.cpu().detach().numpy()
y_pred = np.transpose(y_pred, (0, 2, 1))
y_test = np.transpose(y_test, (0, 2, 1))
print('y_pred shape:', y_pred.shape)  # [20, 100, 6]

y_pred_denorm = denormalize_data_seq(y_pred, Y_min, Y_max)   # shape: [20, 100, 6], [1,1,6]
y_test_denorm = denormalize_data_seq(y_test, Y_min, Y_max)

grf_x_mean = y_pred_denorm[:, :, 0].mean(axis=0)
grf_real_x_mean = y_test_denorm[:,:,0].mean(axis=0)
grf_y_mean = y_pred_denorm[:, :, 1].mean(axis=0)
grf_real_y_mean = y_test_denorm[:,:,1].mean(axis=0)
grf_z_mean = y_pred_denorm[:, :, 2].mean(axis=0)
grf_real_z_mean = y_test_denorm[:,:,2].mean(axis=0)

grm_x_mean = y_pred_denorm[:, :, 3].mean(axis=0)
grm_real_x_mean = y_test_denorm[:,:,3].mean(axis=0)
grm_y_mean = y_pred_denorm[:, :, 4].mean(axis=0)
grm_real_y_mean = y_test_denorm[:,:,4].mean(axis=0)
grm_z_mean = y_pred_denorm[:, :, 5].mean(axis=0)
grm_real_z_mean = y_test_denorm[:,:,5].mean(axis=0)

x_axis = np.arange(grf_real_x_mean.shape[0])  # shape: [100]

plt.figure(figsize=(12, 12))
plt.suptitle('Predicted/Real GRF/GRM over 100 Time Steps', fontsize=16)
plt.subplot(3, 2, 1)
plt.plot(grf_x_mean, label = 'Predicted GRF-X', color='red')
plt.plot(grf_real_x_mean, label = 'Real GRF-X', color='blue')
plt.fill_between(x_axis, grf_real_x_mean+np.std(y_test_denorm[:,:,0],axis=0),
         grf_real_x_mean-np.std(y_test_denorm[:,:,0],axis=0), color='blue', alpha=0.2)
plt.title("Average Predicted GRF-X over 100 Time Steps")
plt.xlabel("Time Step")
plt.ylabel("GRF-X")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(grf_y_mean, label = 'Predicted GRF-Y', color='red')
plt.plot(grf_real_y_mean, label = 'Real GRF-Y', color='blue')
plt.fill_between(x_axis, grf_real_y_mean+np.std(y_test_denorm[:,:,1],axis=0),
         grf_real_y_mean-np.std(y_test_denorm[:,:,1],axis=0), color='blue', alpha=0.2)
plt.title("Average Predicted GRF-Y over 100 Time Steps")
plt.xlabel("Time Step")
plt.ylabel("GRF-Y")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 5)
plt.plot(grf_z_mean, label = 'Predicted GRF-Z', color='red')
plt.plot(grf_real_z_mean, label = 'Real GRF-Z', color='blue')
plt.fill_between(x_axis, grf_real_z_mean+np.std(y_test_denorm[:,:,2],axis=0),
         grf_real_z_mean-np.std(y_test_denorm[:,:,2],axis=0), color='blue', alpha=0.2)
plt.title("Average Predicted GRF-Z over 100 Time Steps")
plt.xlabel("Time Step")
plt.ylabel("GRF-Z")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(grm_x_mean, label = 'Predicted GRM-X', color='red')
plt.plot(grm_real_x_mean, label = 'Real GRM-X', color='blue')
plt.fill_between(x_axis, grm_real_x_mean+np.std(y_test_denorm[:,:,3],axis=0),
         grm_real_x_mean-np.std(y_test_denorm[:,:,3],axis=0), color='blue', alpha=0.2)
plt.title("Average Predicted GRM-X over 100 Time Steps")
plt.xlabel("Time Step")
plt.ylabel("GRM-X")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(grm_y_mean, label = 'Predicted GRM-Y', color='red')
plt.plot(grm_real_y_mean, label = 'Real GRM-Y', color='blue')
plt.fill_between(x_axis, grm_real_y_mean+np.std(y_test_denorm[:,:,4],axis=0),
         grm_real_y_mean-np.std(y_test_denorm[:,:,4],axis=0), color='blue', alpha=0.2)
plt.title("Average Predicted GRM-Y over 100 Time Steps")
plt.xlabel("Time Step")
plt.ylabel("GRM-Y")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(grm_z_mean, label = 'Predicted GRM-Z', color='red')
plt.plot(grm_real_z_mean, label = 'Real GRM-Z', color='blue')
plt.fill_between(x_axis, grm_real_z_mean+np.std(y_test_denorm[:,:,5],axis=0),
         grm_real_z_mean-np.std(y_test_denorm[:,:,5],axis=0), color='blue', alpha=0.2)
plt.title("Average Predicted GRM-Z over 100 Time Steps")
plt.xlabel("Time Step")
plt.ylabel("GRM-Z")
plt.grid(True)
plt.legend()

def get_metrics(y_true, y_pred):
    r = pearsonr(y_true, y_pred)[0]
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    return r, rmse, nrmse

metrics_x = get_metrics(grf_real_x_mean, grf_x_mean)
metrics_y = get_metrics(grf_real_y_mean, grf_y_mean)
metrics_z = get_metrics(grf_real_z_mean, grf_z_mean)

metrics_mx = get_metrics(grm_real_x_mean, grm_x_mean)
metrics_my = get_metrics(grm_real_y_mean, grm_y_mean)
metrics_mz = get_metrics(grm_real_z_mean, grm_z_mean)

print("GRF x-axis: Correlation Coefficient = %.4f, RMSE = %.4f, NRMSE = %.4f" % metrics_x)
print("GRF y-axis: Correlation Coefficient = %.4f, RMSE = %.4f, NRMSE = %.4f" % metrics_y)
print("GRF z-axis: Correlation Coefficient = %.4f, RMSE = %.4f, NRMSE = %.4f" % metrics_z)
print("GRM x-axis: Correlation Coefficient = %.4f, RMSE = %.4f, NRMSE = %.4f" % metrics_mx)
print("GRM y-axis: Correlation Coefficient = %.4f, RMSE = %.4f, NRMSE = %.4f" % metrics_my)
print("GRM z-axis: Correlation Coefficient = %.4f, RMSE = %.4f, NRMSE = %.4f" % metrics_mz)

plt.show()