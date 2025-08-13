##########################################################
# Just a replication of Week3_WNN, using different data from Week4
# Add new features: foot time integral, foot trace, contact areea
##########################################################
#from Gait_Week1 import *
from Week5_NewInterpolation import *
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

steps = 100
norm_right_cop_insole = norm_right_cop_insole[:, :, :steps]
norm_right_insole = norm_right_insole[:, :, :steps]
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
X = np.zeros((num_cycles, time_steps, 5))
Y = np.zeros((num_cycles, time_steps, 6))
Test = np.zeros((norm_right_grf.shape[2], norm_right_grf.shape[0], 6))

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
    # Input: insole_pp + cop
    #X[i] = np.concatenate((insole_r, insole_pp, cop), axis=1)
    X[i] = insole_r
    Y[i] = np.concatenate((grf, grm), axis=1)
# X: [100, 100, 8]
# Y: [100, 100, 3]
print(X)
for i in range(norm_right_grf.shape[2]):
    Test[i] = np.concatenate((norm_right_grf[:,:,i], norm_right_grm[:,:,i]), axis=1)

def normalize_data_seq(data):
    min_val = np.min(data, axis=(0, 1), keepdims=True)
    max_val = np.max(data, axis=(0, 1), keepdims=True)
    normalized_data = 2 * (data - min_val) / (max_val - min_val) - 1
    return normalized_data, min_val, max_val

def denormalize_data_seq(normalized_data, min_val, max_val):
    return (normalized_data + 1) / 2 * (max_val - min_val) + min_val

X_normalized, X_min, X_max = normalize_data_seq(X)  # X shape: [100, 100, 3]
Y_normalized, Y_min, Y_max = normalize_data_seq(Y)
Test_normalized, Test_min, Test_max = normalize_data_seq(Test)
print(X_min)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, Y_normalized, test_size=0.2, random_state=42, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, shuffle=True)

device = torch.device('cuda')
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

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

# Model parameters
n_inputs = X_train.shape[2]  # Number of features in insole_cop 3
n_hidden = 41  # Number of hidden neurons (adjust as needed)
n_outputs = y_train.shape[2]  # Number of features in force_plate_grf 3
print("X_train.shape", X_train.shape)
print("y_train.shape", y_train.shape)
model = WaveletNN(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

train_loss_list = []
val_loss_list = []
for epoch in range(3000):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)  # X_train: shape [batch_size, 1024]
    loss = loss_fn(y_pred, y_train)  # y_train: shape [batch_size, 3]
    train_loss_list.append(loss.item())
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Training Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_val), y_val)
        val_loss_list.append(val_loss.item())

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

# y_pred: [20, 100, 3]
y_pred = y_pred.cpu().detach().numpy()
y_test = y_test.cpu().detach().numpy()

norm_grf_x_mean = y_pred[:, :, 0].mean(axis=0)
norm_grf_real_x_mean = y_test[:,:,0].mean(axis=0)
norm_grf_y_mean = y_pred[:, :, 1].mean(axis=0)
norm_grf_real_y_mean = y_test[:,:,1].mean(axis=0)
norm_grf_z_mean = y_pred[:, :, 2].mean(axis=0)
norm_grf_real_z_mean = y_test[:,:,2].mean(axis=0)

norm_grm_x_mean = y_pred[:, :, 3].mean(axis=0)
norm_grm_real_x_mean = y_test[:,:,3].mean(axis=0)
norm_grm_y_mean = y_pred[:, :, 4].mean(axis=0)
norm_grm_real_y_mean = y_test[:,:,4].mean(axis=0)
norm_grm_z_mean = y_pred[:, :, 5].mean(axis=0)
norm_grm_real_z_mean = y_test[:,:,5].mean(axis=0)

y_pred_denorm = denormalize_data_seq(y_pred, Y_min, Y_max)   # shape: [20, 100, 3]
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