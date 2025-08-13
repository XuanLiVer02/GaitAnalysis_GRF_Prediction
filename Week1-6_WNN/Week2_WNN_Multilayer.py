from Gait_Week1 import *
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

norm_right_insole, avg_right_insole, std_right_insole = process_gait_cycles(
    gait_insole.insole_r, gait_insole.strike_r, gait_insole.off_r)
norm_right_insole_pp, avg_right_insole_pp, std_right_insole_pp = process_gait_cycles(
    gait_insole.pp_r.reshape(-1, 1), gait_insole.strike_r, gait_insole.off_r)

norm_right_cop_insole = norm_right_cop_insole[:, :, :100]
norm_right_insole = norm_right_insole[:, :, :100]
norm_right_insole_pp = norm_right_insole_pp[:, :, :100]
print(norm_right_cop_insole.shape)
print(norm_right_grf.shape)

num_cycles = norm_right_insole.shape[2]  # 100
time_steps = norm_right_insole.shape[0]  # 100

X = np.zeros((num_cycles, time_steps, 3))
Y = np.zeros((num_cycles, time_steps, 3))

for i in range(num_cycles):
    # shape: (100, 1024)
    insole_r = norm_right_insole[:, :, i]
    # shape: (100, 1)
    insole_pp = norm_right_insole_pp[:, :, i]
    # shape: (100, 2)
    cop = norm_right_cop_insole[:, :, i]
    # shape: (100, 3)
    grf = norm_right_grf[:, :, i]

    # Concatenate：2 + 1 = 3
    X[i] = np.concatenate((insole_pp, cop), axis=1)
    Y[i] = grf

def normalize_data_seq(data):
    min_val = np.min(data, axis=(0, 1), keepdims=True)
    max_val = np.max(data, axis=(0, 1), keepdims=True)
    normalized_data = 2 * (data - min_val) / (max_val - min_val) - 1
    return normalized_data, min_val, max_val

def denormalize_data_seq(normalized_data, min_val, max_val):
    return (normalized_data + 1) / 2 * (max_val - min_val) + min_val

X_normalized, X_min, X_max = normalize_data_seq(X)  # X shape: [100, 100, 3]
Y_normalized, Y_min, Y_max = normalize_data_seq(Y)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, Y_normalized, test_size=0.2, random_state=42, shuffle=True)

device = torch.device('cuda')
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
print(X_train[1:100,0])
# Define the Morlet wavelet function
def morlet_wavelet(x):
    return np.cos(5 * x) * np.exp(-x**2 / 2)

class MorletWavelet(nn.Module):
    def __init__(self):
        super(MorletWavelet, self).__init__()

    def forward(self, x):
        return torch.cos(1.6 * x) * torch.exp(-x ** 2 / 2)

class WaveletLayer(nn.Module):
    def __init__(self, n_inputs, n_hidden):
        super(WaveletLayer, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden

        self.omega = nn.Parameter(torch.randn(n_hidden, n_inputs))
        self.shift = nn.Parameter(torch.randn(n_hidden, n_inputs))
        self.dilation = nn.Parameter(torch.abs(torch.randn(n_hidden, n_inputs)) + 1e-6)

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

        out = H.view(B, T, self.n_hidden)                                   # [B, T, n_outputs]
        return out

class WaveletNN(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super(WaveletNN, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        layers = []
        for hidden_dim in n_hidden:
            layers.append(WaveletLayer(n_inputs, hidden_dim))
            n_inputs = hidden_dim
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(n_hidden[-1], n_outputs)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
# Model parameters
n_inputs = X_train.shape[2]  # Number of features in insole_cop 3
n_hidden = [7,7]  # Number of hidden neurons (adjust as needed)
n_outputs = y_train.shape[2]  # Number of features in force_plate_grf 3
print("X_train.shape", X_train.shape)
print("y_train.shape", y_train.shape)
model = WaveletNN(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
train_loss_list = []

for epoch in range(1000):
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
    y_pred = model(X_test)

# y_pred: [20, 100, 3]
y_pred = y_pred.cpu().detach().numpy()
y_test = y_test.cpu().detach().numpy()

y_pred_denorm = denormalize_data_seq(y_pred, Y_min, Y_max)   # shape: [20, 100, 3]
y_test_denorm = denormalize_data_seq(y_test, Y_min, Y_max)

grf_x_mean = y_pred_denorm[:, :, 0].mean(axis=0)
grf_real_x_mean = y_test_denorm[:,:,0].mean(axis=0)
grf_y_mean = y_pred_denorm[:, :, 1].mean(axis=0)
grf_real_y_mean = y_test_denorm[:,:,1].mean(axis=0)
grf_z_mean = y_pred_denorm[:, :, 2].mean(axis=0)
grf_real_z_mean = y_test_denorm[:,:,2].mean(axis=0)

x_axis = np.arange(grf_real_x_mean.shape[0])  # shape: [100]

plt.figure()
plt.plot(grf_x_mean, label = 'Predicted GRF-X', color='red')
plt.plot(grf_real_x_mean, label = 'Real GRF-X', color='blue')
plt.fill_between(x_axis, grf_real_x_mean+np.std(y_test_denorm[:,:,0],axis=0),
         grf_real_x_mean-np.std(y_test_denorm[:,:,0],axis=0), color='blue', alpha=0.2)
plt.title("Average Predicted GRF-X over 100 Time Steps")
plt.xlabel("Time Step")
plt.ylabel("GRF-X")
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(grf_y_mean, label = 'Predicted GRF-Y', color='red')
plt.plot(grf_real_y_mean, label = 'Real GRF-Y', color='blue')
plt.fill_between(x_axis, grf_real_y_mean+np.std(y_test_denorm[:,:,1],axis=0),
         grf_real_y_mean-np.std(y_test_denorm[:,:,1],axis=0), color='blue', alpha=0.2)
plt.title("Average Predicted GRF-Y over 100 Time Steps")
plt.xlabel("Time Step")
plt.ylabel("GRF-Y")
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(grf_z_mean, label = 'Predicted GRF-Z', color='red')
plt.plot(grf_real_z_mean, label = 'Real GRF-Z', color='blue')
plt.fill_between(x_axis, grf_real_z_mean+np.std(y_test_denorm[:,:,2],axis=0),
         grf_real_z_mean-np.std(y_test_denorm[:,:,2],axis=0), color='blue', alpha=0.2)
plt.title("Average Predicted GRF-Z over 100 Time Steps")
plt.xlabel("Time Step")
plt.ylabel("GRF-Z")
plt.grid(True)
plt.legend()

plt.show()