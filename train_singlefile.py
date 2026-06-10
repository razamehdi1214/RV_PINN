import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# === Load Excel ===
file_path = "RV_in-vivo_ex-vivo.xlsx"
xls = pd.ExcelFile(file_path)
df_meta = pd.read_excel(xls, sheet_name="RV", header=None)

# Extract hemodynamic features
hemo_feature_names = df_meta.iloc[0, 4:-1].tolist()
hemo_features_all = df_meta.iloc[1:, 4:-1].apply(pd.to_numeric, errors='coerce').reset_index(drop=True)

# Extract fiber and collagen orientation
fiber_orientations_deg = hemo_features_all.iloc[:, 8].astype(np.float32)
collagen_orientations_deg = hemo_features_all.iloc[:, 9].astype(np.float32)
fiber_orientations_deg
# Convert to radians
fiber_orientations_rad = fiber_orientations_deg * np.pi / 180.0
collagen_orientations_rad = collagen_orientations_deg * np.pi / 180.0

hemo_features = hemo_features_all

# Sample ID mapping
hemo_ids_cleaned = df_meta.iloc[1:, 0].apply(lambda x: str(int(float(x)))).tolist()
circ_df = pd.read_excel(xls, sheet_name="circum_stress", header=None)
long_df = pd.read_excel(xls, sheet_name="longit_stress", header=None)
circ_data = circ_df.iloc[1:].reset_index(drop=True)
long_data = long_df.iloc[1:].reset_index(drop=True)
specimen_names = circ_df.iloc[0].astype(str).tolist()
specimen_to_hemo_row = {str(int(float(hemo_ids_cleaned[i]))): i for i in range(len(hemo_ids_cleaned))}

# === Extract All Samples ===
def extract_all_samples():
    Xf, Xn, H, Pf, Pn, fiber_angle, collagen_angle, specimen_ids_cleaned = [], [], [], [], [], [], [], []
    for idx in range(circ_data.shape[1]):
        if pd.isna(circ_data.iloc[:, idx]).all():
            continue
        specimen_name = str(int(float(specimen_names[idx]))) if specimen_names[idx] != 'species' else None
        hemo_idx = specimen_to_hemo_row.get(specimen_name)
        if hemo_idx is None or hemo_idx >= len(hemo_features):
            continue
        hemo_vec = hemo_features.iloc[hemo_idx].values.astype(np.float32)
        fiber_angle_rad = fiber_orientations_rad.iloc[hemo_idx]
        collagen_angle_rad = collagen_orientations_rad.iloc[hemo_idx]
        pf = circ_data.iloc[:, idx].dropna().astype(np.float32).values
        pn = long_data.iloc[:, idx].dropna().astype(np.float32).values
        lam = np.linspace(1.0, 1.3, len(pf)).astype(np.float32)

        Xf.append(lam)
        Xn.append(lam)
        H.append(hemo_vec)
        Pf.append(pf)
        Pn.append(pn)
        fiber_angle.append(fiber_angle_rad)
        collagen_angle.append(collagen_angle_rad)
        specimen_ids_cleaned.append(specimen_name)  # <=== append ID
    return map(np.array, [Xf, Xn, H, Pf, Pn, fiber_angle, collagen_angle, specimen_ids_cleaned])

Xf_all, Xn_all, H_all, Pf_all, Pn_all, fiber_angles_all, collagen_angles_all, specimen_ids_all = extract_all_samples()
Ptotal_all = Pf_all + Pn_all

# === Select test sample index
test_index = 0  # Change this to any index you want for LOOCV test

# === Split test sample
Xf_test = Xf_all[test_index:test_index+1]
Xn_test = Xn_all[test_index:test_index+1]
H_test = H_all[test_index:test_index+1]
Pf_test = Pf_all[test_index:test_index+1]
Pn_test = Pn_all[test_index:test_index+1]
F_test = fiber_angles_all[test_index:test_index+1]
C_test = collagen_angles_all[test_index:test_index+1]
ID_test = specimen_ids_all[test_index]  # Test ID

# === Training data
Xf_train = np.delete(Xf_all, test_index, axis=0)
Xn_train = np.delete(Xn_all, test_index, axis=0)
H_train = np.delete(H_all, test_index, axis=0)
Pf_train = np.delete(Pf_all, test_index, axis=0)
Pn_train = np.delete(Pn_all, test_index, axis=0)
F_train = np.delete(fiber_angles_all, test_index, axis=0)
C_train = np.delete(collagen_angles_all, test_index, axis=0)
ID_train = np.delete(specimen_ids_all, test_index, axis=0)  # Training IDs

Xf_train_main, Xf_val, \
Xn_train_main, Xn_val, \
H_train_main, H_val, \
Pf_train_main, Pf_val, \
Pn_train_main, Pn_val, \
F_train_main, F_val, \
C_train_main, C_val, \
ID_train_main, ID_val = train_test_split(
    Xf_train, Xn_train, H_train, Pf_train, Pn_train, F_train, C_train, ID_train,
    test_size=0.2, random_state=42
)

print("Test ID:", ID_test)
print("Train IDs:", ID_train_main)
print("Validation IDs:", ID_val)


class MultiTaskPINN(tf.keras.Model):
    def __init__(self, hemo_dim):
        super().__init__()
        self.param_net = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dense(8, activation='softplus', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        ])

    def call(self, stretch_f, stretch_n, hemo, fiber_angle_rad, collagen_angle_rad):
        eps = 1e-6

        # Predict material parameters from hemodynamics
        params = self.param_net(hemo)
        params = tf.where(tf.math.is_nan(params), tf.zeros_like(params), params)
        params = tf.clip_by_value(params, 0.0, 50.0)  # Clip to avoid inf values
        a, b, af, bf, ac, bc, afc, bfc = tf.split(params, 8, axis=-1)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch([stretch_f, stretch_n])

            stretch_s = 1.0 / (stretch_f * stretch_n + eps)
            C_ff = stretch_f**2
            C_nn = stretch_n**2
            C_ss = stretch_s**2

            f1 = tf.expand_dims(tf.math.cos(fiber_angle_rad), axis=1)
            f2 = tf.expand_dims(tf.math.sin(fiber_angle_rad), axis=1)
            c1 = tf.expand_dims(tf.math.cos(collagen_angle_rad), axis=1)
            c2 = tf.expand_dims(tf.math.sin(collagen_angle_rad), axis=1)

            I1 = C_ff + C_nn + C_ss
            I4f = f1**2 * C_ff + f2**2 * C_nn
            I4c = c1**2 * C_ff + c2**2 * C_nn
            I8fc = f1 * c1 * C_ff + f2 * c2 * C_nn

            def safe_exp(x): return tf.exp(tf.clip_by_value(x, -50.0, 50.0))

            psi = (
                a * (safe_exp(b * (I1 - 3.0)) - 1.0) +
                af * (safe_exp(bf * tf.square(I4f - 1.0)) - 1.0) +
                ac * (safe_exp(bc * tf.square(I4c - 1.0)) - 1.0) +
                afc * (safe_exp(bfc * tf.square(I8fc)) - 1.0)
            )

        # Derivatives
        dPsi_dI1 = tape.gradient(psi, I1)
        dPsi_dI4f = tape.gradient(psi, I4f)
        dPsi_dI4c = tape.gradient(psi, I4c)
        dPsi_dI8fc = tape.gradient(psi, I8fc)
        del tape

        # Pressure term p
        p = 2.0 / (stretch_f * stretch_n + eps) * dPsi_dI1

        dI1_dlf = 2.0 * stretch_f - 2.0 / (stretch_f**3 * stretch_n**2 + eps)
        dI1_dln = 2.0 * stretch_n - 2.0 / (stretch_f**2 * stretch_n**3 + eps)
        dps_ds_dlf = 2.0 / (stretch_f**2 * stretch_n + eps)
        dps_ds_dln = 2.0 / (stretch_f * stretch_n**2 + eps)

        dI4f_dlf = 2.0 * stretch_f * f1**2
        dI4f_dln = 2.0 * stretch_n * f2**2
        dI4c_dlf = 2.0 * stretch_f * c1**2
        dI4c_dln = 2.0 * stretch_n * c2**2
        dI8fc_dlf = 2.0 * stretch_f * f1 * c1
        dI8fc_dln = 2.0 * stretch_n * f2 * c2

        Pf = (
            dPsi_dI1 * dI1_dlf +
            dPsi_dI4f * dI4f_dlf +
            dPsi_dI4c * dI4c_dlf +
            dPsi_dI8fc * dI8fc_dlf +
            p * dps_ds_dlf
        )

        Pn = (
            dPsi_dI1 * dI1_dln +
            dPsi_dI4f * dI4f_dln +
            dPsi_dI4c * dI4c_dln +
            dPsi_dI8fc * dI8fc_dln +
            p * dps_ds_dln
        )

        # return Pf + Pn, params
        return Pf + Pn, (a, af, ac, afc, b, bf, bc, bfc)


def stress_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# === Training ===
@tf.function
def train_step(xf, xn, hemo, fiber_angle, collagen_angle, y_total_true):
    with tf.GradientTape() as tape:
        P_total_pred, _ = model(xf, xn, hemo, fiber_angle, collagen_angle)
        loss = stress_loss(y_total_true, P_total_pred)

    grads = tape.gradient(loss, model.trainable_variables)
    grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# Compute total stress
Ptotal_train_main = Pf_train_main + Pn_train_main
Ptotal_val = Pf_val + Pn_val

# Create training dataset
train_ds = tf.data.Dataset.from_tensor_slices((
    Xf_train_main, Xn_train_main, H_train_main, F_train_main, C_train_main, Ptotal_train_main
)).batch(Xf_train_main.shape[0])

# Create validation dataset
val_ds = tf.data.Dataset.from_tensor_slices((
    Xf_val, Xn_val, H_val, F_val, C_val, Ptotal_val
)).batch(Xf_val.shape[0])


# === Initialize model and optimizer
model = MultiTaskPINN(hemo_dim=H_train.shape[1])
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=200,
    decay_rate=0.9,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# === Define validation loss function (same as training)
def val_step(xf, xn, hemo, fiber_angle, collagen_angle, y_true):
    y_pred, _ = model(xf, xn, hemo, fiber_angle, collagen_angle)
    return stress_loss(y_true, y_pred)

# === Training loop
best_val_loss = float('inf')  # initialize with a large value
best_epoch = -1  # optional, just for tracking

for epoch in range(5000):
    epoch_loss = 0.0
    batch_count = 0

    for batch in train_ds:
        xf, xn, hemo, fiber_angle, collagen_angle, y_total = batch
        loss = train_step(xf, xn, hemo, fiber_angle, collagen_angle, y_total)
        epoch_loss += loss.numpy()
        batch_count += 1

    # === Validation step
    val_loss = 0.0
    val_batches = 0
    for batch in val_ds:
        xf, xn, hemo, fiber_angle, collagen_angle, y_val = batch
        loss = val_step(xf, xn, hemo, fiber_angle, collagen_angle, y_val)
        val_loss += loss.numpy()
        val_batches += 1

    val_loss_avg = val_loss / val_batches
    train_loss_avg = epoch_loss / batch_count

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}: train_loss = {train_loss_avg:.5f}, val_loss = {val_loss_avg:.5f}")

    # === Save weights if current val loss is lower
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        best_epoch = epoch + 1
        model.save_weights("trained_model_weights.h5")
        print(f"✅ Saved model at epoch {epoch+1} with new best val_loss = {val_loss_avg:.5f}")


# === Save weights
model.save_weights("trained_model_weights.h5")
print("Model weights saved.")

# === Reload model and predict on test sample
model_test = MultiTaskPINN(hemo_dim=H_train.shape[1])

# Warm-up with dummy TensorFlow inputs (not NumPy arrays)
dummy_stretch = tf.convert_to_tensor(np.zeros((1, 31), dtype=np.float32))
dummy_hemo = tf.convert_to_tensor(np.zeros((1, H_train.shape[1]), dtype=np.float32))
dummy_angle = tf.convert_to_tensor(np.zeros((1, 1), dtype=np.float32))

_ = model_test(dummy_stretch, dummy_stretch, dummy_hemo, dummy_angle, dummy_angle)

# Load weights after model is built
model_test.load_weights("trained_model_weights.h5")
print("Model weights loaded for test prediction.")


# === Predict
X_f = tf.convert_to_tensor(Xf_test[0].reshape(1, -1), dtype=tf.float32)
X_n = tf.convert_to_tensor(Xn_test[0].reshape(1, -1), dtype=tf.float32)
H_t = tf.convert_to_tensor(H_test[0].reshape(1, -1), dtype=tf.float32)
F_t = tf.convert_to_tensor(F_test[0].reshape(1, -1), dtype=tf.float32)
C_t = tf.convert_to_tensor(C_test[0].reshape(1, -1), dtype=tf.float32)


from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
# === Predict again
P_total_true = Pf_test[0] + Pn_test[0]

P_total_pred, param_tensors = model_test(X_f, X_n, H_t, F_t, C_t)

# Each element in param_tensors is a (1, 1) tensor → extract scalar
param_names = ['a', 'af', 'ac', 'afc', 'b', 'bf', 'bc', 'bfc']
param_values = [p.numpy().flatten()[0] for p in param_tensors]

# Print them
print("Predicted Material Parameters:")
for name, value in zip(param_names, param_values):
    print(f" {name} = {value:.4f}")

# === Calculate R²
r2 = r2_score(P_total_true, P_total_pred.numpy().flatten())

# === Store results
df_result = pd.DataFrame({
    'lambda': Xf_test[0],
    'P_total_true': P_total_true,
    'P_total_pred': P_total_pred.numpy().flatten()
})
output_path = f"test_sample_ID_{ID_test}_prediction.csv"
df_result.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")



# === Plot with R² and parameters
plt.figure(figsize=(8, 5))
plt.plot(df_result['lambda'], df_result['P_total_true'], 'o-', label='P_total_true')
plt.plot(df_result['lambda'], df_result['P_total_pred'], 's--', label='P_total_pred')
plt.xlabel("Lambda (Stretch)")
plt.ylabel("Total First Piola-Kirchhoff Stress (kPa)")
param_str = ", ".join([f"{name}={value:.4f}" for name, value in zip(param_names, param_values)])
plt.title(f"Specimen ID: {ID_test} - Predicted vs True Total Stress\nR² = {r2:.4f}\n{param_str}", fontsize=10)
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the figure
fig_path = f"test_sample_ID_{ID_test}_prediction_plot.png"
plt.savefig(fig_path, dpi=300)
print(f"Figure saved to {fig_path}")
plt.show()

# === Plot each training sample (excluding test) in its own subplot
num_samples = len(Xf_all)
num_plots = num_samples - 1  # exclude test sample
cols = 4  # number of subplot columns
rows = (num_plots + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), sharex=True, sharey=True)
axes = axes.flatten()

specimen_ids_all = np.array(specimen_ids_all)  # in case it's still a list
plot_idx = 0
for i in range(num_samples):
    if i == test_index:
        continue  # skip test sample
    ax = axes[plot_idx]

    X_f_i = tf.convert_to_tensor(Xf_all[i].reshape(1, -1), dtype=tf.float32)
    X_n_i = tf.convert_to_tensor(Xn_all[i].reshape(1, -1), dtype=tf.float32)
    H_i = tf.convert_to_tensor(H_all[i].reshape(1, -1), dtype=tf.float32)
    F_i = tf.convert_to_tensor(fiber_angles_all[i].reshape(1, -1), dtype=tf.float32)
    C_i = tf.convert_to_tensor(collagen_angles_all[i].reshape(1, -1), dtype=tf.float32)

    P_true_i = Pf_all[i] + Pn_all[i]
    P_pred_i, _ = model_test(X_f_i, X_n_i, H_i, F_i, C_i)

    # Calculate R² for the current sample
    r2_i = r2_score(P_true_i, P_pred_i.numpy().flatten())

    # Plot with R² in the legend
    ax.plot(Xf_all[i], P_true_i, 'k-', label='True')
    ax.plot(Xf_all[i], P_pred_i.numpy().flatten(), 'r--', label=f'Predicted (R²={r2_i:.2f})')
    ax.set_title(f"ID: {specimen_ids_all[i]}")
    ax.grid(True)
    ax.legend()
    plot_idx += 1

# Hide unused subplots
for j in range(plot_idx, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Predicted vs True Total Stress on Training Samples", fontsize=16)
fig.text(0.5, 0.04, 'Lambda (Stretch)', ha='center')
fig.text(0.04, 0.5, 'Total Stress (kPa)', va='center', rotation='vertical')
fig.tight_layout(rect=[0.03, 0.03, 1, 0.97])
fig.savefig("training_samples_subplots.png", dpi=300)
plt.show()

