import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import argparse
import sys
from scipy.linalg import cholesky, LinAlgError
import scipy
import time
from time import perf_counter
import os, random


def set_seed(seed: int = 42, deterministic_tf: bool = True):
    # 1) Python & hashing
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2) (optional) TF determinism 
    if deterministic_tf:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"   # older+current TF flag

    # 3) Now import and seed libs
    import numpy as np
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Newer TF also has an explicit determinism switch:
    try:
        # available in TF 2.8+ (name may vary slightly across versions)
        tf.config.experimental.enable_op_determinism(True)
    except Exception:
        pass

    return seed


def enable_float64():
    import numpy as np
    import tensorflow as tf

    # ---- NumPy ----
    # NumPy already prefers float64 for many constructors:
    np_float = np.float64

    # If you create arrays, do: np.array(..., dtype=np_float) or np.ones(..., dtype=np_float)
    # For rng outputs: np.random.rand(...).astype(np_float)

    # ---- TensorFlow ----
    # Make Keras layers/variables default to float64
    tf.keras.backend.set_floatx('float64')

    # If you use the dtype policy API (TF 2.x), set it globally:
    try:
        from tensorflow.keras import mixed_precision as mp
        mp.set_global_policy('float64')   # compute & variable dtypes -> float64
    except Exception:
        pass  # older TF versions may not have/accept this policy; set_floatx above is enough for Keras

    # Optional: match tf.experimental.numpy to NumPy behavior in float64
    try:
        tf.experimental.numpy.experimental_enable_numpy_behavior(prefer_float64=True)
    except Exception:
        pass

    # Optional (NVIDIA Ampere+): disable TF32 so matmul/conv numerics aren’t silently “looser”
    try:
        tf.config.experimental.enable_tensor_float_32_execution(False)
    except Exception:
        pass

# call once, early:
enable_float64()



parser = argparse.ArgumentParser(description="Multi-task PINN runner")
parser.add_argument(
    "--test-index",
    type=int,
    default=0,
    help="Zero-based index of the test sample (default: 0)",
)

parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Seed for initialization (default: 0)",
)

args = parser.parse_args()
test_index = args.test_index

# call once, early:
set_seed(args.seed)


# === Load Excel ===
file_path = "RV_in-vivo_ex-vivo.xlsx"
xls = pd.ExcelFile(file_path)
df_meta = pd.read_excel(xls, sheet_name="RV", header=None)

# Extract all hemodynamic features excluding last column (e.g., label)
hemo_feature_names = df_meta.iloc[0, 4:-1].tolist()
hemo_features_all = df_meta.iloc[1:, 4:-1].apply(pd.to_numeric, errors='coerce').reset_index(drop=True)

# Extract fiber and collagen orientation in degrees
fiber_orientations_deg = hemo_features_all.iloc[:, 8].astype(np.float64)
collagen_orientations_deg = hemo_features_all.iloc[:, 9].astype(np.float64)
fiber_orientations_deg
# Convert to radians
fiber_orientations_rad = fiber_orientations_deg * np.pi / 180.0
collagen_orientations_rad = collagen_orientations_deg * np.pi / 180.0

# Use all hemodynamic features
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
        hemo_vec = hemo_features.iloc[hemo_idx].values.astype(np.float64)
        fiber_angle_rad = fiber_orientations_rad.iloc[hemo_idx]
        collagen_angle_rad = collagen_orientations_rad.iloc[hemo_idx]
        pf = circ_data.iloc[:, idx].dropna().astype(np.float64).values
        pn = long_data.iloc[:, idx].dropna().astype(np.float64).values
        lam = np.linspace(1.0, 1.3, len(pf)).astype(np.float64)

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
    test_size=0.05, random_state=42
)

print("Test ID:", ID_test)
print("Train IDs:", ID_train_main)
print("Validation IDs:", ID_val)


import tensorflow as tf

class MultiTaskPINN(tf.keras.Model):
    def __init__(self, hemo_dim, bmax=2.0, eps=1e-6):
        super().__init__()
        self.eps = float(eps)
        self.bmax = float(bmax)

        # Raw head (no final activation). We'll map to sensible params below.
        self.param_net = tf.keras.Sequential([
            tf.keras.layers.Dense(4, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dense(4, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dense(4, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dense(8, activation=None, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        ])

    # ---- Stable exponential: bound the exponent argument smoothly with tanh ----
    # Returns exp(s_bounded) - 1 using expm1 for better numerics near 0
    @tf.function
    def safe_expm1(self, s, smax=5.0):
        smax = tf.cast(smax, s.dtype)
        s_bounded = smax * tf.tanh(s / smax)   # smoothly in (-smax, smax)
        return tf.math.expm1(s_bounded)

    def call(self, stretch_f, stretch_n, hemo, fiber_angle_rad, collagen_angle_rad):
        eps = self.eps

        # Predict raw params
        raw = self.param_net(hemo)
        raw = tf.where(tf.math.is_nan(raw), tf.zeros_like(raw), raw)

        # Map to physically sensible, smoothly bounded params:
        # a-like >= 0 (softplus), b-like in [0, bmax] (scaled sigmoid)
        a, b, af, bf, ac, bc, afc, bfc = tf.split(raw, 8, axis=-1)
        a   = tf.nn.softplus(a)   + 1e-6
        af  = tf.nn.softplus(af)  + 1e-6
        ac  = tf.nn.softplus(ac)  + 1e-6
        afc = tf.nn.softplus(afc) + 1e-6

        b   = self.bmax * tf.nn.sigmoid(b)
        bf  = self.bmax * tf.nn.sigmoid(bf)
        bc  = self.bmax * tf.nn.sigmoid(bc)
        bfc = self.bmax * tf.nn.sigmoid(bfc)

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

            I1   = C_ff + C_nn + C_ss
            I4f  = f1**2 * C_ff + f2**2 * C_nn
            I4c  = c1**2 * C_ff + c2**2 * C_nn
            I8fc = f1 * c1 * C_ff + f2 * c2 * C_nn

            # Exponential form preserved; only the exponent's argument is smoothly bounded
            psi = (
                a   * self.safe_expm1(b  * (I1 - 3.0),             smax=5.0) +
                af  * self.safe_expm1(bf * tf.square(I4f - 1.0),   smax=5.0) +
                ac  * self.safe_expm1(bc * tf.square(I4c - 1.0),   smax=5.0) +
                afc * self.safe_expm1(bfc * tf.square(I8fc),       smax=5.0)
            )

        # Derivatives wrt invariants
        dPsi_dI1   = tape.gradient(psi, I1)
        dPsi_dI4f  = tape.gradient(psi, I4f)
        dPsi_dI4c  = tape.gradient(psi, I4c)
        dPsi_dI8fc = tape.gradient(psi, I8fc)
        del tape

        # Pressure-like term p
        p = 2.0 / (stretch_f * stretch_n + eps) * dPsi_dI1

        # Useful partials
        dI1_dlf    = 2.0 * stretch_f - 2.0 / (stretch_f**3 * stretch_n**2 + eps)
        dI1_dln    = 2.0 * stretch_n - 2.0 / (stretch_f**2 * stretch_n**3 + eps)
        dps_ds_dlf = 2.0 / (stretch_f**2 * stretch_n + eps)
        dps_ds_dln = 2.0 / (stretch_f * stretch_n**2 + eps)

        dI4f_dlf   = 2.0 * stretch_f * f1**2
        dI4f_dln   = 2.0 * stretch_n * f2**2
        dI4c_dlf   = 2.0 * stretch_f * c1**2
        dI4c_dln   = 2.0 * stretch_n * c2**2
        dI8fc_dlf  = 2.0 * stretch_f * f1 * c1
        dI8fc_dln  = 2.0 * stretch_n * f2 * c2

        Pf = (
            dPsi_dI1  * dI1_dlf  +
            dPsi_dI4f * dI4f_dlf +
            dPsi_dI4c * dI4c_dlf +
            dPsi_dI8fc* dI8fc_dlf +
            p         * dps_ds_dlf
        )

        Pn = (
            dPsi_dI1  * dI1_dln  +
            dPsi_dI4f * dI4f_dln +
            dPsi_dI4c * dI4c_dln +
            dPsi_dI8fc* dI8fc_dln +
            p         * dps_ds_dln
        )

        return Pf + Pn, (a, af, ac, afc, b, bf, bc, bfc)


'''
huber = tf.keras.losses.Huber(delta=1.0)   # tune 0.5–2.0
def stress_loss(y_true, y_pred):
    return huber(y_true, y_pred)
'''

def stress_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# === Training ===
@tf.function
def train_step(xf, xn, hemo, fiber_angle, collagen_angle, y_total_true):
    with tf.GradientTape() as tape:
        P_total_pred, _ = model(xf, xn, hemo, fiber_angle, collagen_angle)
        loss = stress_loss(y_total_true, P_total_pred) +  tf.add_n(model.losses) 

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

###################################
# 1) Full-batch tensors (deterministic gradients for quasi-Newton)
xf_tr = tf.convert_to_tensor(Xf_train_main, dtype=tf.float64)
xn_tr = tf.convert_to_tensor(Xn_train_main, dtype=tf.float64)
he_tr = tf.convert_to_tensor(H_train_main,  dtype=tf.float64)
fi_tr = tf.convert_to_tensor(F_train_main,  dtype=tf.float64)
co_tr = tf.convert_to_tensor(C_train_main,  dtype=tf.float64)
y_tr  = tf.convert_to_tensor(Ptotal_train_main, dtype=tf.float64)

xf_va = tf.convert_to_tensor(Xf_val, dtype=tf.float64)
xn_va = tf.convert_to_tensor(Xn_val, dtype=tf.float64)
he_va = tf.convert_to_tensor(H_val,  dtype=tf.float64)
fi_va = tf.convert_to_tensor(F_val,  dtype=tf.float64)
co_va = tf.convert_to_tensor(C_val, dtype=tf.float64)
y_va  = tf.convert_to_tensor(Ptotal_val, dtype=tf.float64)




# --- 0) Setup: validation step (unchanged)
def val_step(xf, xn, hemo, fiber_angle, collagen_angle, y_true):
    y_pred, _ = model(xf, xn, hemo, fiber_angle, collagen_angle)
    return stress_loss(y_true, y_pred)

# --- 1) TF checkpoint manager (NO .h5 anywhere)
import tensorflow as tf
from pathlib import Path

ckpt_dir = Path(f"ckpts_{ID_test}"); ckpt_dir.mkdir(exist_ok=True)
ckpt = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(
    ckpt, directory=str(ckpt_dir), max_to_keep=1, checkpoint_name="best"
)

best_val_loss = float("inf")
best_epoch = -1

start_time = time.time()

# --- 2) Your SGD/Adam epoch loop
for epoch in range(10):
    epoch_loss = 0.0
    batch_count = 0

    for batch in train_ds:
        xf, xn, hemo, fiber_angle, collagen_angle, y_total = batch
        loss = train_step(xf, xn, hemo, fiber_angle, collagen_angle, y_total)
        epoch_loss += float(loss.numpy())
        batch_count += 1

    # Validation
    val_loss = 0.0
    val_batches = 0
    for batch in val_ds:
        xf, xn, hemo, fiber_angle, collagen_angle, y_val = batch
        loss = val_step(xf, xn, hemo, fiber_angle, collagen_angle, y_val)
        val_loss += float(loss.numpy())
        val_batches += 1

    val_loss_avg = val_loss / val_batches
    train_loss_avg = epoch_loss / batch_count

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}: train_loss = {train_loss_avg:.5f}, val_loss = {val_loss_avg:.5f}")

    # Save "best" via TF checkpoints
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        best_epoch = epoch + 1
        manager.save()  # writes ckpts/best-00001, ckpts/best-00002, ...
        print(f"✅ Saved model at epoch {best_epoch} with new best val_loss = {val_loss_avg:.5f}")

# --- 6) Restore best into the training model
latest = tf.train.latest_checkpoint(str(ckpt_dir))
if latest is not None:
    ckpt.restore(latest).expect_partial()
    print(f"Restored best from {latest} "
          f"(epoch {best_epoch if best_epoch>0 else best.get('it','?')}, "
          f"val_loss={best_val_loss if best_val_loss<float('inf') else best.get('val'):.6f})")
else:
    print("No checkpoint found; keeping current weights.")

ckpt_second = tf.train.Checkpoint(model=model)
if latest is not None:
    ckpt_second.restore(latest).expect_partial()
    print("model_test restored from best checkpoint.")
else:
    print("No checkpoint to restore into model_test.")

##################################


# --- 3) Flatten/restore helpers (unchanged from your code, keep as-is)
train_vars = model.trainable_variables
shapes = [tuple(v.shape.as_list()) for v in train_vars]
sizes  = [int(np.prod(s)) for s in shapes]
cuts   = np.cumsum([0] + sizes).astype(int)

def pack_weights() -> np.ndarray:
    flat = [tf.reshape(v, [-1]) for v in train_vars]
    vec  = tf.concat(flat, axis=0).numpy()
    return vec.astype(np.float64)

def unpack_weights(theta: np.ndarray):
    for v, i0, i1, shp in zip(train_vars, cuts[:-1], cuts[1:], shapes):
        chunk = theta[i0:i1].reshape(shp)
        v.assign(tf.convert_to_tensor(chunk, dtype=tf.as_dtype(v.dtype)))

@tf.function
def compute_loss_and_grads():
    with tf.GradientTape() as tape:
        y_pred, _ = model(xf_tr, xn_tr, he_tr, fi_tr, co_tr)
        total = stress_loss(y_tr, y_pred) + tf.add_n(model.losses)
    grads = tape.gradient(total, train_vars)
    grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, train_vars)]
    return total, grads

def f_and_grad(theta: np.ndarray):
    unpack_weights(theta)
    total, grads = compute_loss_and_grads()
    gflat = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
    return float(total.numpy()), gflat.numpy().astype(np.float64)

def val_loss_now() -> float:
    ypv, _ = model(xf_va, xn_va, he_va, fi_va, co_va)
    return float(stress_loss(y_va, ypv).numpy())

# --- 4) SciPy/BFGS callback that ALSO saves a TF checkpoint when val improves
best = {"val": float(best_val_loss), "it": -1}

def s_callback(xk: np.ndarray):
    unpack_weights(xk)          # put xk into model
    v = val_loss_now()

    if v < best["val"]:
        best["val"] = float(v)
        best["it"]  = int(s_callback.it)
        manager.save()          # <--- TF checkpoint save (same manager)
        print(f"✅ [iter {s_callback.it}] new best val_loss = {v:.6f} — checkpoint saved")

    if s_callback.it % 25 == 0:
        tr, _ = f_and_grad(xk)
        print(f"[iter {s_callback.it}] train_loss={tr:.6f}  val_loss={v:.6f}")

    s_callback.it += 1

s_callback.it = 1

# --- 5) Run SciPy minimize (unchanged)
theta0 = pack_weights()
H0 = tf.eye(len(theta0), dtype=tf.float64).numpy()

count = 0
while count < 5:
    res = minimize(
        f_and_grad,
        theta0,
        method="BFGS",
        jac=True,
        options={
            "maxiter": 500,
            "gtol": 0,
            "hess_inv0": H0,
            "method_bfgs": "SSBroyden",
            "initial_scale": False,
        },
        tol=0,
        callback=s_callback,
    )
    
    
    theta0 = res.x
    H0 = res.hess_inv
    H0 = 0.5 * (H0 + H0.T)
    try:
        cholesky(H0)
    except LinAlgError:
        H0 = tf.eye(len(theta0), dtype=tf.float64).numpy()
    count = count + 1


end_time = time.time()

#print("SciPy:", res.message)

# --- 6) Restore best into the training model
latest = tf.train.latest_checkpoint(str(ckpt_dir))
if latest is not None:
    ckpt.restore(latest).expect_partial()
    print(f"Restored best from {latest} "
          f"(epoch {best_epoch if best_epoch>0 else best.get('it','?')}, "
          f"val_loss={best_val_loss if best_val_loss<float('inf') else best.get('val'):.6f})")
else:
    print("No checkpoint found; keeping current weights.")

# --- 7) Restore into a fresh model_test (optional)
model_test = MultiTaskPINN(hemo_dim=H_train.shape[1])
# Make sure variables exist (dummy forward pass with a real batch):
xf, xn, hemo, fiber_angle, collagen_angle, y_val = next(iter(val_ds))
_ = model_test(xf, xn, hemo, fiber_angle, collagen_angle, training=False)

ckpt_test = tf.train.Checkpoint(model=model_test)
if latest is not None:
    ckpt_test.restore(latest).expect_partial()
    print("model_test restored from best checkpoint.")
else:
    print("No checkpoint to restore into model_test.")

##################################



# === Predict
X_f = tf.convert_to_tensor(Xf_test[0].reshape(1, -1), dtype=tf.float64)
X_n = tf.convert_to_tensor(Xn_test[0].reshape(1, -1), dtype=tf.float64)
H_t = tf.convert_to_tensor(H_test[0].reshape(1, -1), dtype=tf.float64)
F_t = tf.convert_to_tensor(F_test[0].reshape(1, -1), dtype=tf.float64)
C_t = tf.convert_to_tensor(C_test[0].reshape(1, -1), dtype=tf.float64)


from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
# === Predict again
P_total_true = Pf_test[0] + Pn_test[0]

P_total_pred, param_tensors = model_test(X_f, X_n, H_t, F_t, C_t)

# Each element in param_tensors is a (1, 1) tensor → extract scalar
param_names = ['a', 'af', 'ac', 'afc', 'b', 'bf', 'bc', 'bfc']
param_values = [p.numpy().flatten()[0] for p in param_tensors]

# Print them nicely
print("Predicted Material Parameters:")
for name, value in zip(param_names, param_values):
    print(f" {name} = {value:.4f}")

# === Calculate R²
r2 = r2_score(P_total_true, P_total_pred.numpy().flatten())

print(f"RRRRRRRR22222222222222: {r2}")

print(f"time taken: {end_time - start_time}")

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

    X_f_i = tf.convert_to_tensor(Xf_all[i].reshape(1, -1), dtype=tf.float64)
    X_n_i = tf.convert_to_tensor(Xn_all[i].reshape(1, -1), dtype=tf.float64)
    H_i = tf.convert_to_tensor(H_all[i].reshape(1, -1), dtype=tf.float64)
    F_i = tf.convert_to_tensor(fiber_angles_all[i].reshape(1, -1), dtype=tf.float64)
    C_i = tf.convert_to_tensor(collagen_angles_all[i].reshape(1, -1), dtype=tf.float64)

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
fig.savefig(f"training_samples_subplots_{ID_test}.png", dpi=300)
plt.show()

