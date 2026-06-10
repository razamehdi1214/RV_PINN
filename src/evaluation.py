import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from .config import MODEL_WEIGHTS
from .model import MultiTaskPINN
from .trainer import warmup_model


def build_test_model(hemo_dim):
    model = MultiTaskPINN(hemo_dim=hemo_dim)
    warmup_model(model, hemo_dim)
    model.load_weights(MODEL_WEIGHTS)
    return model


def evaluate_test_sample(model, split):
    hemo_dim = split["H_train"].shape[1]
    model_test = build_test_model(hemo_dim)

    X_f = tf.convert_to_tensor(split["Xf_test"][0].reshape(1, -1), dtype=tf.float32)
    X_n = tf.convert_to_tensor(split["Xn_test"][0].reshape(1, -1), dtype=tf.float32)
    H_t = tf.convert_to_tensor(split["H_test"][0].reshape(1, -1), dtype=tf.float32)
    F_t = tf.convert_to_tensor(split["F_test"][0].reshape(1, -1), dtype=tf.float32)
    C_t = tf.convert_to_tensor(split["C_test"][0].reshape(1, -1), dtype=tf.float32)

    P_total_true = split["Pf_test"][0] + split["Pn_test"][0]
    P_total_pred, param_tensors = model_test(X_f, X_n, H_t, F_t, C_t)

    param_names = ['a', 'af', 'ac', 'afc', 'b', 'bf', 'bc', 'bfc']
    param_values = [p.numpy().flatten()[0] for p in param_tensors]
    r2 = r2_score(P_total_true, P_total_pred.numpy().flatten())

    df_result = pd.DataFrame({
        'lambda': split["Xf_test"][0],
        'P_total_true': P_total_true,
        'P_total_pred': P_total_pred.numpy().flatten()
    })
    output_path = f"test_sample_ID_{split['ID_test']}_prediction.csv"
    df_result.to_csv(output_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df_result['lambda'], df_result['P_total_true'], 'o-', label='P_total_true')
    plt.plot(df_result['lambda'], df_result['P_total_pred'], 's--', label='P_total_pred')
    plt.xlabel("Lambda (Stretch)")
    plt.ylabel("Total First Piola-Kirchhoff Stress (kPa)")
    param_str = ", ".join([f"{name}={value:.4f}" for name, value in zip(param_names, param_values)])
    plt.title(f"Specimen ID: {split['ID_test']} - Predicted vs True Total Stress\nR² = {r2:.4f}\n{param_str}", fontsize=10)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"test_sample_ID_{split['ID_test']}_prediction_plot.png", dpi=300)
    plt.close()


def plot_training_samples(model, data, test_index):
    hemo_dim = data["H_all"].shape[1]
    model_test = build_test_model(hemo_dim)

    num_samples = len(data["Xf_all"])
    num_plots = num_samples - 1
    cols = 4
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), sharex=True, sharey=True)
    axes = axes.flatten()
    plot_idx = 0

    for i in range(num_samples):
        if i == test_index:
            continue
        ax = axes[plot_idx]
        X_f_i = tf.convert_to_tensor(data["Xf_all"][i].reshape(1, -1), dtype=tf.float32)
        X_n_i = tf.convert_to_tensor(data["Xn_all"][i].reshape(1, -1), dtype=tf.float32)
        H_i = tf.convert_to_tensor(data["H_all"][i].reshape(1, -1), dtype=tf.float32)
        F_i = tf.convert_to_tensor(data["fiber_angles_all"][i].reshape(1, -1), dtype=tf.float32)
        C_i = tf.convert_to_tensor(data["collagen_angles_all"][i].reshape(1, -1), dtype=tf.float32)

        P_true_i = data["Pf_all"][i] + data["Pn_all"][i]
        P_pred_i, _ = model_test(X_f_i, X_n_i, H_i, F_i, C_i)
        r2_i = r2_score(P_true_i, P_pred_i.numpy().flatten())

        ax.plot(data["Xf_all"][i], P_true_i, 'k-', label='True')
        ax.plot(data["Xf_all"][i], P_pred_i.numpy().flatten(), 'r--', label=f'Predicted (R²={r2_i:.2f})')
        ax.set_title(f"ID: {data['specimen_ids_all'][i]}")
        ax.grid(True)
        ax.legend()
        plot_idx += 1

    for j in range(plot_idx, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Predicted vs True Total Stress on Training Samples", fontsize=16)
    fig.text(0.5, 0.04, 'Lambda (Stretch)', ha='center')
    fig.text(0.04, 0.5, 'Total Stress (kPa)', va='center', rotation='vertical')
    fig.tight_layout(rect=[0.03, 0.03, 1, 0.97])
    fig.savefig("training_samples_subplots.png", dpi=300)
    plt.close(fig)
