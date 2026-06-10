import numpy as np
import tensorflow as tf
from .config import LEARNING_RATE, DECAY_STEPS, DECAY_RATE, MODEL_WEIGHTS, GRAD_CLIP_MIN, GRAD_CLIP_MAX
from .losses import stress_loss


def build_optimizer():
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=DECAY_STEPS,
        decay_rate=DECAY_RATE,
        staircase=True,
    )
    return tf.keras.optimizers.Adam(learning_rate=lr_schedule)


@tf.function
def train_step(model, optimizer, xf, xn, hemo, fiber_angle, collagen_angle, y_total_true):
    with tf.GradientTape() as tape:
        y_pred, _ = model(xf, xn, hemo, fiber_angle, collagen_angle)
        loss = stress_loss(y_total_true, y_pred)
    grads = tape.gradient(loss, model.trainable_variables)
    grads = [tf.clip_by_value(g, GRAD_CLIP_MIN, GRAD_CLIP_MAX) if g is not None else g for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def val_step(model, xf, xn, hemo, fiber_angle, collagen_angle, y_true):
    y_pred, _ = model(xf, xn, hemo, fiber_angle, collagen_angle)
    return stress_loss(y_true, y_pred)


def warmup_model(model, hemo_dim, stretch_len=31):
    dummy_stretch = tf.convert_to_tensor(np.zeros((1, stretch_len), dtype=np.float32))
    dummy_hemo = tf.convert_to_tensor(np.zeros((1, hemo_dim), dtype=np.float32))
    dummy_angle = tf.convert_to_tensor(np.zeros((1, 1), dtype=np.float32))
    _ = model(dummy_stretch, dummy_stretch, dummy_hemo, dummy_angle, dummy_angle)


def train_model(model, train_ds, val_ds, epochs=500):
    optimizer = build_optimizer()
    best_val_loss = float('inf')
    best_epoch = -1
    history = {"train_loss": [], "val_loss": [], "best_epoch": None}

    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for batch in train_ds:
            xf, xn, hemo, fiber_angle, collagen_angle, y_total = batch
            loss = train_step(model, optimizer, xf, xn, hemo, fiber_angle, collagen_angle, y_total)
            epoch_loss += loss.numpy()
            batch_count += 1

        val_loss = 0.0
        val_batches = 0
        for batch in val_ds:
            xf, xn, hemo, fiber_angle, collagen_angle, y_val = batch
            loss = val_step(model, xf, xn, hemo, fiber_angle, collagen_angle, y_val)
            val_loss += loss.numpy()
            val_batches += 1

        train_loss_avg = epoch_loss / batch_count
        val_loss_avg = val_loss / val_batches
        history["train_loss"].append(train_loss_avg)
        history["val_loss"].append(val_loss_avg)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: train_loss = {train_loss_avg:.5f}, val_loss = {val_loss_avg:.5f}")

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_epoch = epoch + 1
            model.save_weights(MODEL_WEIGHTS)
            print(f"Saved model at epoch {epoch+1} with new best val_loss = {val_loss_avg:.5f}")

    history["best_epoch"] = best_epoch
    print(f"Best epoch: {best_epoch}")
    return model, history
