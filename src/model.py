import tensorflow as tf
from .config import EPS, PARAM_CLIP_MIN, PARAM_CLIP_MAX
from .physics import compute_invariants, compute_stress_terms


class MultiTaskPINN(tf.keras.Model):
    def __init__(self, hemo_dim):
        super().__init__()
        self.hemo_dim = hemo_dim
        self.param_net = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            tf.keras.layers.Dense(8, activation='softplus', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        ])

    def call(self, stretch_f, stretch_n, hemo, fiber_angle_rad, collagen_angle_rad):
        params = self.param_net(hemo)
        params = tf.where(tf.math.is_nan(params), tf.zeros_like(params), params)
        params = tf.clip_by_value(params, PARAM_CLIP_MIN, PARAM_CLIP_MAX)
        a, b, af, bf, ac, bc, afc, bfc = tf.split(params, 8, axis=-1)

        invariants = compute_invariants(stretch_f, stretch_n, fiber_angle_rad, collagen_angle_rad, eps=EPS)
        Pf, Pn = compute_stress_terms(invariants, a, b, af, bf, ac, bc, afc, bfc, eps=EPS)
        return Pf + Pn, (a, af, ac, afc, b, bf, bc, bfc)
