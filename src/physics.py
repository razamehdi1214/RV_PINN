import tensorflow as tf


def safe_exp(x):
    return tf.exp(tf.clip_by_value(x, -50.0, 50.0))


def compute_invariants(stretch_f, stretch_n, fiber_angle_rad, collagen_angle_rad, eps=1e-6):
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

    return {
        "stretch_f": stretch_f, "stretch_n": stretch_n, "stretch_s": stretch_s,
        "f1": f1, "f2": f2, "c1": c1, "c2": c2,
        "I1": I1, "I4f": I4f, "I4c": I4c, "I8fc": I8fc,
    }


def compute_stress_terms(inv, a, b, af, bf, ac, bc, afc, bfc, eps=1e-6):
    stretch_f = inv["stretch_f"]
    stretch_n = inv["stretch_n"]
    f1, f2, c1, c2 = inv["f1"], inv["f2"], inv["c1"], inv["c2"]
    I1, I4f, I4c, I8fc = inv["I1"], inv["I4f"], inv["I4c"], inv["I8fc"]

    with tf.GradientTape(persistent=True) as tape:
        tape.watch([stretch_f, stretch_n])
        psi = (
            a * (safe_exp(b * (I1 - 3.0)) - 1.0) +
            af * (safe_exp(bf * tf.square(I4f - 1.0)) - 1.0) +
            ac * (safe_exp(bc * tf.square(I4c - 1.0)) - 1.0) +
            afc * (safe_exp(bfc * tf.square(I8fc)) - 1.0)
        )

    dPsi_dI1 = tape.gradient(psi, I1)
    dPsi_dI4f = tape.gradient(psi, I4f)
    dPsi_dI4c = tape.gradient(psi, I4c)
    dPsi_dI8fc = tape.gradient(psi, I8fc)
    del tape

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

    Pf = dPsi_dI1 * dI1_dlf + dPsi_dI4f * dI4f_dlf + dPsi_dI4c * dI4c_dlf + dPsi_dI8fc * dI8fc_dlf + p * dps_ds_dlf
    Pn = dPsi_dI1 * dI1_dln + dPsi_dI4f * dI4f_dln + dPsi_dI4c * dI4c_dln + dPsi_dI8fc * dI8fc_dln + p * dps_ds_dln
    return Pf, Pn
