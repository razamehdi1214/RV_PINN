import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from .config import RANDOM_STATE, VAL_SIZE


def load_excel_data(file_path):
    xls = pd.ExcelFile(file_path)
    df_meta = pd.read_excel(xls, sheet_name="ML_final_data", header=None)
    circ_df = pd.read_excel(xls, sheet_name="circum_stress", header=None)
    long_df = pd.read_excel(xls, sheet_name="longit_stress", header=None)
    return xls, df_meta, circ_df, long_df


def extract_hemodynamic_features(df_meta):
    hemo_feature_names = df_meta.iloc[0, 4:-1].tolist()
    hemo_features = df_meta.iloc[1:, 4:-1].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    fiber_deg = hemo_features.iloc[:, 8].astype(np.float32)
    collagen_deg = hemo_features.iloc[:, 9].astype(np.float32)
    fiber_rad = fiber_deg * np.pi / 180.0
    collagen_rad = collagen_deg * np.pi / 180.0
    hemo_ids = df_meta.iloc[1:, 0].apply(lambda x: str(int(float(x)))).tolist()
    return hemo_feature_names, hemo_features, fiber_rad, collagen_rad, hemo_ids


def extract_all_samples(circ_df, long_df, hemo_features, fiber_rad, collagen_rad, hemo_ids):
    circ_data = circ_df.iloc[1:].reset_index(drop=True)
    long_data = long_df.iloc[1:].reset_index(drop=True)
    specimen_names = circ_df.iloc[0].astype(str).tolist()
    specimen_to_hemo_row = {str(int(float(hemo_ids[i]))): i for i in range(len(hemo_ids))}

    Xf, Xn, H, Pf, Pn, F, C, specimen_ids = [], [], [], [], [], [], [], []
    for idx in range(circ_data.shape[1]):
        if pd.isna(circ_data.iloc[:, idx]).all():
            continue
        specimen_name = str(int(float(specimen_names[idx]))) if specimen_names[idx] != "species" else None
        hemo_idx = specimen_to_hemo_row.get(specimen_name)
        if hemo_idx is None or hemo_idx >= len(hemo_features):
            continue
        hemo_vec = hemo_features.iloc[hemo_idx].values.astype(np.float32)
        pf = circ_data.iloc[:, idx].dropna().astype(np.float32).values
        pn = long_data.iloc[:, idx].dropna().astype(np.float32).values
        lam = np.linspace(1.0, 1.3, len(pf)).astype(np.float32)

        Xf.append(lam)
        Xn.append(lam)
        H.append(hemo_vec)
        Pf.append(pf)
        Pn.append(pn)
        F.append(fiber_rad.iloc[hemo_idx])
        C.append(collagen_rad.iloc[hemo_idx])
        specimen_ids.append(specimen_name)

    return {
        "Xf_all": np.array(Xf),
        "Xn_all": np.array(Xn),
        "H_all": np.array(H),
        "Pf_all": np.array(Pf),
        "Pn_all": np.array(Pn),
        "fiber_angles_all": np.array(F),
        "collagen_angles_all": np.array(C),
        "specimen_ids_all": np.array(specimen_ids),
    }


def load_all_samples(file_path):
    _, df_meta, circ_df, long_df = load_excel_data(file_path)
    _, hemo_features, fiber_rad, collagen_rad, hemo_ids = extract_hemodynamic_features(df_meta)
    data = extract_all_samples(circ_df, long_df, hemo_features, fiber_rad, collagen_rad, hemo_ids)
    data["Ptotal_all"] = data["Pf_all"] + data["Pn_all"]
    return data


def make_train_val_test_split(data, test_index=0):
    split = {"test_index": test_index}
    for key in ["Xf_all", "Xn_all", "H_all", "Pf_all", "Pn_all", "fiber_angles_all", "collagen_angles_all", "specimen_ids_all"]:
        split[key] = data[key]

    split["Xf_test"] = data["Xf_all"][test_index:test_index+1]
    split["Xn_test"] = data["Xn_all"][test_index:test_index+1]
    split["H_test"] = data["H_all"][test_index:test_index+1]
    split["Pf_test"] = data["Pf_all"][test_index:test_index+1]
    split["Pn_test"] = data["Pn_all"][test_index:test_index+1]
    split["F_test"] = data["fiber_angles_all"][test_index:test_index+1]
    split["C_test"] = data["collagen_angles_all"][test_index:test_index+1]
    split["ID_test"] = data["specimen_ids_all"][test_index]

    Xf_train = np.delete(data["Xf_all"], test_index, axis=0)
    Xn_train = np.delete(data["Xn_all"], test_index, axis=0)
    H_train = np.delete(data["H_all"], test_index, axis=0)
    Pf_train = np.delete(data["Pf_all"], test_index, axis=0)
    Pn_train = np.delete(data["Pn_all"], test_index, axis=0)
    F_train = np.delete(data["fiber_angles_all"], test_index, axis=0)
    C_train = np.delete(data["collagen_angles_all"], test_index, axis=0)
    ID_train = np.delete(data["specimen_ids_all"], test_index, axis=0)

    (split["Xf_train_main"], split["Xf_val"],
     split["Xn_train_main"], split["Xn_val"],
     split["H_train_main"], split["H_val"],
     split["Pf_train_main"], split["Pf_val"],
     split["Pn_train_main"], split["Pn_val"],
     split["F_train_main"], split["F_val"],
     split["C_train_main"], split["C_val"],
     split["ID_train_main"], split["ID_val"]) = train_test_split(
        Xf_train, Xn_train, H_train, Pf_train, Pn_train, F_train, C_train, ID_train,
        test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    split["H_train"] = H_train
    split["Ptotal_train_main"] = split["Pf_train_main"] + split["Pn_train_main"]
    split["Ptotal_val"] = split["Pf_val"] + split["Pn_val"]
    return split


def make_tf_datasets(split):
    train_ds = tf.data.Dataset.from_tensor_slices((
        split["Xf_train_main"], split["Xn_train_main"], split["H_train_main"],
        split["F_train_main"], split["C_train_main"], split["Ptotal_train_main"]
    )).batch(split["Xf_train_main"].shape[0])

    val_ds = tf.data.Dataset.from_tensor_slices((
        split["Xf_val"], split["Xn_val"], split["H_val"],
        split["F_val"], split["C_val"], split["Ptotal_val"]
    )).batch(split["Xf_val"].shape[0])
    return train_ds, val_ds
