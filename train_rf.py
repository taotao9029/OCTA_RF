import os
import re
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import joblib
except ModuleNotFoundError:
    import pickle

    class _JoblibFallback:
        @staticmethod
        def dump(obj, filename):
            with open(filename, "wb") as f:
                pickle.dump(obj, f)

        @staticmethod
        def load(filename):
            with open(filename, "rb") as f:
                return pickle.load(f)

    joblib = _JoblibFallback()

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


DATA_PATH = "./data/features_summary_renamed.csv"
SAVE_ROOT = "./output"
os.makedirs(SAVE_ROOT, exist_ok=True)

FEATURE_COLS = [
    "mean_speed",
    "std_speed",
    "mean_thd",
    "std_thd",
    "pulse_freq",
    "pulse_amp",
    "event_density",
]

USE_MISSING_INDICATORS = True
USE_NUM_SEGMENTS = False

SEED = 42
OUTER_FOLD = 5
INNER_FOLD = 5
N_ESTIMATORS = 500
CLASS_WEIGHT_CANDIDATES = [
    None,
    "balanced",
    {0: 1.5, 1: 1.0},
    {0: 2.0, 1: 1.0},
]
BOOTSTRAP_REPS = 2000
CALIBRATION_BINS = 10
THRESHOLD_OBJECTIVE = "balanced_acc"
THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)

EPOCH_METRIC_COLUMNS = [
    "model", "ablation", "outer_fold", "inner_fold", "seed", "epoch",
    "train_loss", "val_loss", "val_auc", "val_ap",
    "val_balanced_accuracy", "learning_rate", "checkpoint_saved",
]
RUN_MANIFEST_COLUMNS = [
    "run_id", "model", "ablation", "outer_fold", "seed", "config_file",
    "config_sha256", "code_commit", "start_time", "end_time",
    "selected_epoch", "selected_threshold", "threshold_objective",
    "checkpoint_path",
]
PREDICTION_COLUMNS = [
    "model", "ablation", "animal_id", "label", "outer_fold",
    "probability", "fold_threshold", "prediction", "threshold_objective",
    "checkpoint_id",
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_original_video_key(filename):
    """_aug_0、_aug_1 与原始视频归入同一视频级 cluster。"""
    filename = str(filename)
    return re.sub(r"_aug_\d+(\.csv)?$", "", filename).replace(".csv", "")


def save_split_manifest(video_meta, output_path):
    """保存 outer/inner 的固定视频级划分清单。"""
    rows = []
    label_map = dict(
        zip(video_meta["video_key"].astype(str), video_meta["label"].astype(int))
    )

    outer_cv = StratifiedGroupKFold(
        n_splits=OUTER_FOLD,
        shuffle=True,
        random_state=SEED,
    )

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(
            video_meta,
            video_meta["label"],
            groups=video_meta["video_key"],
        ),
        start=1,
    ):
        outer_train = video_meta.iloc[outer_train_idx].reset_index(drop=True)
        outer_test = video_meta.iloc[outer_test_idx].reset_index(drop=True)

        for key in outer_test["video_key"].astype(str):
            rows.append(
                {
                    "animal_id": key,
                    "video_key": key,
                    "label": label_map[key],
                    "outer_fold": outer_fold,
                    "role": "outer_test",
                    "inner_fold": "",
                    "split_seed": SEED,
                    "split_version": "sgkf_v1",
                }
            )

        n_inner = min(
            INNER_FOLD,
            int(outer_train["label"].value_counts().min()),
        )
        inner_cv = StratifiedGroupKFold(
            n_splits=n_inner,
            shuffle=True,
            random_state=SEED + outer_fold * 100,
        )

        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
            inner_cv.split(
                outer_train,
                outer_train["label"],
                groups=outer_train["video_key"],
            ),
            start=1,
        ):
            for role, indices in (
                ("inner_train", inner_train_idx),
                ("inner_val", inner_val_idx),
            ):
                for key in outer_train.iloc[indices]["video_key"].astype(str):
                    rows.append(
                        {
                            "animal_id": key,
                            "video_key": key,
                            "label": label_map[key],
                            "outer_fold": outer_fold,
                            "role": role,
                            "inner_fold": inner_fold,
                            "split_seed": SEED + outer_fold * 100,
                            "split_version": "sgkf_v1",
                        }
                    )

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


def save_run_manifest(summary, output_root):
    """保存每个 outer fold 的运行、阈值和模型文件记录。"""
    now = datetime.now().isoformat(timespec="seconds")
    required = pd.DataFrame({
        "run_id": [f"rf_ensemble_outer_fold_{int(fold)}" for fold in summary["fold"]],
        "model": "rf_ensemble",
        "ablation": "full",
        "outer_fold": summary["fold"].astype(int),
        "seed": SEED,
        "config_file": "",
        "config_sha256": "",
        "code_commit": "",
        "start_time": now,
        "end_time": now,
        "selected_epoch": np.nan,
        "selected_threshold": summary["threshold"],
        "threshold_objective": THRESHOLD_OBJECTIVE,
        "checkpoint_path": [
            os.path.abspath(os.path.join(output_root, path))
            for path in summary["model_bundle"]
        ],
    })
    extras = summary.drop(columns=["threshold"], errors="ignore").reset_index(drop=True)
    manifest = pd.concat([required, extras], axis=1)
    logs_dir = os.path.join(output_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    manifest[RUN_MANIFEST_COLUMNS + [
        col for col in manifest.columns if col not in RUN_MANIFEST_COLUMNS
    ]].to_csv(
        os.path.join(logs_dir, "run_manifest.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def save_epoch_metrics(rows, output_root):
    logs_dir = os.path.join(output_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    pd.DataFrame(rows, columns=EPOCH_METRIC_COLUMNS).to_csv(
        os.path.join(logs_dir, "epoch_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def validate_input(df):
    required = set(FEATURE_COLS + ["filename", "label"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少字段: {sorted(missing)}")
    if len(df) == 0:
        raise ValueError("输入数据为空")
    labels = set(df["label"].dropna().astype(int).unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"label 必须是 0/1，实际为: {labels}")
    label_check = df.groupby("video_key")["label"].nunique()
    if label_check.max() > 1:
        bad = label_check[label_check > 1].index.tolist()
        raise ValueError(f"同一视频对应多个标签: {bad[:10]}")
    print("特征缺失率:")
    print(df[FEATURE_COLS].isna().mean().to_string())


def fit_fill_values(df_train):
    values = {}
    for col in FEATURE_COLS:
        value = pd.to_numeric(df_train[col], errors="coerce")
        value = value.replace([np.inf, -np.inf], np.nan)
        median = value.median()
        values[col] = float(median) if np.isfinite(median) else 0.0
    return values


def safe_divide(a, b, eps=1e-3):
    denominator = np.where(
        np.abs(b) < eps,
        np.where(b >= 0, eps, -eps),
        b,
    )
    return np.clip(a / denominator, -1e4, 1e4)


def build_segment_features(df, fill_values):
    base = pd.DataFrame(index=df.index)
    for col in FEATURE_COLS:
        value = pd.to_numeric(df[col], errors="coerce")
        value = value.replace([np.inf, -np.inf], np.nan)
        if USE_MISSING_INDICATORS:
            base[f"{col}__missing"] = value.isna().astype(np.float32)
        base[col] = value.fillna(fill_values[col]).astype(np.float64)

    for col in ["event_density", "pulse_amp"]:
        base[col] = np.log1p(np.clip(base[col].to_numpy(), 0.0, None))

    speed = base["mean_speed"].to_numpy()
    speed_std = base["std_speed"].to_numpy()
    thd = base["mean_thd"].to_numpy()
    thd_std = base["std_thd"].to_numpy()
    freq = base["pulse_freq"].to_numpy()
    amp = base["pulse_amp"].to_numpy()
    density = base["event_density"].to_numpy()

    derived = pd.DataFrame(index=df.index)
    derived["d_amp_thd"] = amp * thd
    derived["d_density_speed"] = safe_divide(density, speed)
    derived["d_stdthd_thd"] = safe_divide(thd_std, thd)
    derived["d_freq_amp"] = freq * amp
    derived["d_amp_density"] = amp * density
    derived["d_freq_density"] = freq * density
    derived["d_thd_density"] = thd * density
    derived["d_speed_density"] = speed * density
    derived["d_amp_sq"] = np.clip(amp ** 2, -1e4, 1e4)
    derived["d_density_sq"] = np.clip(density ** 2, -1e4, 1e4)
    derived["d_speed_sq"] = np.clip(speed ** 2, -1e4, 1e4)
    derived["d_thd_sq"] = np.clip(thd ** 2, -1e4, 1e4)
    derived["d_stdspeed_speed"] = safe_divide(speed_std, speed)
    derived["d_freq_density_ratio"] = safe_divide(freq, density)
    derived["d_freq_sq"] = np.clip(freq ** 2, -1e4, 1e4)
    derived["d_amp_minus_thd"] = amp - thd
    derived["d_sqrt_freq_speedstd"] = np.sqrt(np.abs(freq * speed_std))

    return pd.concat([base, derived], axis=1).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0).clip(-1e4, 1e4).astype(np.float32)


def aggregate_video_features(segment_df, row_feature_cols):
    rows = []
    for video_key, group in segment_df.groupby("video_key", sort=True):
        values = group[row_feature_cols].to_numpy(dtype=np.float32)
        row = {"video_key": video_key, "label": int(group["label"].iloc[0])}
        if USE_NUM_SEGMENTS:
            row["num_segments"] = float(len(group))
        for idx, col in enumerate(row_feature_cols):
            value = values[:, idx]
            row[f"{col}__mean"] = float(np.mean(value))
            row[f"{col}__std"] = float(np.std(value))
            row[f"{col}__min"] = float(np.min(value))
            row[f"{col}__max"] = float(np.max(value))
            row[f"{col}__median"] = float(np.median(value))
            row[f"{col}__q25"] = float(np.percentile(value, 25))
            row[f"{col}__q75"] = float(np.percentile(value, 75))
        rows.append(row)
    return pd.DataFrame(rows).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)


def prepare_video_matrices(df_fit, df_eval):
    fill_values = fit_fill_values(df_fit)
    fit_row = build_segment_features(df_fit, fill_values).copy()
    eval_row = build_segment_features(df_eval, fill_values).copy()
    fit_row["video_key"] = df_fit["video_key"].to_numpy()
    fit_row["label"] = df_fit["label"].to_numpy()
    eval_row["video_key"] = df_eval["video_key"].to_numpy()
    eval_row["label"] = df_eval["label"].to_numpy()

    row_feature_cols = [
        col for col in fit_row.columns
        if col not in {"video_key", "label"}
    ]
    fit_video = aggregate_video_features(fit_row, row_feature_cols)
    eval_video = aggregate_video_features(eval_row, row_feature_cols)
    video_feature_cols = [
        col for col in fit_video.columns
        if col not in {"video_key", "label"}
    ]
    return {
        "X_fit": fit_video[video_feature_cols].to_numpy(np.float32),
        "y_fit": fit_video["label"].to_numpy(np.int64),
        "X_eval": eval_video[video_feature_cols].to_numpy(np.float32),
        "y_eval": eval_video["label"].to_numpy(np.int64),
        "fit_video": fit_video,
        "eval_video": eval_video,
        "fill_values": fill_values,
        "row_feature_cols": row_feature_cols,
        "video_feature_cols": video_feature_cols,
    }


def prepare_saved_eval_matrix(df_eval, metadata):
    """按已保存的训练期特征配置构造离线测试矩阵。"""
    eval_row = build_segment_features(
        df_eval,
        metadata["fill_values"],
    ).copy()
    eval_row["video_key"] = df_eval["video_key"].to_numpy()
    eval_row["label"] = df_eval["label"].to_numpy()
    eval_video = aggregate_video_features(
        eval_row,
        metadata["row_feature_cols"],
    )
    expected_cols = ["video_key", "label"] + list(metadata["video_feature_cols"])
    missing = set(expected_cols) - set(eval_video.columns)
    if missing:
        raise ValueError(f"离线特征缺少字段: {sorted(missing)}")
    eval_video = eval_video[expected_cols]
    return {
        "X_eval": eval_video[metadata["video_feature_cols"]].to_numpy(np.float32),
        "y_eval": eval_video["label"].to_numpy(np.int64),
        "eval_video": eval_video,
    }


def make_models(seed, class_weight):
    return [
        ExtraTreesClassifier(
            n_estimators=N_ESTIMATORS,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=-1,
        ),
        RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight=class_weight,
            random_state=seed + 1,
            n_jobs=-1,
        ),
        Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=0.3,
                class_weight=class_weight,
                max_iter=3000,
                random_state=seed,
            )),
        ]),
    ]


def fit_models(X, y, seed, class_weight):
    models = make_models(seed, class_weight)
    for model in models:
        model.fit(X, y)
    return models


def predict_ensemble(models, X):
    probs = [model.predict_proba(X)[:, 1] for model in models]
    return 0.45 * probs[0] + 0.35 * probs[1] + 0.20 * probs[2]


def load_video_ensemble_bundle(bundle_path):
    """加载某个 outer fold 的 video_ensemble_bundle.pkl。"""
    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, dict) or "models" not in bundle or "metadata" not in bundle:
        raise ValueError(f"无效的 ensemble bundle: {bundle_path}")
    return bundle["models"], bundle["metadata"]


def evaluate_saved_video_ensemble_bundle(bundle_path, data_path=DATA_PATH):
    """用保存的 RF/ExtraTrees/LR 集成模型重新测试对应 outer-test 视频。"""
    models, metadata = load_video_ensemble_bundle(bundle_path)
    if list(metadata["feature_cols"]) != list(FEATURE_COLS):
        raise ValueError("当前 FEATURE_COLS 与 bundle 中的特征配置不一致")
    saved_config = metadata.get("feature_config", {})
    if saved_config.get("use_missing_indicators", USE_MISSING_INDICATORS) != USE_MISSING_INDICATORS:
        raise ValueError("USE_MISSING_INDICATORS 与 bundle 不一致")
    if saved_config.get("use_num_segments", USE_NUM_SEGMENTS) != USE_NUM_SEGMENTS:
        raise ValueError("USE_NUM_SEGMENTS 与 bundle 不一致")
    df = pd.read_csv(data_path)
    df.columns = df.columns.astype(str).str.strip()
    df["filename"] = df["filename"].astype(str)
    df["video_key"] = df["filename"].apply(get_original_video_key)
    df["label"] = df["label"].astype(int)

    saved_test_keys = metadata.get("test_video_keys")
    if saved_test_keys is None:
        fold_prediction_path = os.path.join(
            os.path.dirname(bundle_path),
            "fold_test_vid_pred.csv",
        )
        if not os.path.exists(fold_prediction_path):
            raise ValueError(
                "bundle 缺少 test_video_keys，且同目录没有 fold_test_vid_pred.csv"
            )
        saved_test_keys = pd.read_csv(fold_prediction_path)["video_key"].tolist()
    test_keys = set(str(key) for key in saved_test_keys)
    df_test = df[df["video_key"].astype(str).isin(test_keys)].copy()
    if df_test.empty:
        raise ValueError(f"没有找到 bundle 对应的测试视频: {bundle_path}")

    data = prepare_saved_eval_matrix(df_test, metadata)
    prob = predict_ensemble(models, data["X_eval"])
    threshold = float(metadata["threshold"])
    pred = (prob >= threshold).astype(np.int64)
    metrics = calculate_metrics(data["y_eval"], pred, prob)
    predictions = data["eval_video"][["video_key", "label"]].copy()
    predictions["prob"] = prob
    predictions["pred"] = pred
    predictions["fold"] = int(metadata.get("fold", -1))
    predictions["fold_threshold"] = threshold
    return metrics, predictions


def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = (
        float(roc_auc_score(y_true, y_prob))
        if len(np.unique(y_true)) >= 2 else float("nan")
    )
    return {
        "ACC": float(accuracy_score(y_true, y_pred)),
        "AUC": auc,
        "PR-AUC": float(average_precision_score(y_true, y_prob)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Balanced_ACC": float(balanced_accuracy_score(y_true, y_pred)),
        "Sensitivity": float(tp / (tp + fn + 1e-8)),
        "Specificity": float(tn / (tn + fp + 1e-8)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def search_best_threshold(y_true, y_prob):
    """
    与 Fusion 模型保持一致：
    使用 inner-OOF 预测，以 Balanced Accuracy 为主要阈值选择目标。
    """
    best_threshold = 0.5
    best_score = None

    for threshold in THRESHOLD_GRID:
        pred = (y_prob >= threshold).astype(np.int64)
        metric = calculate_metrics(
            y_true=y_true,
            y_pred=pred,
            y_prob=y_prob,
        )

        if THRESHOLD_OBJECTIVE == "balanced_acc":
            objective = metric["Balanced_ACC"]
        elif THRESHOLD_OBJECTIVE == "f1":
            objective = metric["F1"]
        else:
            objective = metric["ACC"]

        score = (
            objective,
            metric["F1"],
            metric["Balanced_ACC"],
            metric["ACC"],
            -abs(float(threshold) - 0.5),
        )

        if best_score is None or score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold

def build_calibration_curve(df_video, n_bins=CALIBRATION_BINS):
    y_true = df_video["label"].to_numpy(dtype=np.int64)
    y_prob = np.clip(df_video["prob"].to_numpy(dtype=np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, edges[1:-1], right=False)
    rows = []
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            rows.append({
                "bin": bin_idx,
                "bin_lower": edges[bin_idx],
                "bin_upper": edges[bin_idx + 1],
                "n_videos": 0,
                "mean_predicted_prob": np.nan,
                "fraction_positive": np.nan,
                "absolute_gap": np.nan,
            })
            continue
        mean_prob = float(np.mean(y_prob[mask]))
        fraction_positive = float(np.mean(y_true[mask]))
        rows.append({
            "bin": bin_idx,
            "bin_lower": edges[bin_idx],
            "bin_upper": edges[bin_idx + 1],
            "n_videos": int(np.sum(mask)),
            "mean_predicted_prob": mean_prob,
            "fraction_positive": fraction_positive,
            "absolute_gap": abs(mean_prob - fraction_positive),
        })
    return pd.DataFrame(rows)


def calculate_calibration_metrics(df_video, n_bins=CALIBRATION_BINS):
    y_true = df_video["label"].to_numpy(dtype=np.int64)
    y_prob = np.clip(df_video["prob"].to_numpy(dtype=np.float64), 0.0, 1.0)
    curve = build_calibration_curve(df_video, n_bins=n_bins)
    counts = curve["n_videos"].to_numpy(dtype=np.float64)
    gaps = curve["absolute_gap"].fillna(0.0).to_numpy(dtype=np.float64)
    result = {
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "ECE": float(np.sum(counts * gaps) / max(float(np.sum(counts)), 1.0)),
        "MCE": float(np.max(gaps)) if len(gaps) else np.nan,
        "Calibration_Intercept": np.nan,
        "Calibration_Slope": np.nan,
        "n_videos": int(len(df_video)),
    }
    if len(np.unique(y_true)) < 2:
        return result
    clipped = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    try:
        calibration_model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        calibration_model.fit(logits, y_true)
        result["Calibration_Intercept"] = float(calibration_model.intercept_[0])
        result["Calibration_Slope"] = float(calibration_model.coef_[0, 0])
    except Exception:
        pass
    return result


def _bootstrap_metric_values(df_video):
    y_true = df_video["label"].to_numpy(dtype=np.int64)
    y_prob = df_video["prob"].to_numpy(dtype=np.float64)
    y_pred = df_video["prediction"].to_numpy(dtype=np.int64)
    metric = calculate_metrics(y_true, y_pred, y_prob)
    calibration = calculate_calibration_metrics(df_video)
    return {
        "ACC": metric["ACC"], "AUC": metric["AUC"],
        "PR-AUC": metric["PR-AUC"], "F1": metric["F1"],
        "Balanced_ACC": metric["Balanced_ACC"],
        "Sensitivity": metric["Sensitivity"],
        "Specificity": metric["Specificity"],
        **calibration,
    }


def bootstrap_cluster_ci(df_video, n_bootstrap=BOOTSTRAP_REPS, seed=SEED, alpha=0.05):
    if df_video["video_key"].duplicated().any():
        raise ValueError("bootstrap input must contain one row per video_key")
    if len(df_video) == 0:
        raise ValueError("bootstrap input is empty")
    rng = np.random.default_rng(seed)
    metric_names = [
        "ACC", "AUC", "PR-AUC", "F1", "Balanced_ACC", "Sensitivity",
        "Specificity", "Brier", "ECE", "MCE", "Calibration_Intercept",
        "Calibration_Slope",
    ]
    values = {name: [] for name in metric_names}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(df_video), size=len(df_video))
        sample = df_video.iloc[indices].copy().reset_index(drop=True)
        sample["video_key"] = np.arange(len(sample))
        result = _bootstrap_metric_values(sample)
        for name in metric_names:
            values[name].append(result.get(name, np.nan))
    estimate = _bootstrap_metric_values(df_video)
    lower_q, upper_q = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)
    rows = []
    for name in metric_names:
        valid = np.asarray(values[name], dtype=np.float64)
        valid = valid[np.isfinite(valid)]
        rows.append({
            "metric": name,
            "estimate": float(estimate.get(name, np.nan)),
            "ci_lower": float(np.percentile(valid, lower_q)) if len(valid) else np.nan,
            "ci_upper": float(np.percentile(valid, upper_q)) if len(valid) else np.nan,
            "n_valid_bootstrap": int(len(valid)),
            "n_bootstrap": int(n_bootstrap),
            "cluster_unit": "video_key (one video = one patient)",
        })
    return pd.DataFrame(rows)


def _save_curve_plot(df_video, output_root, model_name):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib 不可用，仅保存 ROC/PR 曲线 CSV")
        return

    y_true = df_video["label"].to_numpy(dtype=np.int64)
    y_prob = np.clip(df_video["prob"].to_numpy(dtype=np.float64), 0.0, 1.0)
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)
    prauc_value = average_precision_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_value:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Video-level ROC curve")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="lower right")

    axes[1].plot(recall, precision, linewidth=2, label=f"{model_name} (PR-AUC={prauc_value:.3f})")
    axes[1].axhline(float(np.mean(y_true)), linestyle="--", color="gray", linewidth=1)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Video-level precision-recall curve")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower left")

    fig.savefig(os.path.join(output_root, "roc_pr_curves.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_root, "roc_pr_curves.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_root, f"{model_name}_roc_pr_curves.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(output_root, f"{model_name}_roc_pr_curves.pdf"), bbox_inches="tight")
    plt.close(fig)


def save_test_statistics(df_video, output_root, model_name, predicted=None):
    """保存汇总指标、Bootstrap CI、校准、预测结果和 ROC/PR 曲线。"""
    os.makedirs(output_root, exist_ok=True)
    result = df_video.copy()
    if predicted is not None:
        result["prediction"] = np.asarray(predicted, dtype=np.int64)

    metrics = calculate_metrics(
        result["label"].to_numpy(dtype=np.int64),
        result["prediction"].to_numpy(dtype=np.int64),
        result["prob"].to_numpy(dtype=np.float64),
    )
    pd.DataFrame([metrics]).to_csv(
        os.path.join(output_root, "pooled_outer_test_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap = bootstrap_cluster_ci(result)
    bootstrap.to_csv(
        os.path.join(output_root, "pooled_outer_test_bootstrap_ci.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    calibration = calculate_calibration_metrics(result)
    pd.DataFrame([calibration]).to_csv(
        os.path.join(output_root, "pooled_outer_test_calibration.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    build_calibration_curve(result).to_csv(
        os.path.join(output_root, "pooled_outer_test_calibration_curve.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    y_true = result["label"].to_numpy(dtype=np.int64)
    y_prob = result["prob"].to_numpy(dtype=np.float64)
    if len(np.unique(y_true)) >= 2:
        fpr, tpr, roc_threshold = roc_curve(y_true, y_prob)
        precision, recall, pr_threshold = precision_recall_curve(y_true, y_prob)
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_threshold}).to_csv(
            os.path.join(output_root, "roc_curve.csv"), index=False, encoding="utf-8-sig"
        )
        pd.DataFrame({
            "recall": recall,
            "precision": precision,
            "threshold": np.append(pr_threshold, np.nan),
        }).to_csv(
            os.path.join(output_root, "pr_curve.csv"), index=False, encoding="utf-8-sig"
        )
        _save_curve_plot(result, output_root, model_name)

    result.to_csv(
        os.path.join(output_root, f"{model_name}_all_outer_test_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"{model_name} calibration: Brier={calibration['Brier']:.4f}, "
        f"ECE={calibration['ECE']:.4f}, MCE={calibration['MCE']:.4f}, "
        f"Intercept={calibration['Calibration_Intercept']:.4f}, "
        f"Slope={calibration['Calibration_Slope']:.4f}"
    )
    print(f"{model_name} bootstrap CI saved: {len(bootstrap)} metrics, {BOOTSTRAP_REPS} replicates")
    return metrics, bootstrap, calibration


def save_fold_mean_std(summary, output_root):
    metric_columns = {
        "ACC": "test_acc", "AUC": "test_auc", "PR-AUC": "test_prauc",
        "F1": "test_f1", "Balanced_ACC": "test_bal_acc",
        "Sensitivity": "test_sensitivity", "Specificity": "test_specificity",
    }
    rows = []
    for metric, column in metric_columns.items():
        values = summary[column].dropna().to_numpy(dtype=np.float64)
        rows.append({
            "metric": metric,
            "mean": float(np.mean(values)) if len(values) else np.nan,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            "mean_pm_std": (
                f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}"
                if len(values) > 1 else "NaN"
            ),
        })
    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(output_root, "outer_test_mean_std.csv"), index=False, encoding="utf-8-sig")
    result.to_csv(os.path.join(output_root, "paper_outer_test_mean_std_long.csv"), index=False, encoding="utf-8-sig")
    result.to_csv(os.path.join(output_root, "paper_table_outer_test_mean_std.csv"), index=False, encoding="utf-8-sig")
    try:
        result.to_latex(os.path.join(output_root, "paper_table_outer_test_mean_std.tex"), index=False, escape=True)
    except Exception:
        pass


def save_pooled_paper_table(metrics, bootstrap, calibration, output_root):
    """保存带 Bootstrap 95% CI 的论文汇总表。"""
    table = {}
    ci_lookup = bootstrap.set_index("metric")
    for metric in ["ACC", "AUC", "PR-AUC", "F1", "Sensitivity", "Specificity"]:
        row = ci_lookup.loc[metric]
        table[metric] = {
            "estimate": float(metrics[metric]),
            "95CI": f"{row['estimate']:.4f} ({row['ci_lower']:.4f}, {row['ci_upper']:.4f})",
        }
    for metric in ["Brier", "ECE", "MCE", "Calibration_Intercept", "Calibration_Slope"]:
        row = ci_lookup.loc[metric]
        table[metric] = {
            "estimate": float(calibration[metric]),
            "95CI": f"{row['estimate']:.4f} ({row['ci_lower']:.4f}, {row['ci_upper']:.4f})",
        }
    result = pd.DataFrame(table).T.reset_index().rename(columns={"index": "metric"})
    result.to_csv(os.path.join(output_root, "paper_table_pooled_outer_test_ci.csv"), index=False, encoding="utf-8-sig")
    try:
        result.to_latex(os.path.join(output_root, "paper_table_pooled_outer_test_ci.tex"), index=False, escape=True)
    except Exception:
        pass


def make_inner_oof_predictions(df_outer_train, train_meta, fold_idx, class_weight):
    min_count = int(train_meta["label"].value_counts().min())
    n_splits = min(INNER_FOLD, min_count)
    if n_splits < 2:
        raise ValueError("内部 OOF 至少需要每类 2 个视频")

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED + fold_idx * 100,
    )
    keys = train_meta["video_key"].to_numpy()
    labels = train_meta["label"].to_numpy()
    key_to_index = {key: idx for idx, key in enumerate(keys)}
    oof_prob = np.full(len(keys), np.nan, dtype=np.float64)

    for inner_idx, (fit_idx, val_idx) in enumerate(
        cv.split(train_meta, labels, groups=train_meta["video_key"]), start=1
    ):
        fit_keys = set(train_meta.iloc[fit_idx]["video_key"])
        val_keys = set(train_meta.iloc[val_idx]["video_key"])
        df_fit = df_outer_train[df_outer_train["video_key"].isin(fit_keys)].copy()
        df_val = df_outer_train[df_outer_train["video_key"].isin(val_keys)].copy()
        data = prepare_video_matrices(df_fit, df_val)
        models = fit_models(data["X_fit"], data["y_fit"], SEED + fold_idx * 100 + inner_idx, class_weight)
        val_prob = predict_ensemble(models, data["X_eval"])
        for key, prob in zip(data["eval_video"]["video_key"].to_numpy(), val_prob):
            oof_prob[key_to_index[key]] = prob

    if np.any(~np.isfinite(oof_prob)):
        raise RuntimeError("内部 OOF 预测没有覆盖所有训练视频")
    return labels, oof_prob


def select_class_weight_and_threshold(
    df_outer_train,
    train_meta,
    fold_idx,
):
    """
    在 outer-train 内部使用 inner-OOF 同时选择 class_weight 和阈值。
    阈值及候选模型均以 Balanced Accuracy 为主要目标。
    """
    candidates = []

    for candidate_idx, class_weight in enumerate(
        CLASS_WEIGHT_CANDIDATES
    ):
        y_oof, prob_oof = make_inner_oof_predictions(
            df_outer_train,
            train_meta,
            fold_idx,
            class_weight,
        )

        threshold = search_best_threshold(
            y_true=y_oof,
            y_prob=prob_oof,
        )

        pred = (prob_oof >= threshold).astype(np.int64)

        metric = calculate_metrics(
            y_true=y_oof,
            y_pred=pred,
            y_prob=prob_oof,
        )

        candidates.append(
            {
                "candidate_idx": candidate_idx,
                "class_weight": class_weight,
                "threshold": threshold,
                "metric": metric,
                "y_oof": y_oof,
                "prob_oof": prob_oof,
            }
        )

        print(
            f"  class_weight={class_weight}, "
            f"OOF BalAcc={metric['Balanced_ACC']:.4f}, "
            f"OOF F1={metric['F1']:.4f}, "
            f"OOF ACC={metric['ACC']:.4f}, "
            f"OOF AUC={metric['AUC']:.4f}, "
            f"threshold={threshold:.4f}"
        )

    best = max(
        candidates,
        key=lambda item: (
            item["metric"]["Balanced_ACC"],
            item["metric"]["F1"],
            item["metric"]["ACC"],
            (
                item["metric"]["AUC"]
                if np.isfinite(item["metric"]["AUC"])
                else -1.0
            ),
        ),
    )
    best["candidate_metrics"] = [
        {
            "class_weight": str(item["class_weight"]),
            "threshold": item["threshold"],
            "oof_balanced_accuracy": item["metric"]["Balanced_ACC"],
            "oof_f1": item["metric"]["F1"],
            "oof_accuracy": item["metric"]["ACC"],
            "oof_auc": item["metric"]["AUC"],
            "threshold_objective": THRESHOLD_OBJECTIVE,
        }
        for item in candidates
    ]
    return best

def main():
    seed_everything(SEED)
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.astype(str).str.strip()
    df["filename"] = df["filename"].astype(str)
    df["video_key"] = df["filename"].apply(get_original_video_key)
    df["label"] = df["label"].astype(int)
    validate_input(df)

    video_meta = (
        df[["video_key", "label"]]
        .drop_duplicates()
        .sort_values("video_key")
        .reset_index(drop=True)
    )
    min_count = int(video_meta["label"].value_counts().min())
    if min_count < OUTER_FOLD:
        raise ValueError(f"最少类别只有 {min_count} 个视频，无法进行 {OUTER_FOLD} 折 CV")
    save_split_manifest(
        video_meta,
        os.path.join(SAVE_ROOT, "split_manifest.csv"),
    )
    print(
        f"片段数={len(df)}, 视频数={len(video_meta)}, "
        f"类别分布={video_meta['label'].value_counts().to_dict()}"
    )

    outer_cv = StratifiedGroupKFold(
        n_splits=OUTER_FOLD,
        shuffle=True,
        random_state=SEED,
    )
    summary_rows, all_test, fold_thresholds = [], [], []
    epoch_metric_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        outer_cv.split(video_meta, video_meta["label"], groups=video_meta["video_key"]),
        start=1,
    ):
        fold_dir = os.path.join(SAVE_ROOT, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        train_meta = video_meta.iloc[train_idx].reset_index(drop=True)
        test_meta = video_meta.iloc[test_idx].reset_index(drop=True)
        train_keys, test_keys = set(train_meta["video_key"]), set(test_meta["video_key"])
        df_train = df[df["video_key"].isin(train_keys)].copy()
        df_test = df[df["video_key"].isin(test_keys)].copy()

        print(f"\n================ Fold {fold_idx}/{OUTER_FOLD} ================")
        print("选择 class_weight 和 OOF threshold...")
        best = select_class_weight_and_threshold(df_train, train_meta, fold_idx)
        class_weight = best["class_weight"]
        threshold = best["threshold"]
        fold_thresholds.append(threshold)
        pd.DataFrame(best["candidate_metrics"]).to_csv(
            os.path.join(fold_dir, "threshold_selection_candidates.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        inner_oof_frame = pd.DataFrame(
            {
                "model": "rf_ensemble",
                "ablation": "full",
                "animal_id": train_meta["video_key"].astype(str),
                "video_key": train_meta["video_key"].astype(str),
                "label": best["y_oof"],
                "probability": best["prob_oof"],
                "prediction": (
                    best["prob_oof"] >= threshold
                ).astype(np.int64),
                "prob": best["prob_oof"],
                "pred": (
                    best["prob_oof"] >= threshold
                ).astype(np.int64),
                "outer_fold": fold_idx,
                "fold_threshold": threshold,
                "threshold_objective": THRESHOLD_OBJECTIVE,
            }
        )
        inner_oof_frame.to_csv(
            os.path.join(fold_dir, "inner_oof_video_pred.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        data = prepare_video_matrices(df_train, df_test)
        models = fit_models(data["X_fit"], data["y_fit"], SEED + 1000 + fold_idx, class_weight)
        test_prob = predict_ensemble(models, data["X_eval"])
        test_pred = (test_prob >= threshold).astype(np.int64)
        test_metric = calculate_metrics(data["y_eval"], test_pred, test_prob)

        metadata = {
            "feature_cols": FEATURE_COLS,
            "row_feature_cols": data["row_feature_cols"],
            "video_feature_cols": data["video_feature_cols"],
            "fill_values": data["fill_values"],
            "class_weight": class_weight,
            "threshold": threshold,
            "threshold_objective": THRESHOLD_OBJECTIVE,
            "threshold_grid": {
                "min": float(THRESHOLD_GRID.min()),
                "max": float(THRESHOLD_GRID.max()),
                "step": 0.01,
            },
            "seed": int(SEED + 1000 + fold_idx),
            "fold": int(fold_idx),
            "test_video_keys": [str(key) for key in test_meta["video_key"]],
            "feature_config": {
                "use_missing_indicators": bool(USE_MISSING_INDICATORS),
                "use_num_segments": bool(USE_NUM_SEGMENTS),
            },
        }
        joblib.dump(
            {"models": models, "metadata": metadata},
            os.path.join(fold_dir, "video_ensemble_bundle.pkl"),
        )
        epoch_metric_rows.append({
            "model": "rf_ensemble",
            "ablation": "full",
            "outer_fold": fold_idx,
            "inner_fold": 0,
            "seed": SEED + 1000 + fold_idx,
            "epoch": 0,
            "train_loss": np.nan,
            "val_loss": np.nan,
            "val_auc": best["metric"]["AUC"],
            "val_ap": best["metric"]["PR-AUC"],
            "val_balanced_accuracy": best["metric"]["Balanced_ACC"],
            "learning_rate": np.nan,
            "checkpoint_saved": 1,
        })

        test_video = data["eval_video"][["video_key", "label"]].copy()
        test_video["model"] = "rf_ensemble"
        test_video["ablation"] = "full"
        test_video["animal_id"] = test_video["video_key"].astype(str)
        test_video["prob"] = test_prob
        test_video["pred"] = test_pred
        test_video["probability"] = test_prob
        test_video["prediction"] = test_pred
        test_video["fold"] = fold_idx
        test_video["outer_fold"] = fold_idx
        test_video["fold_threshold"] = threshold
        test_video["threshold_objective"] = THRESHOLD_OBJECTIVE
        test_video["checkpoint_id"] = os.path.relpath(
            os.path.join(fold_dir, "video_ensemble_bundle.pkl"),
            SAVE_ROOT,
        )
        test_video.to_csv(os.path.join(fold_dir, "fold_test_vid_pred.csv"), index=False, encoding="utf-8-sig")
        all_test.append(test_video)

        print(f"selected weight={class_weight}, threshold={threshold:.4f}")
        print(
            f"Test ACC={test_metric['ACC']:.4f}, AUC={test_metric['AUC']:.4f}, "
            f"PR-AUC={test_metric['PR-AUC']:.4f}, Spec={test_metric['Specificity']:.4f}, "
            f"Sens={test_metric['Sensitivity']:.4f}"
        )
        summary_rows.append({
            "fold": fold_idx,
            "class_weight": str(class_weight),
            "threshold": threshold,
            "test_acc": test_metric["ACC"],
            "test_auc": test_metric["AUC"],
            "test_prauc": test_metric["PR-AUC"],
            "test_f1": test_metric["F1"],
            "test_bal_acc": test_metric["Balanced_ACC"],
            "test_specificity": test_metric["Specificity"],
            "test_sensitivity": test_metric["Sensitivity"],
            "model_bundle": os.path.relpath(
                os.path.join(fold_dir, "video_ensemble_bundle.pkl"),
                SAVE_ROOT,
            ),
        })

    summary = pd.DataFrame(summary_rows)
    save_fold_mean_std(summary, SAVE_ROOT)
    summary.to_csv(os.path.join(SAVE_ROOT, "outer_5fold_summary.csv"), index=False, encoding="utf-8-sig")
    all_test = pd.concat(all_test, ignore_index=True)
    all_test.to_csv(os.path.join(SAVE_ROOT, "all_outer_test_vid_pred.csv"), index=False, encoding="utf-8-sig")
    all_test[PREDICTION_COLUMNS].to_csv(
        os.path.join(SAVE_ROOT, "standardized_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    save_run_manifest(summary, SAVE_ROOT)
    save_epoch_metrics(epoch_metric_rows, SAVE_ROOT)

    # 直接使用每个 outer fold 根据 inner-OOF 阈值生成的 prediction。
    # 不再使用全局中位数阈值覆盖各 fold 的 outer-test prediction。
    pooled_pred = all_test["prediction"].to_numpy(dtype=np.int64)

    pooled_metric = calculate_metrics(
        y_true=all_test["label"].to_numpy(dtype=np.int64),
        y_pred=pooled_pred,
        y_prob=all_test["prob"].to_numpy(dtype=np.float64),
    )

    pooled_metrics, pooled_bootstrap, pooled_calibration = save_test_statistics(
        all_test,
        SAVE_ROOT,
        model_name="ensemble",
        predicted=pooled_pred,
    )
    save_pooled_paper_table(
        pooled_metrics,
        pooled_bootstrap,
        pooled_calibration,
        SAVE_ROOT,
    )

    print("\n================ OOF 5-fold Summary ================")
    print(summary.to_string(index=False))
    print("\n================ Pooled OOF Test ================")
    print(
        f"threshold=per-fold inner-OOF balanced-accuracy, ACC={pooled_metric['ACC']:.4f}, "
        f"AUC={pooled_metric['AUC']:.4f}, PR-AUC={pooled_metric['PR-AUC']:.4f}, "
        f"F1={pooled_metric['F1']:.4f}, BalAcc={pooled_metric['Balanced_ACC']:.4f}, "
        f"Spec={pooled_metric['Specificity']:.4f}, Sens={pooled_metric['Sensitivity']:.4f}"
    )
    print("\n================ Mean ± Std ================")
    for col in [
        "test_acc", "test_auc", "test_prauc", "test_f1", "test_bal_acc",
        "test_sensitivity", "test_specificity",
    ]:
        print(f"{col:12s}: {summary[col].mean():.4f} ± {summary[col].std():.4f}")


if __name__ == "__main__":
    main()
