import hashlib
import json
import os
import random
import re
import shutil
import subprocess
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


DATA_PATH = "./feature_out/features_summary_renamed.csv"
SPLIT_MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "feature_out",
    "split_manifest.csv",
)
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
ABLATION_MODES = [
    "full",
    "wo_mean_speed",
    "wo_std_speed",
    "wo_mean_thd",
    "wo_std_thd",
    "wo_pulse_freq",
    "wo_pulse_amp",
    "wo_event_density",
    "wo_missing",
    "wo_derived",
    "base_only",
]
DERIVED_FEATURE_DEPENDENCIES = {
    "mean_speed": {
        "d_density_speed",
        "d_speed_density",
        "d_speed_sq",
        "d_stdspeed_speed",
    },
    "std_speed": {
        "d_stdspeed_speed",
        "d_sqrt_freq_speedstd",
    },
    "mean_thd": {
        "d_amp_thd",
        "d_stdthd_thd",
        "d_thd_density",
        "d_thd_sq",
        "d_amp_minus_thd",
    },
    "std_thd": {
        "d_stdthd_thd",
    },
    "pulse_freq": {
        "d_freq_amp",
        "d_freq_density",
        "d_freq_density_ratio",
        "d_freq_sq",
        "d_sqrt_freq_speedstd",
    },
    "pulse_amp": {
        "d_amp_thd",
        "d_freq_amp",
        "d_amp_density",
        "d_amp_sq",
        "d_amp_minus_thd",
    },
    "event_density": {
        "d_density_speed",
        "d_amp_density",
        "d_freq_density",
        "d_thd_density",
        "d_speed_density",
        "d_density_sq",
        "d_freq_density_ratio",
    },
}
BOOTSTRAP_REPS = 2000
CALIBRATION_BINS = 10
THRESHOLD_OBJECTIVE = "balanced_acc"
THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)
CONFIG_SNAPSHOT_PATH = ""
CONFIG_SHA256 = ""
CODE_COMMIT = ""

EPOCH_METRIC_COLUMNS = [
    "model", "ablation", "outer_fold", "inner_fold", "seed", "epoch",
    "train_loss", "val_loss", "val_auc", "val_ap",
    "val_balanced_accuracy", "learning_rate", "checkpoint_saved",
]
RUN_MANIFEST_COLUMNS = [
    "run_id", "model", "ablation", "outer_fold", "seed", "config_file",
    "config_sha256", "training_manifest_sha256", "code_commit",
    "start_time", "end_time", "selected_epoch", "selected_threshold",
    "threshold_objective", "checkpoint_path", "relative_path", "bytes",
    "sha256",
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


def sha256_file(file_path):
    """计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with open(file_path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit():
    """获取当前代码所在 Git 仓库的完整 commit。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )
        commit = result.stdout.strip()
        return commit if commit else "not_available"
    except (OSError, subprocess.SubprocessError):
        return "not_available"


def build_config_snapshot(ablation_mode=None):
    """收集本次训练实际使用的配置。"""
    config_names = [
        "DATA_PATH", "SPLIT_MANIFEST_PATH", "SAVE_ROOT", "FEATURE_COLS",
        "USE_MISSING_INDICATORS", "USE_NUM_SEGMENTS", "SEED",
        "OUTER_FOLD", "INNER_FOLD", "N_ESTIMATORS",
        "CLASS_WEIGHT_CANDIDATES", "BOOTSTRAP_REPS", "CALIBRATION_BINS",
        "THRESHOLD_OBJECTIVE", "THRESHOLD_GRID", "ABLATION_MODES",
    ]
    config = {"source_file": os.path.abspath(__file__)}
    for name in config_names:
        value = globals()[name]
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, tuple):
            value = list(value)
        config[name] = value
    if ablation_mode is not None:
        if ablation_mode not in ABLATION_MODES:
            raise ValueError(f"未知消融模式: {ablation_mode}")
        config["ablation_mode"] = ablation_mode
    return config


def prepare_run_metadata(output_root=None, ablation_mode=None):
    """生成配置快照、配置 SHA-256 和代码 commit。"""
    global CONFIG_SNAPSHOT_PATH, CONFIG_SHA256, CODE_COMMIT

    if output_root is None:
        output_root = SAVE_ROOT
    os.makedirs(output_root, exist_ok=True)
    config_path = os.path.abspath(
        os.path.join(output_root, "config_snapshot.json")
    )
    with open(config_path, "w", encoding="utf-8") as file_obj:
        json.dump(
            build_config_snapshot(ablation_mode),
            file_obj,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        file_obj.write("\n")

    CONFIG_SNAPSHOT_PATH = config_path
    CONFIG_SHA256 = sha256_file(config_path)
    CODE_COMMIT = get_git_commit()


def load_split_manifest(video_meta, manifest_path):
    """读取并校验固定的 outer/inner 五折视频划分。"""
    manifest_path = os.path.abspath(manifest_path)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"训练划分清单不存在：{manifest_path}")

    manifest = pd.read_csv(
        manifest_path,
        dtype={"animal_id": str, "video_key": str},
    )
    manifest.columns = manifest.columns.astype(str).str.strip()
    required = {"video_key", "label", "outer_fold", "role", "inner_fold"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"训练划分清单缺少字段：{sorted(missing)}")

    manifest["video_key"] = manifest["video_key"].astype(str).str.strip()
    manifest["role"] = manifest["role"].astype(str).str.strip()
    manifest["label"] = pd.to_numeric(
        manifest["label"], errors="raise"
    ).astype(int)
    manifest["outer_fold"] = pd.to_numeric(
        manifest["outer_fold"], errors="raise"
    ).astype(int)
    manifest["inner_fold"] = pd.to_numeric(
        manifest["inner_fold"], errors="coerce"
    )

    allowed_roles = {"outer_test", "inner_train", "inner_val"}
    unknown_roles = sorted(set(manifest["role"]) - allowed_roles)
    if unknown_roles:
        raise ValueError(f"训练划分清单包含未知 role：{unknown_roles}")
    if manifest.loc[
        manifest["role"].isin({"inner_train", "inner_val"}),
        "inner_fold",
    ].isna().any():
        raise ValueError("inner_train/inner_val 记录缺少 inner_fold")
    if manifest.duplicated(
        ["video_key", "outer_fold", "role", "inner_fold"]
    ).any():
        raise ValueError("训练划分清单包含重复记录")

    video_keys = video_meta["video_key"].astype(str)
    dataset_keys = set(video_keys)
    manifest_keys = set(manifest["video_key"])
    if dataset_keys != manifest_keys:
        missing_keys = sorted(dataset_keys - manifest_keys)
        extra_keys = sorted(manifest_keys - dataset_keys)
        raise ValueError(
            "训练数据与划分清单的视频不一致："
            f"missing={missing_keys[:10]}, extra={extra_keys[:10]}"
        )

    label_map = dict(zip(video_keys, video_meta["label"].astype(int)))
    expected_labels = manifest["video_key"].map(label_map)
    if not manifest["label"].eq(expected_labels).all():
        bad_keys = manifest.loc[
            ~manifest["label"].eq(expected_labels), "video_key"
        ].drop_duplicates().tolist()
        raise ValueError(f"训练数据与划分清单标签不一致：{bad_keys[:10]}")

    expected_outer_folds = list(range(1, OUTER_FOLD + 1))
    actual_outer_folds = sorted(manifest["outer_fold"].unique().tolist())
    if actual_outer_folds != expected_outer_folds:
        raise ValueError(
            f"outer fold 应为 {expected_outer_folds}，实际为 {actual_outer_folds}"
        )

    outer_test_rows = manifest[manifest["role"].eq("outer_test")]
    outer_test_counts = outer_test_rows["video_key"].value_counts()
    if (
        set(outer_test_counts.index) != dataset_keys
        or not outer_test_counts.eq(1).all()
    ):
        raise ValueError("每个视频必须且只能出现于一个 outer_test fold")

    index_by_key = {key: index for index, key in enumerate(video_keys)}
    outer_splits = []
    for outer_fold in expected_outer_folds:
        fold_manifest = manifest[manifest["outer_fold"].eq(outer_fold)]
        test_keys = set(
            fold_manifest.loc[
                fold_manifest["role"].eq("outer_test"), "video_key"
            ]
        )
        train_keys = dataset_keys - test_keys
        inner_manifest = fold_manifest[
            fold_manifest["role"].isin({"inner_train", "inner_val"})
        ]
        if set(inner_manifest["video_key"]) != train_keys:
            raise ValueError(f"outer fold {outer_fold} 的训练视频集合不完整")

        expected_inner_folds = list(range(1, INNER_FOLD + 1))
        actual_inner_folds = sorted(
            inner_manifest["inner_fold"].astype(int).unique().tolist()
        )
        if actual_inner_folds != expected_inner_folds:
            raise ValueError(
                f"outer fold {outer_fold} 的 inner fold 应为 "
                f"{expected_inner_folds}，实际为 {actual_inner_folds}"
            )

        for inner_fold in expected_inner_folds:
            current = inner_manifest[
                inner_manifest["inner_fold"].eq(inner_fold)
            ]
            inner_train_keys = set(
                current.loc[current["role"].eq("inner_train"), "video_key"]
            )
            inner_val_keys = set(
                current.loc[current["role"].eq("inner_val"), "video_key"]
            )
            if inner_train_keys & inner_val_keys:
                raise ValueError(
                    f"outer fold {outer_fold} / inner fold {inner_fold} 存在重叠"
                )
            if inner_train_keys | inner_val_keys != train_keys:
                raise ValueError(
                    f"outer fold {outer_fold} / inner fold {inner_fold} "
                    "未完整覆盖 outer train"
                )

        inner_val_counts = inner_manifest.loc[
            inner_manifest["role"].eq("inner_val"), "video_key"
        ].value_counts()
        if (
            set(inner_val_counts.index) != train_keys
            or not inner_val_counts.eq(1).all()
        ):
            raise ValueError(
                f"outer fold {outer_fold} 的每个训练视频必须且只能验证一次"
            )

        train_idx = np.asarray(
            [index_by_key[key] for key in video_keys if key in train_keys],
            dtype=np.int64,
        )
        test_idx = np.asarray(
            [index_by_key[key] for key in video_keys if key in test_keys],
            dtype=np.int64,
        )
        outer_splits.append((train_idx, test_idx))

    return manifest, outer_splits


def save_run_manifest(summary, output_root, ablation_mode):
    """保存正式运行清单和模型文件哈希。"""
    global CONFIG_SNAPSHOT_PATH, CONFIG_SHA256, CODE_COMMIT

    if not CONFIG_SNAPSHOT_PATH:
        prepare_run_metadata(output_root, ablation_mode)

    now = datetime.now().isoformat(timespec="seconds")
    model_paths = []
    for relative_model_path in summary["model_bundle"].astype(str):
        model_path = os.path.abspath(
            os.path.join(output_root, relative_model_path)
        )
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"模型文件不存在：{model_path}")
        model_paths.append(model_path)

    required = pd.DataFrame({
        "run_id": [
            f"rf_ensemble_{ablation_mode}_outer_fold_{int(fold)}"
            for fold in summary["fold"]
        ],
        "model": "rf_ensemble",
        "ablation": ablation_mode,
        "outer_fold": summary["fold"].astype(int),
        "seed": SEED,
        "config_file": CONFIG_SNAPSHOT_PATH,
        "config_sha256": CONFIG_SHA256,
        "training_manifest_sha256": sha256_file(SPLIT_MANIFEST_PATH),
        "code_commit": CODE_COMMIT,
        "start_time": now,
        "end_time": now,
        "selected_epoch": np.nan,
        "selected_threshold": summary["threshold"],
        "threshold_objective": THRESHOLD_OBJECTIVE,
        "checkpoint_path": model_paths,
        "relative_path": [
            os.path.relpath(path, SAVE_ROOT).replace(os.sep, "/")
            for path in model_paths
        ],
        "bytes": [os.path.getsize(path) for path in model_paths],
        "sha256": [sha256_file(path) for path in model_paths],
    })
    extras = summary.drop(columns=["threshold"], errors="ignore").reset_index(drop=True)
    manifest = required.copy()
    for column in extras.columns:
        if column not in manifest.columns:
            manifest[column] = extras[column].to_numpy()
    manifest = manifest.loc[:, ~manifest.columns.duplicated()]
    ordered_columns = RUN_MANIFEST_COLUMNS + [
        column for column in manifest.columns
        if column not in RUN_MANIFEST_COLUMNS
    ]
    logs_dir = os.path.join(output_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    manifest[ordered_columns].to_csv(
        os.path.join(logs_dir, "run_manifest.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    global_path = os.path.join(SAVE_ROOT, "logs", "run_manifest.csv")
    os.makedirs(os.path.dirname(global_path), exist_ok=True)
    header = not os.path.exists(global_path) or os.path.getsize(global_path) == 0
    manifest[RUN_MANIFEST_COLUMNS].to_csv(
        global_path,
        mode="a",
        header=header,
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


def select_feature_columns(all_columns, ablation_mode):
    if ablation_mode not in ABLATION_MODES:
        raise ValueError(f"未知消融模式: {ablation_mode}")
    if ablation_mode == "full":
        return list(all_columns)
    if ablation_mode == "wo_missing":
        return [col for col in all_columns if not col.endswith("__missing")]
    if ablation_mode == "wo_derived":
        return [col for col in all_columns if not col.startswith("d_")]
    if ablation_mode == "base_only":
        return [col for col in all_columns if col in FEATURE_COLS]
    if ablation_mode.startswith("wo_"):
        base_feature = ablation_mode[3:]
        if base_feature not in FEATURE_COLS:
            raise ValueError(f"未知特征消融模式: {ablation_mode}")
        removed = {
            base_feature,
            f"{base_feature}__missing",
            *DERIVED_FEATURE_DEPENDENCIES.get(base_feature, set()),
        }
        return [col for col in all_columns if col not in removed]
    raise AssertionError(ablation_mode)


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


def prepare_video_matrices(df_fit, df_eval, ablation_mode="full"):
    fill_values = fit_fill_values(df_fit)
    fit_row = build_segment_features(df_fit, fill_values).copy()
    eval_row = build_segment_features(df_eval, fill_values).copy()
    fit_row["video_key"] = df_fit["video_key"].to_numpy()
    fit_row["label"] = df_fit["label"].to_numpy()
    eval_row["video_key"] = df_eval["video_key"].to_numpy()
    eval_row["label"] = df_eval["label"].to_numpy()

    all_row_feature_cols = [
        col for col in fit_row.columns
        if col not in {"video_key", "label"}
    ]
    row_feature_cols = select_feature_columns(
        all_row_feature_cols, ablation_mode
    )
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
        "ablation_mode": ablation_mode,
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


def make_inner_oof_predictions(
    df_outer_train,
    train_meta,
    split_manifest,
    fold_idx,
    class_weight,
    ablation_mode,
):
    keys = train_meta["video_key"].to_numpy()
    labels = train_meta["label"].to_numpy()
    key_to_index = {key: idx for idx, key in enumerate(keys)}
    oof_prob = np.full(len(keys), np.nan, dtype=np.float64)

    fold_manifest = split_manifest[
        split_manifest["outer_fold"].eq(fold_idx)
    ]
    for inner_idx in range(1, INNER_FOLD + 1):
        inner_manifest = fold_manifest[
            fold_manifest["inner_fold"].eq(inner_idx)
        ]
        fit_keys = set(
            inner_manifest.loc[
                inner_manifest["role"].eq("inner_train"), "video_key"
            ]
        )
        val_keys = set(
            inner_manifest.loc[
                inner_manifest["role"].eq("inner_val"), "video_key"
            ]
        )
        df_fit = df_outer_train[df_outer_train["video_key"].isin(fit_keys)].copy()
        df_val = df_outer_train[df_outer_train["video_key"].isin(val_keys)].copy()
        data = prepare_video_matrices(df_fit, df_val, ablation_mode)
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
    split_manifest,
    fold_idx,
    ablation_mode,
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
            split_manifest,
            fold_idx,
            class_weight,
            ablation_mode,
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


def run_one_ablation(
    df,
    video_meta,
    split_manifest,
    outer_splits,
    ablation_mode,
):
    output_root = os.path.join(SAVE_ROOT, ablation_mode)
    os.makedirs(output_root, exist_ok=True)
    prepare_run_metadata(output_root, ablation_mode)
    summary_rows, all_test = [], []
    epoch_metric_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        outer_splits,
        start=1,
    ):
        fold_dir = os.path.join(output_root, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        train_meta = video_meta.iloc[train_idx].reset_index(drop=True)
        test_meta = video_meta.iloc[test_idx].reset_index(drop=True)
        train_keys = set(train_meta["video_key"])
        test_keys = set(test_meta["video_key"])
        df_train = df[df["video_key"].isin(train_keys)].copy()
        df_test = df[df["video_key"].isin(test_keys)].copy()

        print(
            f"\n================ {ablation_mode} | "
            f"Fold {fold_idx}/{OUTER_FOLD} ================"
        )
        print("选择 class_weight 和 OOF threshold...")
        best = select_class_weight_and_threshold(
            df_train,
            train_meta,
            split_manifest,
            fold_idx,
            ablation_mode,
        )
        class_weight = best["class_weight"]
        threshold = best["threshold"]
        pd.DataFrame(best["candidate_metrics"]).to_csv(
            os.path.join(fold_dir, "threshold_selection_candidates.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        inner_oof_frame = pd.DataFrame(
            {
                "model": "rf_ensemble",
                "ablation": ablation_mode,
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

        data = prepare_video_matrices(
            df_train, df_test, ablation_mode
        )
        models = fit_models(
            data["X_fit"],
            data["y_fit"],
            SEED + 1000 + fold_idx,
            class_weight,
        )
        test_prob = predict_ensemble(models, data["X_eval"])
        test_pred = (test_prob >= threshold).astype(np.int64)
        test_metric = calculate_metrics(
            data["y_eval"], test_pred, test_prob
        )

        metadata = {
            "feature_cols": FEATURE_COLS,
            "row_feature_cols": data["row_feature_cols"],
            "video_feature_cols": data["video_feature_cols"],
            "fill_values": data["fill_values"],
            "ablation_mode": ablation_mode,
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
                "ablation_mode": ablation_mode,
            },
        }
        bundle_path = os.path.join(
            fold_dir, "video_ensemble_bundle.pkl"
        )
        joblib.dump(
            {"models": models, "metadata": metadata},
            bundle_path,
        )
        epoch_metric_rows.append({
            "model": "rf_ensemble",
            "ablation": ablation_mode,
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
        test_video["ablation"] = ablation_mode
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
            bundle_path,
            SAVE_ROOT,
        )
        test_video.to_csv(
            os.path.join(fold_dir, "fold_test_vid_pred.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        all_test.append(test_video)

        print(f"selected weight={class_weight}, threshold={threshold:.4f}")
        print(
            f"Test ACC={test_metric['ACC']:.4f}, "
            f"AUC={test_metric['AUC']:.4f}, "
            f"PR-AUC={test_metric['PR-AUC']:.4f}, "
            f"Spec={test_metric['Specificity']:.4f}, "
            f"Sens={test_metric['Sensitivity']:.4f}"
        )
        summary_rows.append({
            "ablation": ablation_mode,
            "fold": fold_idx,
            "outer_fold": fold_idx,
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
                bundle_path,
                output_root,
            ),
        })

    summary = pd.DataFrame(summary_rows)
    save_fold_mean_std(summary, output_root)
    summary.to_csv(
        os.path.join(output_root, "outer_5fold_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    all_test = pd.concat(all_test, ignore_index=True)
    all_test.to_csv(
        os.path.join(output_root, "all_outer_test_vid_pred.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    all_test[PREDICTION_COLUMNS].to_csv(
        os.path.join(output_root, "standardized_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    save_run_manifest(summary, output_root, ablation_mode)
    save_epoch_metrics(epoch_metric_rows, output_root)

    # 直接使用每个 outer fold 根据 inner-OOF 阈值生成的 prediction。
    pooled_pred = all_test["prediction"].to_numpy(dtype=np.int64)
    pooled_metric = calculate_metrics(
        y_true=all_test["label"].to_numpy(dtype=np.int64),
        y_pred=pooled_pred,
        y_prob=all_test["prob"].to_numpy(dtype=np.float64),
    )
    pooled_metrics, pooled_bootstrap, pooled_calibration = save_test_statistics(
        all_test,
        output_root,
        model_name=f"rf_ensemble_{ablation_mode}",
        predicted=pooled_pred,
    )
    save_pooled_paper_table(
        pooled_metrics,
        pooled_bootstrap,
        pooled_calibration,
        output_root,
    )

    print(f"\n================ {ablation_mode} OOF 5-fold Summary ================")
    print(summary.to_string(index=False))
    print(f"\n================ {ablation_mode} Pooled OOF Test ================")
    print(
        "threshold=per-fold inner-OOF balanced-accuracy, "
        f"ACC={pooled_metric['ACC']:.4f}, AUC={pooled_metric['AUC']:.4f}, "
        f"PR-AUC={pooled_metric['PR-AUC']:.4f}, "
        f"F1={pooled_metric['F1']:.4f}, "
        f"BalAcc={pooled_metric['Balanced_ACC']:.4f}, "
        f"Spec={pooled_metric['Specificity']:.4f}, "
        f"Sens={pooled_metric['Sensitivity']:.4f}"
    )
    print("\n================ Mean ± Std ================")
    for column in [
        "test_acc", "test_auc", "test_prauc", "test_f1", "test_bal_acc",
        "test_sensitivity", "test_specificity",
    ]:
        print(
            f"{column:12s}: {summary[column].mean():.4f} "
            f"± {summary[column].std():.4f}"
        )
    return (
        summary,
        all_test,
        pooled_metrics,
        pooled_bootstrap,
        pooled_calibration,
        epoch_metric_rows,
    )


def main():
    seed_everything(SEED)
    prepare_run_metadata()
    os.makedirs(os.path.join(SAVE_ROOT, "logs"), exist_ok=True)
    pd.DataFrame(columns=RUN_MANIFEST_COLUMNS).to_csv(
        os.path.join(SAVE_ROOT, "logs", "run_manifest.csv"),
        index=False,
        encoding="utf-8-sig",
    )

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
        raise ValueError(
            f"最少类别只有 {min_count} 个视频，无法进行 {OUTER_FOLD} 折 CV"
        )
    split_manifest, outer_splits = load_split_manifest(
        video_meta, SPLIT_MANIFEST_PATH
    )
    output_manifest_path = os.path.join(SAVE_ROOT, "split_manifest.csv")
    if os.path.abspath(output_manifest_path) != os.path.abspath(SPLIT_MANIFEST_PATH):
        shutil.copyfile(SPLIT_MANIFEST_PATH, output_manifest_path)
    print(
        f"片段数={len(df)}, 视频数={len(video_meta)}, "
        f"类别分布={video_meta['label'].value_counts().to_dict()}"
    )

    summaries = []
    prediction_frames = []
    pooled_metric_rows = []
    pooled_bootstrap_frames = []
    pooled_calibration_rows = []
    all_epoch_metric_rows = []
    for ablation_mode in ABLATION_MODES:
        (
            summary,
            predictions,
            metrics,
            bootstrap,
            calibration,
            epoch_metric_rows,
        ) = run_one_ablation(
            df,
            video_meta,
            split_manifest,
            outer_splits,
            ablation_mode,
        )
        summaries.append(summary)
        prediction_frames.append(predictions)
        pooled_metric_rows.append({"ablation": ablation_mode, **metrics})
        bootstrap = bootstrap.copy()
        bootstrap.insert(0, "ablation", ablation_mode)
        pooled_bootstrap_frames.append(bootstrap)
        pooled_calibration_rows.append({
            "ablation": ablation_mode,
            **calibration,
        })
        all_epoch_metric_rows.extend(epoch_metric_rows)

    pd.concat(summaries, ignore_index=True).to_csv(
        os.path.join(SAVE_ROOT, "ablation_outer_5fold_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(pooled_metric_rows).to_csv(
        os.path.join(SAVE_ROOT, "ablation_pooled_outer_test_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(pooled_bootstrap_frames, ignore_index=True).to_csv(
        os.path.join(SAVE_ROOT, "ablation_pooled_outer_test_bootstrap_ci.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(pooled_calibration_rows).to_csv(
        os.path.join(SAVE_ROOT, "ablation_pooled_outer_test_calibration.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(prediction_frames, ignore_index=True)[PREDICTION_COLUMNS].to_csv(
        os.path.join(SAVE_ROOT, "standardized_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    save_epoch_metrics(all_epoch_metric_rows, SAVE_ROOT)

    print("\n================ Feature Ablation Summary ================")
    print(pd.DataFrame(pooled_metric_rows).to_string(index=False))


if __name__ == "__main__":
    main()
