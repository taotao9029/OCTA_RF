# -*- coding: utf-8 -*-
"""RF/ExtraTrees/LR 集成模型的独立 Group-out 训练入口。"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import train_rf as model
from group_out_common import (
    attach_manifest,
    load_group_manifest,
    save_group_predictions,
    validate_group_table,
)


DATA_PATH = model.DATA_PATH
MANIFEST_PATH = "./feature_out/rename_manifest_with_groups.csv"
OUTPUT_ROOT = "./rf_group_out_results"
GROUP_TYPES = {
    "session": "session_id",
    "date": "date",
}


def load_data():
    dataframe = pd.read_csv(DATA_PATH)
    dataframe.columns = dataframe.columns.astype(str).str.strip()
    dataframe["filename"] = dataframe["filename"].astype(str)
    dataframe["video_key"] = dataframe["filename"].apply(
        model.get_original_video_key
    )
    dataframe["label"] = dataframe["label"].astype(int)
    manifest = load_group_manifest(MANIFEST_PATH)
    dataframe = attach_manifest(dataframe, manifest)
    model.validate_input(dataframe)
    return dataframe


def build_video_meta(dataframe, group_column):
    group_counts = dataframe.groupby("video_key")[group_column].nunique()
    if group_counts.max() > 1:
        bad = group_counts[group_counts > 1].index.tolist()
        raise ValueError(
            f"同一视频对应多个 {group_column}，无法进行 Group-out：{bad[:10]}"
        )
    return (
        dataframe[
            ["video_key", "label", group_column]
        ]
        .drop_duplicates("video_key")
        .sort_values("video_key")
        .reset_index(drop=True)
    )


def run_one_group_type(dataframe, group_type, group_column):
    groups = validate_group_table(dataframe, group_column)
    video_meta = build_video_meta(dataframe, group_column)
    output_dir = Path(OUTPUT_ROOT) / group_type
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for group_index, held_out_group in enumerate(groups, start=1):
        print(
            f"\n========== RF {group_type}: "
            f"held-out {held_out_group} ({group_index}/{len(groups)}) =========="
        )
        train_keys = set(
            video_meta.loc[
                video_meta[group_column].astype(str) != str(held_out_group),
                "video_key",
            ]
        )
        test_keys = set(
            video_meta.loc[
                video_meta[group_column].astype(str) == str(held_out_group),
                "video_key",
            ]
        )
        df_train = dataframe[dataframe["video_key"].isin(train_keys)].copy()
        df_test = dataframe[dataframe["video_key"].isin(test_keys)].copy()
        train_meta = video_meta[video_meta["video_key"].isin(train_keys)].copy()

        if train_meta["label"].nunique() < 2:
            raise ValueError(
                f"留出 {held_out_group} 后训练集只有一个类别，无法训练"
            )

        best = model.select_class_weight_and_threshold(
            df_train,
            train_meta,
            group_index,
        )
        data = model.prepare_video_matrices(df_train, df_test)
        models = model.fit_models(
            data["X_fit"],
            data["y_fit"],
            model.SEED + 1000 + group_index,
            best["class_weight"],
        )
        probability = model.predict_ensemble(models, data["X_eval"])
        threshold = float(best["threshold"])
        prediction = (probability >= threshold).astype(np.int64)

        checkpoint_path = checkpoint_dir / f"held_out_{group_index:03d}.pkl"
        metadata = {
            "model": "rf_ensemble",
            "group_type": group_type,
            "held_out_group": str(held_out_group),
            "threshold": threshold,
            "threshold_objective": model.THRESHOLD_OBJECTIVE,
            "class_weight": best["class_weight"],
            "test_video_keys": sorted(str(key) for key in test_keys),
        }
        joblib.dump(
            {"models": models, "metadata": metadata},
            checkpoint_path,
        )

        for key, label, prob, pred in zip(
            data["eval_video"]["video_key"].astype(str),
            data["eval_video"]["label"].astype(int),
            probability,
            prediction,
        ):
            rows.append(
                {
                    "group_type": group_type,
                    "held_out_group": str(held_out_group),
                    "animal_id": key,
                    "label": int(label),
                    "probability": float(prob),
                    "threshold": threshold,
                    "prediction": int(pred),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_group_predictions(
        rows,
        output_dir / "leave_{}_out_predictions.csv".format(group_type),
    )


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    dataframe = load_data()
    for group_type, group_column in GROUP_TYPES.items():
        run_one_group_type(dataframe, group_type, group_column)
    print(f"RF Group-out 结果已保存到：{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
