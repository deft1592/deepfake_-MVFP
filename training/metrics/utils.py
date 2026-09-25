from sklearn import metrics
import numpy as np


def parse_metric_for_print(metric_dict):
    if metric_dict is None:
        return "\n"
    str = "\n"
    str += "================================ Each dataset best metric ================================ \n"
    for key, value in metric_dict.items():
        if key != "avg":
            str = str + f"| {key}: "
            for k, v in value.items():
                str = str + f" {k}={v} "
            str = str + "| \n"
        else:
            str += "============================================================================================= \n"
            str += "================================== Average best metric ====================================== \n"
            avg_dict = value
            for avg_key, avg_value in avg_dict.items():
                if avg_key == "dataset_dict":
                    for key, value in avg_value.items():
                        str = str + f"| {key}: {value} | \n"
                else:
                    str = str + f"| avg {avg_key}: {avg_value} | \n"
    str += "============================================================================================="
    return str


import os
import numpy as np
import pandas as pd
from sklearn import metrics
from collections import defaultdict

def get_test_metrics(y_pred, y_true, img_names, save_path=None):
    def get_video_metrics(image, pred, label):
        video_frames = defaultdict(lambda: {"pred": [], "label": []})
        for image_path, frame_pred, frame_label in zip(image, pred, label):
            # Each extracted video's frames are stored in one directory. Use the
            # complete normalized parent path to avoid collisions across subsets.
            normalized_path = os.fspath(image_path).replace("\\", "/")
            video_id = normalized_path.rsplit("/", 1)[0]
            video_frames[video_id]["pred"].append(float(frame_pred))
            video_frames[video_id]["label"].append(int(frame_label))

        new_label = []
        new_pred = []
        for video_id, frames in video_frames.items():
            labels = np.unique(frames["label"])
            if len(labels) != 1:
                raise ValueError(
                    f"Inconsistent frame labels in video {video_id}: {labels.tolist()}"
                )
            new_pred.append(float(np.mean(frames["pred"])))
            new_label.append(int(labels[0]))

        if not new_label:
            raise ValueError("No valid videos were found for video-level evaluation")

        print(f"视频数量: {len(new_label)}")
        print(f"视频级标签分布: {np.unique(new_label, return_counts=True)}")
        print(f"视频级预测值范围: [{np.min(new_pred):.3f}, {np.max(new_pred):.3f}]")
        print(f"预测值包含NaN: {np.any(np.isnan(new_pred))}")
        print(f"预测值包含Inf: {np.any(np.isinf(new_pred))}")

        if len(np.unique(new_label)) < 2:
            raise ValueError(
                f"Video-level AUC requires both classes, got {np.unique(new_label)}"
            )

        return metrics.roc_auc_score(new_label, new_pred)

    # 转成 numpy，避免后面保存出问题
    y_pred = np.asarray(y_pred).squeeze()
    y_true = np.asarray(y_true).copy()

    # For UCF, where labels for different manipulations are not consistent.
    y_true[y_true >= 1] = 1

    # 保存逐样本预测结果，便于后续做 DeLong
    if save_path is not None:
        save_dict = {
            "img_name": img_names,
            "y_true": y_true.astype(int),
            "y_pred": y_pred.astype(float),
        }
        df = pd.DataFrame(save_dict)
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"预测结果已保存到: {save_path}")

    # auc
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    # eer
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    # ap
    ap = metrics.average_precision_score(y_true, y_pred)

    # acc
    prediction_class = (y_pred > 0.5).astype(int)
    correct = (prediction_class == np.clip(y_true, a_min=0, a_max=1)).sum().item()
    acc = correct / len(prediction_class)

    if not isinstance(img_names[0], (list, tuple, np.ndarray)):
        # calculate video-level auc for the frame-level methods.
        v_auc = get_video_metrics(img_names, y_pred, y_true)
    else:
        # video-level methods
        v_auc = auc

    return {
        "acc": acc,
        "auc": auc,
        "eer": eer,
        "ap": ap,
        "pred": y_pred,
        "video_auc": v_auc,
        "label": y_true,
    }
