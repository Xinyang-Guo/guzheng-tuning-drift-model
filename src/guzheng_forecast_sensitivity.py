"""环境估计误差的示例情景分析，不代表历史天气预报准确率。"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from guzheng_future_validation import FEATURES, OUT, load_data, scores


def main():
    data, _, _ = load_data()
    manifest = pd.read_csv(OUT / "split_manifest.csv")
    test = data[data.session_id.isin(manifest.loc[manifest.split == "final_holdout", "session_id"])].copy()
    model = joblib.load(OUT / "evaluated_model.joblib")
    nominal = model.predict(test[FEATURES])
    rows, predictions = [], []
    # 人为设定的示例扰动；假设上一时点实测环境已知，因此当前值与变化量同向变化。
    for humidity_error in [-0.05, 0.0, 0.05]:
        for temperature_error in [-2.0, 0.0, 2.0]:
            inputs = test[FEATURES].copy()
            shifted = (inputs["湿度"] + humidity_error).clip(0, 1)
            actual_shift = shifted - inputs["湿度"]
            inputs["湿度"] = shifted
            inputs["湿度变化"] += actual_shift
            inputs["温度"] += temperature_error
            inputs["温度变化"] += temperature_error
            pred = model.predict(inputs)
            predictions.append(pred)
            rows.append({"humidity_error": humidity_error, "temperature_error": temperature_error,
                         "mean_abs_prediction_shift": float(np.abs(pred - nominal).mean()),
                         **scores(test["偏差"], pred)})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "environment_sensitivity.csv", index=False, encoding="utf-8-sig")
    ranges = test[["session_id", "日期", "古筝编号", "弦编号"]].copy()
    ranges["nominal_prediction"] = nominal
    ranges["scenario_min"] = np.min(predictions, axis=0)
    ranges["scenario_max"] = np.max(predictions, axis=0)
    ranges.to_csv(OUT / "scenario_prediction_ranges.csv", index=False, encoding="utf-8-sig")
    font = "/Library/Fonts/Arial Unicode.ttf"
    if Path(font).is_file():
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    grid = summary.pivot(index="humidity_error", columns="temperature_error", values="mae")
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = ax.imshow(grid.values, cmap="YlOrRd")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{grid.iloc[i,j]:.3f}", ha="center", va="center")
    ax.set_xticks(range(3), ["−2℃", "0℃", "+2℃"])
    ax.set_yticks(range(3), ["−5 个百分点", "0", "+5 个百分点"])
    ax.set(xlabel="估计温度误差", ylabel="估计湿度误差", title="环境估计偏差情景：最终留出集 MAE\n人为扰动，不代表实际预报误差分布")
    fig.colorbar(image, ax=ax, label="MAE")
    fig.savefig(OUT / "environment_sensitivity.png", dpi=180)
    plt.close(fig)
    path = OUT / "report.md"
    original = path.read_text().split("\n## 使用天气预报或估计值时\n")[0]
    original += (
        "\n## 使用天气预报或估计值时\n\n"
        "用户确认实际只能提供天气预报或环境估计值。因此上面的实测环境回溯成绩是参考，不能当作真实预报输入下的准确率。"
        "需要收集预测发出时保存的环境估计、预测提前量，以及实际调音结果，再做前瞻评估。室外天气预报也不等于古筝所在室内环境，应估计室内温湿度。\n\n"
        "补充了 9 组人为情景：湿度估计误差 −5/0/+5 个百分点、温度误差 −2/0/+2℃；这些幅度仅用于示例敏感性分析，未经历史预报数据校准。"
        "假设上一次环境值已知，变化量与目标时点环境值同步扰动；湿度限制在 0～1。没有用这些情景重新选择模型。\n\n"
        f"九种情景的 MAE 范围为 {summary.mae.min():.4f}～{summary.mae.max():.4f}；相对实测输入的平均预测变化最大为 {summary.mean_abs_prediction_shift.max():.4f}。"
        "该范围不是概率区间或置信区间；某个扰动情景误差降低也不代表应故意修改环境输入。\n\n"
        "![环境估计敏感性](environment_sensitivity.png)\n\n"
        "environment_sensitivity.csv 保存情景指标；scenario_prediction_ranges.csv 保存每条记录在九种情景下的预测范围。"
        "复现：`python src/guzheng_forecast_sensitivity.py`。\n"
    )
    path.write_text(original, encoding="utf-8")
    metadata_path = OUT / "selection.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["environment_at_prediction"] = "weather forecast or estimates; actual forecast accuracy not yet evaluated"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
