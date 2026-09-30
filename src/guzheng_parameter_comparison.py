"""随机森林与岭回归各运行三组参数，结果保存到 output/without_guzheng_id。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from guzheng_feature_report import write_report


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output" / "without_guzheng_id"
FEATURES = ["湿度", "湿度变化", "温度", "温度变化", "弦编号"]
CONFIGS = {
    "random_forest": [
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 300, "max_depth": 7},
        {"n_estimators": 500, "max_depth": 9},
    ],
    "ridge": [{"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0}],
}
MODEL_NAMES = {"random_forest": "随机森林", "ridge": "岭回归"}


def load_data():
    """保持与原有四个脚本相同的预处理及按 session 划分方法。"""
    file_path = PROJECT_DIR / "data" / "古筝调音数据.xlsx"
    environment = pd.read_excel(file_path, sheet_name="Environment")
    pitch = pd.read_excel(file_path, sheet_name="Pitch_Data")
    pitch["session_id"] = np.arange(len(pitch)) // 21 + 1
    session_info = pitch.groupby("session_id", as_index=False).first()[
        ["session_id", "日期", "古筝编号"]
    ]
    keys = ["日期", "古筝编号"]
    session_info["record_no"] = session_info.groupby(keys).cumcount()
    environment["record_no"] = environment.groupby(keys).cumcount()
    matched = environment.merge(session_info, on=keys + ["record_no"], how="inner")
    data = pitch.merge(
        matched[["session_id", "湿度", "湿度变化", "温度", "温度变化"]],
        on="session_id", how="left",
    )
    data["湿度"] = data["湿度"].astype(str).str.replace("*", "", regex=False)
    for column in ["湿度", "湿度变化", "温度变化"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["湿度变化", "温度变化"]).copy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train, test = next(splitter.split(data[FEATURES], data["偏差"], groups=data["session_id"]))
    assert set(data.iloc[train]["session_id"]).isdisjoint(data.iloc[test]["session_id"])
    return data.iloc[train].copy(), data.iloc[test].copy()


def save_scatter(ax, actual, predicted, title, metrics, limits):
    ax.scatter(actual, predicted, s=22, alpha=0.5)
    ax.plot(limits, limits, "--", color="gray", label="理想预测线 y=x")
    ax.set(xlim=limits, ylim=limits, xlabel="真实偏差", ylabel="预测偏差", title=title)
    ax.set_aspect("equal", adjustable="box")
    ax.text(0.04, 0.95, f"测试 R² = {metrics['test_r2']:.6f}\n测试 MSE = {metrics['test_mse']:.4f}",
            transform=ax.transAxes, va="top")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")


def main():
    font_path = Path("/Library/Fonts/Arial Unicode.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test = load_data()
    all_metrics = []

    for model_name, configs in CONFIGS.items():
        model_dir = OUTPUT_DIR / model_name
        runs = []
        for run_id, params in enumerate(configs, 1):
            run_dir = model_dir / f"run_{run_id}"
            run_dir.mkdir(parents=True, exist_ok=True)
            if model_name == "ridge":
                # 每次固定 alpha；标准化只在训练集上拟合。
                model = make_pipeline(StandardScaler(), Ridge(**params))
            else:
                model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            model.fit(train[FEATURES], train["偏差"])
            assert model.n_features_in_ == 5
            assert "古筝编号" not in model.feature_names_in_
            train_pred = model.predict(train[FEATURES])
            test_pred = model.predict(test[FEATURES])
            label = ", ".join(f"{key}={value}" for key, value in params.items())
            metrics = {
                "model": model_name, "run": run_id, "parameters": label,
                "features": ", ".join(FEATURES), "feature_count": len(FEATURES),
                "split_random_state": 42, "test_size": 0.3,
                "model_random_state": 42 if model_name == "random_forest" else None,
                "train_count": len(train), "test_count": len(test),
                "train_r2": r2_score(train["偏差"], train_pred),
                "test_r2": r2_score(test["偏差"], test_pred),
                "train_mse": mean_squared_error(train["偏差"], train_pred),
                "test_mse": mean_squared_error(test["偏差"], test_pred),
            }
            predictions = test[["session_id", "日期", "古筝编号", "弦编号", "偏差"]].rename(
                columns={"偏差": "真实偏差"}
            ).copy()
            predictions["预测偏差"] = test_pred
            predictions.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame([metrics]).to_csv(run_dir / "metrics.csv", index=False, encoding="utf-8-sig")
            runs.append((run_dir, label, metrics, test_pred))
            all_metrics.append(metrics)
            print(f"{model_name} run_{run_id}: {label}, R²={metrics['test_r2']:.6f}, MSE={metrics['test_mse']:.4f}", flush=True)

        # 同一模型的三张散点图统一坐标范围，便于比较。
        values = np.concatenate([test["偏差"].to_numpy()] + [run[3] for run in runs])
        padding = (values.max() - values.min()) * 0.05 or 1.0
        limits = (values.min() - padding, values.max() + padding)
        combined, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        for ax, (run_dir, label, metrics, predicted) in zip(axes, runs):
            title = f"{MODEL_NAMES[model_name]} · 第 {metrics['run']} 组\n{label}"
            save_scatter(ax, test["偏差"], predicted, title, metrics, limits)
            fig, single_ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
            save_scatter(single_ax, test["偏差"], predicted, title, metrics, limits)
            fig.savefig(run_dir / "actual_vs_predicted.png", dpi=200)
            plt.close(fig)
        combined.savefig(model_dir / "prediction_comparison.png", dpi=200)
        plt.close(combined)

        summary = pd.DataFrame([run[2] for run in runs])
        summary.to_csv(model_dir / "comparison.csv", index=False, encoding="utf-8-sig")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        labels = [f"第 {i + 1} 组\n{run[1].replace(', ', chr(10))}" for i, run in enumerate(runs)]
        for ax, metric, title in zip(axes, ["r2", "mse"], ["R²（越高越好）", "MSE（越低越好）"]):
            for split, caption in [("train", "训练集"), ("test", "测试集")]:
                values = summary[f"{split}_{metric}"]
                ax.plot(range(3), values, marker="o", label=caption)
                for i, value in enumerate(values):
                    ax.annotate(f"{value:.6f}", (i, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
            ax.set_xticks(range(3), labels)
            ax.set_title(title)
            ax.margins(y=0.2, x=0.15)
            ax.grid(alpha=0.25)
            ax.legend()
        fig.suptitle(f"{MODEL_NAMES[model_name]}：三组参数对比（固定同一数据划分）")
        fig.savefig(model_dir / "metrics_comparison.png", dpi=200)
        plt.close(fig)

    pd.DataFrame(all_metrics).to_csv(OUTPUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "README.md").write_text(
        "# 移除古筝编号后的三组参数实验\n\n"
        "输入特征仅为湿度、湿度变化、温度、温度变化、弦编号。古筝编号仅用于数据匹配和结果追溯。\n\n"
        "随机森林：100 棵树/深度 5、300 棵树/深度 7、500 棵树/深度 9；随机种子均为 42。\n"
        "岭回归：alpha 为 0.1、1、10，使用 StandardScaler，仅在训练集拟合标准化参数。\n\n"
        "两个模型各运行三组参数，共六次训练。所有实验使用相同的 GroupShuffleSplit，"
        "按 session_id 分组，test_size=0.3，random_state=42，预处理与原脚本一致。\n"
        "这是固定数据划分上的参数比较，不是三次随机重复实验。随机森林同时改变两个参数，"
        "结果不能用于单独判断某一个参数的影响，也不代表交叉验证后的最优参数。\n\n"
        "comparison.csv 为总指标表；各模型目录包含指标表、预测对比图和指标对比图。"
        "run_1 至 run_3 分别包含参数及指标、测试集预测结果和预测散点图。\n\n"
        "运行：`python src/guzheng_parameter_comparison.py`\n",
        encoding="utf-8",
    )
    write_report(OUTPUT_DIR, train, test)
    print(f"结果保存至：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
