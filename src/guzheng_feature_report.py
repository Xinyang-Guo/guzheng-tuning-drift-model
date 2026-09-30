"""比较去掉古筝编号前后的实验结果，生成中文报告及图表。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_report(output_dir, train, test):
    baseline_dir = Path(__file__).resolve().parent.parent / "output" / "parameter_comparison"
    before = pd.read_csv(baseline_dir / "comparison.csv")
    after = pd.read_csv(output_dir / "comparison.csv")
    # 核对同一组参数、相同测试记录及目标值后再做前后比较。
    for row in after.itertuples():
        relative = Path(row.model) / f"run_{row.run}" / "predictions.csv"
        columns = ["session_id", "日期", "古筝编号", "弦编号", "真实偏差"]
        pd.testing.assert_frame_equal(
            pd.read_csv(baseline_dir / relative)[columns],
            pd.read_csv(output_dir / relative)[columns],
        )
    keys = ["model", "run", "parameters", "split_random_state", "test_size", "train_count", "test_count"]
    comparison = before.merge(after, on=keys, suffixes=("_before", "_after"), validate="one_to_one")
    assert len(comparison) == len(before) == len(after) == 6
    comparison["delta_test_r2"] = comparison.test_r2_after - comparison.test_r2_before
    comparison["delta_test_mse"] = comparison.test_mse_after - comparison.test_mse_before
    comparison.to_csv(output_dir / "before_after_comparison.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    names = {"random_forest": "随机森林", "ridge": "岭回归"}
    for row_index, (model, title) in enumerate(names.items()):
        subset = comparison[comparison.model == model]
        x = np.arange(len(subset))
        for col_index, metric in enumerate(["r2", "mse"]):
            ax = axes[row_index, col_index]
            for offset, suffix, label in [(-0.18, "before", "包含古筝编号"), (0.18, "after", "移除古筝编号")]:
                bars = ax.bar(x + offset, subset[f"test_{metric}_{suffix}"], width=0.36, label=label)
                ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
            ax.set_xticks(x, [f"第 {run} 组" for run in subset.run])
            ax.set_title(f"{title} · 测试集 {'R²（越高越好）' if metric == 'r2' else 'MSE（越低越好）'}")
            ax.margins(y=0.22)
            ax.legend(loc="upper center", fontsize=9)
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle("移除古筝编号前后：相同参数、相同测试集")
    fig.savefig(output_dir / "before_after_comparison.png", dpi=200)
    plt.close(fig)

    lines = [
        "# 移除古筝编号后的模型实验报告", "",
        "## 实验设置", "",
        "模型输入由 6 个特征减为 5 个：湿度、湿度变化、温度、温度变化、弦编号。",
        "古筝编号仅用于关联原始数据及标识预测记录，不进入训练或预测输入。",
        f"沿用原预处理，训练集 {len(train)} 条（{train.session_id.nunique()} 次调音），"
        f"测试集 {len(test)} 条（{test.session_id.nunique()} 次调音）。",
        "使用 GroupShuffleSplit 按 session_id 分组，test_size=0.3，random_state=42；训练和测试 session 无交集。",
        "岭回归标准化仅在训练集拟合。每个模型比较三组参数，固定随机种子；共六次参数实验。",
        "移除前的对照取自 ../parameter_comparison，已逐组核对参数、测试记录及真实值一致。", "",
        "## 三组参数的前后对比", "",
        "Δ 为移除后减移除前；R² 越高越好，MSE 越低越好。", "",
        "| 模型 | 参数 | 移除前 R² | 移除后 R² | ΔR² | 移除前 MSE | 移除后 MSE | ΔMSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in comparison.itertuples():
        lines.append(f"| {names[r.model]} | {r.parameters} | {r.test_r2_before:.6f} | {r.test_r2_after:.6f} | {r.delta_test_r2:+.6f} | {r.test_mse_before:.4f} | {r.test_mse_after:.4f} | {r.delta_test_mse:+.4f} |")
    lines += ["", "## 结果解读", ""]
    for model, name in names.items():
        subset = comparison[comparison.model == model]
        best = subset.loc[subset.test_mse_after.idxmin()]
        improved = int((subset.delta_test_mse < 0).sum())
        lines.append(f"- {name}：三组中有 {improved} 组的测试 MSE 降低。移除后本次测试表现最好的是 {best['parameters']}，"
                     f"测试 R²={best.test_r2_after:.6f}，MSE={best.test_mse_after:.4f}。"
                     f"相对同参数对照，ΔR²={best.delta_test_r2:+.6f}，ΔMSE={best.delta_test_mse:+.4f}。")
    lines += [
        "", "这是一种固定数据划分下的特征移除对比，不能据此判断统计显著性或认定全局最优参数。",
        "按调音 session 划分并不保证古筝实体互斥；这些结果不能直接证明模型可泛化到未见过的古筝。",
        "随机森林三组同时改变树数量与深度，因此组间差异不能归因于单一参数。", "",
        "## 图表和输出", "",
        "![移除前后对比](before_after_comparison.png)", "",
        "![随机森林参数指标](random_forest/metrics_comparison.png)", "",
        "![岭回归参数指标](ridge/metrics_comparison.png)", "",
        "![随机森林预测散点](random_forest/prediction_comparison.png)", "",
        "![岭回归预测散点](ridge/prediction_comparison.png)", "",
        "各模型的 run_1、run_2、run_3 目录内包含 predictions.csv、metrics.csv 和 actual_vs_predicted.png。",
        "comparison.csv 保存本次六组参数和指标；before_after_comparison.csv 保存前后差异。",
        "original_config 目录为原独立脚本去掉该特征后的结果：随机森林 500 棵树/深度 7；岭回归使用原 RidgeCV 自动选择 alpha，未纳入上述固定参数六组对比。", "",
        "复现参数实验和本报告：`python src/guzheng_parameter_comparison.py`。", "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
