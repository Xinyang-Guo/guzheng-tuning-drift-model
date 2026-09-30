"""按完整日期滚动验证，搜索五特征模型；最终日期只用于评估。"""

import json
from datetime import date
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "future_validation"
FEATURES = ["湿度", "湿度变化", "温度", "温度变化", "弦编号"]


def load_data():
    source = ROOT / "data" / "古筝调音数据.xlsx"
    environment = pd.read_excel(source, sheet_name="Environment")
    pitch = pd.read_excel(source, sheet_name="Pitch_Data")
    assert len(pitch) % 21 == 0, "音高记录并非完整的 21 弦 session"
    pitch["session_id"] = np.arange(len(pitch)) // 21 + 1
    for _, session in pitch.groupby("session_id"):
        assert sorted(session["弦编号"].tolist()) == list(range(1, 22))
        assert session["日期"].nunique() == session["古筝编号"].nunique() == 1
    keys = ["日期", "古筝编号"]
    sessions = pitch.groupby("session_id", as_index=False).first()[["session_id"] + keys]
    for frame in [sessions, environment]:
        frame["record_no"] = frame.groupby(keys).cumcount()
    matched = sessions.merge(environment, on=keys + ["record_no"], how="left", validate="one_to_one", indicator=True)
    assert matched["_merge"].eq("both").all(), "有调音记录缺少匹配环境"
    data = pitch.merge(matched[["session_id"] + FEATURES[:-1]], on="session_id", validate="many_to_one")
    data["湿度"] = data["湿度"].astype(str).str.replace("*", "", regex=False)
    for column in FEATURES + ["偏差"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    dropped = len(data) - len(data.dropna(subset=FEATURES + ["偏差"]))
    data = data.dropna(subset=FEATURES + ["偏差"]).copy()
    assert data["湿度"].between(0, 1).all(), "湿度应统一为 0～1"
    def date_key(value):
        month = int(value)
        day = int(round((value - month) * 100))
        # 年份未知；2000 仅用于校验月日并排序，不表示真实采集年份。
        return date(2000, month, day).timetuple().tm_yday
    data["date_order"] = data["日期"].map(date_key)
    data["date_label"] = data["日期"].map(lambda v: f"{int(v):02d}-{int(round((v-int(v))*100)):02d}")
    data = data.sort_values(["date_order", "session_id", "弦编号"]).reset_index(drop=True)
    return data, dropped, len(environment) - len(sessions)


def scores(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {"mse": float(mse), "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(actual, predicted)), "r2": float(r2_score(actual, predicted))}


def make_model(name, params):
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1, **params)
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**params))
    return DummyRegressor(strategy="mean")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font = "/Library/Fonts/Arial Unicode.ttf"
    if Path(font).is_file():
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    data, dropped, unmatched = load_data()
    dates = data.date_order.unique()
    cutoff = len(dates) - int(np.ceil(len(dates) * 0.2))
    development = data[data.date_order.isin(dates[:cutoff])].reset_index(drop=True)
    holdout = data[data.date_order.isin(dates[cutoff:])].reset_index(drop=True)
    assert development.date_order.max() < holdout.date_order.min()
    assert set(development.session_id).isdisjoint(holdout.session_id)
    manifest = data[["session_id", "日期", "date_label", "古筝编号"]].drop_duplicates().copy()
    manifest["split"] = np.where(manifest.session_id.isin(development.session_id), "development", "final_holdout")
    manifest.to_csv(OUT / "split_manifest.csv", index=False, encoding="utf-8-sig")
    folds = []
    fold_metadata = []
    for fold, (train_dates, valid_dates) in enumerate(TimeSeriesSplit(n_splits=4).split(dates[:cutoff]), 1):
        train = development[development.date_order.isin(dates[train_dates])]
        valid = development[development.date_order.isin(dates[valid_dates])]
        assert train.date_order.max() < valid.date_order.min()
        assert set(train.session_id).isdisjoint(valid.session_id)
        folds.append((train, valid))
        fold_metadata.append({"fold": fold, "train_start": train.date_label.iloc[0], "train_end": train.date_label.iloc[-1],
                              "valid_start": valid.date_label.iloc[0], "valid_end": valid.date_label.iloc[-1],
                              "train_rows": len(train), "valid_rows": len(valid)})
    pd.DataFrame(fold_metadata).to_csv(OUT / "cv_splits.csv", index=False, encoding="utf-8-sig")
    forest_grid = {"max_depth": [3, 4, 5, 6, 7], "min_samples_leaf": [2, 5, 10, 20],
                   "min_samples_split": [2, 10, 20], "max_features": [0.6, 0.8, 1.0]}
    candidates = [("random_forest", p) for p in ParameterSampler(forest_grid, n_iter=24, random_state=42)]
    # 同时加入历史推荐配置，树数固定 300 以便比较其余参数。
    candidates += [("random_forest", {"max_depth": 5, "min_samples_leaf": 1, "min_samples_split": 2, "max_features": 1.0})]
    candidates += [("ridge", {"alpha": float(a)}) for a in np.logspace(-3, 3, 13)]
    candidates += [("mean_baseline", {})]
    rows, details = [], []
    for candidate_id, (name, params) in enumerate(candidates, 1):
        fold_scores = []
        for fold, (train, valid) in enumerate(folds, 1):
            model = make_model(name, params)
            model.fit(train[FEATURES], train["偏差"])
            result = scores(valid["偏差"], model.predict(valid[FEATURES]))
            fold_scores.append(result)
            details.append({"candidate": candidate_id, "model": name, "fold": fold, **result})
        summary = {"candidate": candidate_id, "model": name, "params": json.dumps(params, ensure_ascii=False),
                   "cv_mse": np.mean([s["mse"] for s in fold_scores]), "cv_mse_std": np.std([s["mse"] for s in fold_scores]),
                   "cv_mae": np.mean([s["mae"] for s in fold_scores]), "cv_r2": np.mean([s["r2"] for s in fold_scores])}
        rows.append(summary)
        print(f"候选 {candidate_id}/{len(candidates)} {name}: CV MSE={summary['cv_mse']:.4f}", flush=True)
    search = pd.DataFrame(rows).sort_values("cv_mse")
    search.to_csv(OUT / "search_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(details).to_csv(OUT / "cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    best_per_model = search.groupby("model", sort=False).head(1)
    # 包含简单均值基线；若复杂模型不如基线，则不强行推荐复杂模型。
    selected = search.iloc[0]
    selected_name = selected["model"]
    metadata = {"features": FEATURES, "selection_rule": "lowest mean MSE over four chronological validation folds",
                "selected_model": selected_name, "selected_params": json.loads(selected.params),
                "holdout_start": holdout.date_label.iloc[0], "holdout_end": holdout.date_label.iloc[-1],
                "year": "unknown; dates interpreted as month.day in a single year", "environment_at_prediction": "must be supplied"}
    # 在查看最终留出成绩前保存选择，最终留出集不用于改选参数或模型。
    (OUT / "selection.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    final_rows = []
    for row in best_per_model.itertuples():
        model = make_model(row.model, json.loads(row.params))
        model.fit(development[FEATURES], development["偏差"])
        predicted = model.predict(holdout[FEATURES])
        result = scores(holdout["偏差"], predicted)
        final_rows.append({"model": row.model, "params": row.params, "selected_by_cv": row.model == selected_name,
                           "cv_mse": row.cv_mse, "cv_mse_std": row.cv_mse_std, **result})
        folder = OUT / row.model
        folder.mkdir(exist_ok=True)
        predictions = holdout[["session_id", "日期", "date_label", "古筝编号", "弦编号", "偏差"]].rename(columns={"偏差": "真实偏差"}).copy()
        predictions["预测偏差"] = predicted
        predictions["绝对误差"] = np.abs(predictions["真实偏差"] - predicted)
        predictions.to_csv(folder / "predictions.csv", index=False, encoding="utf-8-sig")
        for group, filename in [("弦编号", "errors_by_string.csv"), ("session_id", "errors_by_session.csv"), ("date_label", "errors_by_date.csv")]:
            errors = [{group: key, "count": len(part), **scores(part["真实偏差"], part["预测偏差"])} for key, part in predictions.groupby(group)]
            pd.DataFrame(errors).to_csv(folder / filename, index=False, encoding="utf-8-sig")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        axes[0].scatter(holdout["偏差"], predicted, alpha=0.45, s=18)
        limits = [min(holdout["偏差"].min(), predicted.min()), max(holdout["偏差"].max(), predicted.max())]
        axes[0].plot(limits, limits, "--", color="gray")
        axes[0].set(xlabel="真实偏差", ylabel="预测偏差", title=f"{row.model}：未来日期留出集\nR²={result['r2']:.4f}，MAE={result['mae']:.4f}")
        daily = predictions.groupby("date_label")["绝对误差"].mean()
        axes[1].plot(daily.index, daily.values, marker="o")
        axes[1].set(xlabel="日期（月-日）", ylabel="MAE", title="按日期的预测误差")
        fig.savefig(folder / "evaluation.png", dpi=180)
        plt.close(fig)
        if row.model == selected_name:
            joblib.dump(model, OUT / "evaluated_model.joblib")
            # 完成评估后，固定已选参数，在全部历史数据重训供下一次使用。
            production_model = clone(model).fit(data[FEATURES], data["偏差"])
            joblib.dump(production_model, OUT / "future_model.joblib")
    final = pd.DataFrame(final_rows)
    final.to_csv(OUT / "final_metrics.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, metric, title in [(axes[0], "cv_mse", "滚动验证平均 MSE（用于选模型）"), (axes[1], "mse", "最终日期留出 MSE（仅评估）")]:
        bars = ax.bar(final.model, final[metric])
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set_title(title)
        ax.margins(y=0.2)
    fig.savefig(OUT / "model_comparison.png", dpi=180)
    plt.close(fig)
    selected_result = final[final.selected_by_cv].iloc[0]
    report = ["# 面向未来调音的时间验证报告", "", "## 结论", "",
              f"按开发集的四折滚动验证，选择 **{selected_name}**；参数 `{selected.params}`。随机森林统一使用 n_estimators=300、random_state=42。",
              f"所选模型在末段日期的 MSE={selected_result.mse:.6f}，MAE={selected_result.mae:.6f}，R²={selected_result.r2:.6f}。最终留出结果未用于重新选择模型。", "",
              "## 数据与验证边界", "",
              f"有效数据 {len(data)} 行、{data.session_id.nunique()} 次调音、{len(dates)} 个日期；缺失特征/目标删除 {dropped} 行。环境表额外 {unmatched} 条记录无音高记录对应。已检查每个原始 session 的 21 根弦完整、日期和古筝一致，以及湿度在 0～1 内。",
              "日期为数字月.日，未记录年份；按同一年解读。同一天全部 session 和弦不可跨训练/验证边界。日期间隔不均匀，各折覆盖不同长度的日历时间。",
              f"开发期：{development.date_label.iloc[0]} 至 {development.date_label.iloc[-1]}，{len(development)} 行；最终留出期：{holdout.date_label.iloc[0]} 至 {holdout.date_label.iloc[-1]}，{len(holdout)} 行。",
              "开发期进行四折扩展窗口验证，每折只用过去训练、较晚日期验证；标准化在各折内部拟合。25 组随机森林、13 组岭回归与 1 组训练均值基线，共 39 个候选。按各折 MSE 的等权平均选择，标准差描述折间差异，不是置信区间。",
              "这次使用时间留出+开发期滚动调参，没有再套多轮外层嵌套验证。过去的随机划分实验已接触过这批记录，因此末段留出属于新的回溯评估设计，并非从未见过的外部测试集；真正前瞻验证仍需新采集记录。", "",
              "## 环境条件与预测时点", "",
              "仅使用湿度、湿度变化、温度、温度变化、弦编号，不使用古筝编号。模型预测的是给定环境条件下的调音偏差。",
              "当前回溯评估使用记录中的实际环境值。如果实际提前预测时只能提供天气预报或估计环境，应使用预测当时可获得的环境预报进行评估；目前指标不包含环境预测误差。若只能提供当前环境，本模型尚不构成指定提前量的自动未来预测器。",
              "同日多次记录仍按原文件顺序配对；缺少可靠时间戳时不能自动验证同日配对语义。未加入可能不可用的未来特征，也未人为构造上次偏差。", "",
              "## 模型结果", "", "| 模型 | CV MSE ± 折间标准差 | 留出 MSE | 留出 MAE | 留出 R² | CV 选中 |", "|---|---:|---:|---:|---:|---|"]
    for r in final.itertuples():
        report.append(f"| {r.model} | {r.cv_mse:.4f} ± {r.cv_mse_std:.4f} | {r.mse:.4f} | {r.mae:.4f} | {r.r2:.4f} | {'是' if r.selected_by_cv else '否'} |")
    report += ["", "负 R² 表示在该留出集上不如事后使用该集真实值均值的常数预测；mean_baseline 则只使用开发集均值，是实际可用的基线。与历史随机划分的指标因测试样本不同，不能直接用于判断优化幅度。", "",
               "![模型对比](model_comparison.png)", "", f"![所选模型预测]( {selected_name}/evaluation.png )".replace('( ', '(').replace(' )', ')'), "",
               "## 文件与复现", "",
               "- search_results.csv / cv_fold_metrics.csv：全部候选与逐折指标。",
               "- cv_splits.csv / split_manifest.csv：时间边界与 session 归属；selection.json：查看最终成绩前固定的选择。",
               "- 各模型目录：最终留出 predictions.csv、按弦/调音/日期误差 CSV 与图表。",
               "- evaluated_model.joblib：只用开发期训练，对应报告留出成绩。",
               "- future_model.joblib：完成评估后，用全部历史记录重训的同配置模型，供后续使用；其未来效果尚未评估。",
               "- 运行：`python src/guzheng_future_validation.py`（myenv 环境）。", ""]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(final.to_string(index=False), flush=True)
    print(f"完成：{OUT}", flush=True)


if __name__ == "__main__":
    main()
