# 古筝调音预测：实验记录与结果

**网页版：[听弦小筑 · 21 弦跑音预测](https://xinyang-guo.github.io/guzheng-tuning-drift-model/)**

输入预计的温度、湿度及变化量，可选择岭回归或随机森林，查看 21 根弦的预测结果。网页源码见 [docs/index.html](docs/index.html)，按时间划分的模型指标见 [时间验证汇总](results/time_validation_metrics.csv)。部署方法见 [DEPLOY.md](DEPLOY.md)。仓库公开并在 GitHub Pages 中设置 `main` 分支的 `/docs` 目录后，网页版链接才会生效；具体网址以 Pages 设置页显示的地址为准。

这个项目尝试用温度、湿度及其变化等信息预测古筝的音高偏差。我先比较了线性回归、岭回归、决策树和随机森林，随后调整参数，并检查去掉古筝编号后结果会有什么变化。最近把用途明确为预测未来调音，又补做了一轮按时间划分的验证。

下面先放时间验证的结果，再保留之前的随机划分实验。两轮实验用了不同的测试集，需要分开看。

## 目前的选择：用岭回归做后续预测

这一轮暂时选用五特征岭回归，配置为 `StandardScaler + Ridge`，`alpha=316.227766`。它是在开发期的四折滚动验证中选出来的，不是根据最后留出的数据选出来的。前面几轮实验中随机森林表现最好，指的是后文的历史随机划分结果。

既然要预测未来，就不能把预测时还不知道的环境实测值当作输入。实际使用时，这些输入只能来自天气预报或提前作出的环境估计。

这次按同一年内的月、日顺序划分数据：把最后约 20% 的完整日期留作最终评估，较早的日期用于四折扩展窗口验证。同一天的记录不会被拆到时间边界两侧，同一个调音 session 的记录也会放在一起。

### 验证结果

| 模型 | 开发期 CV MSE | 最终日期留出 MSE | MAE | R² |
|---|---:|---:|---:|---:|
| ridge（CV 选中） | 76.9602 | 74.5554 | 6.5324 | 0.5960 |
| mean_baseline | 82.0918 | 209.0188 | 10.7603 | -0.1325 |
| random_forest | 82.6588 | 179.5212 | 10.0396 | 0.0273 |

共比较了 25 组随机森林配置、13 个岭回归 alpha 值和 1 个均值基线，也就是 39 个候选、156 次滚动验证拟合。选模型时只看开发期的 CV MSE，最终留出数据用于评估，不参与选择。

随机森林在这轮搜索中固定为 300 棵树。搜索结果中表现最好的随机森林配置是：最大深度 4、叶节点最少 20 个样本、`max_features=0.6`、`min_samples_split=2`。

### 这些结果还不能说明什么

目前表里的最终留出成绩仍然使用了实际环境记录，还没有把真实天气预报的误差算进去。因此，它还不能直接代表提前预测调音偏差时的效果。

另外做了湿度 ±5 个百分点、温度 ±2℃ 的敏感性分析。这只是用来观察输入变化时预测会怎样变化的示例情景，不是天气预报的准确率，也不是预测的置信区间。实际准备输入时，需要估计古筝所在的室内环境，不能直接把室外天气预报当成室内实测值。

四折验证之间的结果波动较大，而且这批记录已经用于前面的历史实验。下一步仍然需要新采集日期的数据，检验模型在真正的前瞻预测中表现如何。

时间验证和历史随机划分的测试集不同，不能把两边的指标直接相减，就说模型提高了多少。

### 报告和模型文件

- [时间验证完整报告](output/future_validation/report.md)
- [39 个候选搜索结果](output/future_validation/search_results.csv) · [逐折结果](output/future_validation/cv_fold_metrics.csv) · [最终留出指标](output/future_validation/final_metrics.csv)
- [时间划分](output/future_validation/cv_splits.csv) · [Session 归属](output/future_validation/split_manifest.csv)
- [模型对比图](output/future_validation/model_comparison.png) · [岭回归预测图](output/future_validation/ridge/evaluation.png) · [环境估计敏感性图](output/future_validation/environment_sensitivity.png)
- [已评估模型](output/future_validation/evaluated_model.joblib)：只用开发期数据拟合，对应上面的报告成绩。
- [后续预测模型](output/future_validation/future_model.joblib)：固定选好的参数后，用全部历史数据重新训练。它在未来新记录上的效果还没有评估。

### 重跑实验和使用模型

重跑这轮实验：

```bash
conda activate myenv
python src/guzheng_future_validation.py
python src/guzheng_forecast_sensitivity.py
```

做后续预测时，输入列必须依次为 `湿度、湿度变化、温度、温度变化、弦编号`。湿度使用 0～1 的数值，湿度变化和温度变化的计算方式要与训练数据一致。不需要输入古筝编号。

模型本身不会获取天气预报，目前也没有建立针对指定提前量的环境预测模型。需要先准备好环境预报或估计值，再交给模型预测。

```python
import joblib
import pandas as pd

model = joblib.load("output/future_validation/future_model.joblib")
# 使用预测发出时可获得的室内环境估计构建这五列；每根弦一行。
features = ["湿度", "湿度变化", "温度", "温度变化", "弦编号"]
inputs = pd.read_csv("future_environment_estimates.csv")
inputs["预测偏差"] = model.predict(inputs[features])
inputs.to_csv("output/future_predictions.csv", index=False, encoding="utf-8-sig")
```

这里的 `future_environment_estimates.csv` 需要自行用实际环境预报或估计值准备。项目没有生成虚构的预报文件。

---

## 前几轮随机划分实验：随机森林表现最好

这部分保留的是 18 组历史实验。它们使用相同的测试集，主要按测试 MSE 比较，同时参考 R²；这两个指标选出的最好结果一致。

在这 18 组实验里，保留古筝编号的随机森林表现最好：100 棵树、最大深度 5、`random_state=42`，测试 R² 为 0.472979，MSE 为 68.019537。

不使用古筝编号时，表现最好的仍然是相同配置的随机森林：100 棵树、最大深度 5、`random_state=42`，测试 R² 为 0.467383，MSE 为 68.741866。与含编号版本相比，MSE 增加了 0.722329（1.06%），R² 下降了 0.005597。

这里的“最好”只限于这次固定测试集上的已运行实验。这个测试集已经用于多轮比较，还没有用独立的新测试集确认结果，也不能据此说这就是所有参数中最好的配置。

## 数据与实验方法（历史随机划分）

数据来自 [古筝调音数据.xlsx](古筝调音数据.xlsx)，读取其中的 `Environment` 和 `Pitch_Data` 工作表。

每 21 行音高记录作为一次调音 session，按日期、古筝编号和记录序号与环境数据关联。清理时去掉湿度中的星号，并删除湿度变化或温度变化缺失的记录。

清理后共有 1,344 条有效记录。这轮随机划分中，训练集有 924 条，来自 44 个 session；测试集有 420 条，来自 20 个 session。划分使用 `GroupShuffleSplit`，按 `session_id` 分组，设置为 `test_size=0.3`、`random_state=42`。

18 组实验的测试记录、顺序和真实偏差已经逐一核对，确认一致。R² 和 MSE 也已根据预测 CSV 重新计算核对。

六特征版本使用湿度、湿度变化、温度、温度变化、弦编号和古筝编号。五特征版本只去掉古筝编号；编号仍然保留在数据关联和结果追溯中，只是不再作为模型输入。

岭回归使用 `StandardScaler`，标准化参数只在训练集上拟合。随机森林和决策树的 `random_state` 都设为 42。

这里的 R² 越高越好，MSE 越低越好。MSE 的单位是目标“偏差”单位的平方；报告沿用数据中的“偏差”名称，不另外假定它的单位。

## 历史 18 组结果（按测试 MSE 排序）

下表按测试 MSE 从低到高排列。每行末尾的链接指向对应实验目录，里面有指标、逐条预测和预测散点图。

| 排名 | 模型 | 实验 | 特征数 | 参数 | 训练 R² | 测试 R² | 测试 MSE | 结果 |
|---:|---|---|---:|---|---:|---:|---:|---|
| 1 | 随机森林 | 含编号·参数实验 | 6 | n_estimators=100, max_depth=5 | 0.796837 | 0.472979 | 68.019537 | [文件](output/parameter_comparison/random_forest/run_1/) |
| 2 | 随机森林 | 含编号·参数实验 | 6 | n_estimators=300, max_depth=7 | 0.877865 | 0.471689 | 68.186130 | [文件](output/parameter_comparison/random_forest/run_2/) |
| 3 | 随机森林 | 初始四模型 | 6 | n_estimators=500, max_depth=7 | 0.877793 | 0.467511 | 68.725336 | [文件](output/random_forest/) |
| 4 | 随机森林 | 无编号·参数实验 | 5 | n_estimators=100, max_depth=5 | 0.771654 | 0.467383 | 68.741866 | [文件](output/without_guzheng_id/random_forest/run_1/) |
| 5 | 随机森林 | 含编号·参数实验 | 6 | n_estimators=500, max_depth=9 | 0.922446 | 0.460058 | 69.687278 | [文件](output/parameter_comparison/random_forest/run_3/) |
| 6 | 随机森林 | 无编号·原配置 | 5 | n_estimators=500, max_depth=7 | 0.836985 | 0.457212 | 70.054553 | [文件](output/without_guzheng_id/random_forest/original_config/) |
| 7 | 随机森林 | 无编号·参数实验 | 5 | n_estimators=300, max_depth=7 | 0.836767 | 0.457029 | 70.078205 | [文件](output/without_guzheng_id/random_forest/run_2/) |
| 8 | 岭回归 | 无编号·参数实验 | 5 | alpha=10.0 | 0.541965 | 0.450509 | 70.919686 | [文件](output/without_guzheng_id/ridge/run_3/) |
| 9 | 岭回归 | 无编号·参数实验 | 5 | alpha=1.0 | 0.542018 | 0.448457 | 71.184551 | [文件](output/without_guzheng_id/ridge/run_2/) |
| 10 | 岭回归 | 含编号·参数实验 | 6 | alpha=10.0 | 0.547277 | 0.448449 | 71.185517 | [文件](output/parameter_comparison/ridge/run_3/) |
| 11 | 岭回归 | 无编号·参数实验 | 5 | alpha=0.1 | 0.542019 | 0.448235 | 71.213135 | [文件](output/without_guzheng_id/ridge/run_1/) |
| 12 | 岭回归 | 无编号·原配置 | 5 | RidgeCV：50 个 alpha，10⁻³～10³，cv=5 | 0.542019 | 0.448211 | 71.216304 | [文件](output/without_guzheng_id/ridge/original_config/) |
| 13 | 岭回归 | 含编号·参数实验 | 6 | alpha=1.0 | 0.547323 | 0.447543 | 71.302435 | [文件](output/parameter_comparison/ridge/run_2/) |
| 14 | 岭回归 | 含编号·参数实验 | 6 | alpha=0.1 | 0.547324 | 0.447444 | 71.315241 | [文件](output/parameter_comparison/ridge/run_1/) |
| 15 | 岭回归 | 初始四模型 | 6 | RidgeCV：50 个 alpha，10⁻³～10³，cv=5 | 0.547324 | 0.447433 | 71.316663 | [文件](output/ridge/) |
| 16 | 线性回归 | 初始四模型 | 6 | LinearRegression（默认参数） | 0.547324 | 0.447433 | 71.316677 | [文件](output/linear/) |
| 17 | 随机森林 | 无编号·参数实验 | 5 | n_estimators=500, max_depth=9 | 0.872564 | 0.442387 | 71.967921 | [文件](output/without_guzheng_id/random_forest/run_3/) |
| 18 | 决策树 | 初始四模型 | 6 | max_depth=5 | 0.753177 | 0.434921 | 72.931547 | [文件](output/decision_tree/) |

两次 `RidgeCV` 运行没有把选中的 alpha 保存到指标文件里，所以表中只列了搜索范围和方法，没有补写一个推测值。后面的参数实验使用固定的 `alpha=0.1、1、10`，与 `RidgeCV` 自动选择 alpha 不是同一回事。

## 各轮实验里观察到了什么

**最初的四个模型。** 使用六个特征时，随机森林（500 棵树、深度 7）的测试 MSE 最低。线性回归和 `RidgeCV` 的结果非常接近。

**保留古筝编号，调整参数。** 三组随机森林中，100 棵树、深度 5 的组合表现最好。增加树的数量、同时加深树后，训练 R² 上升，测试表现却下降了，过拟合的迹象更明显。岭回归的 alpha 从 0.1 增到 10 时，测试误差略有下降。

**去掉古筝编号，再比较相同的参数组合。** 随机森林三组的测试 MSE 都升高了，岭回归三组则都降低了。五特征版本中，表现最好的仍然是 100 棵树、深度 5 的随机森林。

**按原配置重跑五特征版本。** 另外保留了随机森林（500 棵树、深度 7）和 `RidgeCV` 的结果，在总表中标为“无编号·原配置”。随机森林对 21 根弦的示例预测也已保存。

这些比较还有几个限制。随机森林的三组参数实验同时改了树的数量和深度，所以不能单独判断是哪一个参数造成了变化。所有历史比较也只用了同一次固定的 session 划分。按 session 分组不等于按古筝分组，训练集和测试集并不保证来自不同的古筝，因此还不能用这些结果证明模型能适用于一台新的古筝。

原来的 `RidgeCV` 使用 `cv=5`，内部交叉验证没有按 session 分组。后续若在这类分组实验中继续严格调参，应把标准化放进 `Pipeline`，在训练集内使用 `GroupKFold`，最后再用独立测试数据确认。

## 图表

### 历史五特征实验中表现最好的随机森林（100 棵树、深度 5）

![历史五特征随机森林预测图](output/without_guzheng_id/random_forest/run_1/actual_vs_predicted.png)

### 去掉古筝编号前后的对比

![移除编号前后对比](output/without_guzheng_id/before_after_comparison.png)

### 三组参数的对比

| 实验 | 指标图 | 三组预测散点图 |
|---|---|---|
| 含编号·随机森林 | [指标对比](output/parameter_comparison/random_forest/metrics_comparison.png) | [预测对比](output/parameter_comparison/random_forest/prediction_comparison.png) |
| 含编号·岭回归 | [指标对比](output/parameter_comparison/ridge/metrics_comparison.png) | [预测对比](output/parameter_comparison/ridge/prediction_comparison.png) |
| 无编号·随机森林 | [指标对比](output/without_guzheng_id/random_forest/metrics_comparison.png) | [预测对比](output/without_guzheng_id/random_forest/prediction_comparison.png) |
| 无编号·岭回归 | [指标对比](output/without_guzheng_id/ridge/metrics_comparison.png) | [预测对比](output/without_guzheng_id/ridge/prediction_comparison.png) |

### 最初四个模型的预测图

| 线性回归 | 岭回归 | 决策树 | 随机森林 |
|---|---|---|---|
| [预测图](output/linear/actual_vs_predicted.png) | [预测图](output/ridge/actual_vs_predicted.png) | [预测图](output/decision_tree/actual_vs_predicted.png) | [预测图](output/random_forest/actual_vs_predicted.png) |

## 输出文件在哪里

- 初始四模型：[linear](output/linear/)、[ridge](output/ridge/)、[decision_tree](output/decision_tree/)、[random_forest](output/random_forest/)。
- 含编号的六组参数实验：[总指标](output/parameter_comparison/comparison.csv)、[实验说明](output/parameter_comparison/README.md)。
- 无编号的六组参数实验：[总指标](output/without_guzheng_id/comparison.csv)、[前后差异 CSV](output/without_guzheng_id/before_after_comparison.csv)、[详细报告](output/without_guzheng_id/report.md)。
- 无编号的原配置结果：[随机森林](output/without_guzheng_id/random_forest/original_config/)、[岭回归](output/without_guzheng_id/ridge/original_config/)。
- 21 根弦的示例预测：[含编号](output/random_forest/all_strings_predictions.csv)、[无编号](output/without_guzheng_id/random_forest/original_config/all_strings_predictions.csv)。这些预测使用的是原配置模型和示例输入，没有对应的真实值，不参与模型排名。

每组历史实验的 `metrics.csv` 保存指标，`predictions.csv` 保存 420 条测试预测，`actual_vs_predicted.png` 是真实值与预测值的散点图。

## 如何运行，以及当前脚本使用什么配置

使用的 conda 环境是 `myenv`，依赖 `numpy`、`pandas`、`matplotlib`、`scikit-learn` 和 `openpyxl`。

```bash
conda activate myenv
python src/guzheng_parameter_comparison.py
```

这条命令会重跑当前五特征版本的六组参数实验，并生成去掉编号前后的对比报告。报告需要用到保留在 `output/parameter_comparison` 中的含编号结果。

当前两个独立脚本 `guzheng_random forest.py` 和 `guzheng_ridge.py` 也已经去掉古筝编号。结果写入 `output/without_guzheng_id` 下对应模型的 `original_config` 目录。

这两个脚本分别使用随机森林的 500 棵树、深度 7 配置，以及 `RidgeCV`。其中随机森林的配置不是历史五特征实验中表现最好的 100 棵树、深度 5；后者位于参数对比脚本 `CONFIGS` 的随机森林第一组。用于后续预测的当前选择，则是本文开头时间验证选出的五特征岭回归。

含编号的结果作为历史输出保留，直接运行当前的随机森林和岭回归脚本不会覆盖它们。历史输出只有预测和指标；这次时间验证另外保存了可直接加载的模型文件，链接见本文前面的“报告和模型文件”。
