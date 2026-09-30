# 1. 导入需要的库
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from matplotlib import font_manager
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_squared_error, r2_score

font_path = "/Library/Fonts/Arial Unicode.ttf"
chinese_font = (font_manager.FontProperties(fname=font_path)
                if Path(font_path).is_file() else font_manager.FontProperties())
plt.rcParams["axes.unicode_minus"] = False

# 结果统一保存到项目根目录的 output 下，不受运行目录影响。
project_dir = Path(__file__).resolve().parent.parent
output_dir = project_dir / "output" / "without_guzheng_id" / "random_forest" / "original_config"
output_dir.mkdir(parents=True, exist_ok=True)

# 2. 读取数据
file_path = project_dir / "data" / "古筝调音数据.xlsx"

environment = pd.read_excel(file_path, sheet_name="Environment")
pitch = pd.read_excel(file_path, sheet_name="Pitch_Data")

# 3. 给 Pitch_Data 中每一次调音记录编号
pitch["session_id"] = np.arange(len(pitch)) // 21 + 1  # 每21行数据为一次调音记录，编号从1开始


# 4. 提取每一次实验的基本信息
session_info = pitch.groupby("session_id", as_index=False).first()

session_info = session_info[
    ["session_id", "日期", "古筝编号"]
]

# 5. 给同一天、同一台古筝的多次记录编号

session_info["record_no"] = session_info.groupby(
    ["日期", "古筝编号"]
).cumcount()

environment["record_no"] = environment.groupby(
    ["日期", "古筝编号"]
).cumcount()

# 6. 找出 Environment 中没有对应 Pitch_Data 的记录

check = environment.merge(
    session_info,
    on=["日期", "古筝编号", "record_no"],
    how="left",
    indicator=True
)
extra_environment = check[check["_merge"] == "left_only"]

# 7. 保留能够和 Pitch_Data 对应上的 Environment 记录
matched_environment = check[check["_merge"] == "both"].copy()

# 8. 把环境数据匹配到每一根弦

model_data = pitch.merge(
    matched_environment[
        ["session_id", "湿度", "湿度变化", "温度", "温度变化"]
    ],
    on="session_id",
    how="left"
)

# 9. 检查每一列的数据类型

# 找出湿度列里不能转换成数字的内容
humidity_numeric = pd.to_numeric(
    model_data["湿度"],
    errors="coerce"
)


# 10. 清理湿度列中的 *
model_data["湿度"] = model_data["湿度"].astype(str).str.replace("*", "", regex=False)

model_data["湿度"] = pd.to_numeric(
    model_data["湿度"],
    errors="coerce"
)

model_data["湿度变化"] = pd.to_numeric(
    model_data["湿度变化"],
    errors="coerce"
)

model_data["温度变化"] = pd.to_numeric(
    model_data["温度变化"],
    errors="coerce"
)

# 12. 查看缺失值来自哪些实验

missing_data = model_data[
    model_data["湿度变化"].isna() |
    model_data["温度变化"].isna()
]

# 13. 删除存在缺失值的行，得到用于建模的数据

clean_data = model_data.dropna(
    subset=["湿度变化", "温度变化"]
).copy()

# 14. 准备线性回归的特征 X 和目标 y

X = clean_data[
    ["湿度", "湿度变化", "温度", "温度变化", "弦编号"]
]

y = clean_data["偏差"]

# 15. 按 session 划分训练集和测试集

groups = clean_data["session_id"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.3,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

# 为训练集准备 session 分组，用于 GroupKFold

groups_train = groups.iloc[train_index]

gkf = GroupKFold(n_splits=5)

# 16. 创建并训练随机森林回归模型

forest = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)

forest.fit(X_train, y_train)

# 17. 使用训练集和测试集进行预测

y_train_pred_forest = forest.predict(X_train)
y_test_pred_forest = forest.predict(X_test)

# 18. 评估随机森林模型

train_r2_forest = r2_score(
    y_train,
    y_train_pred_forest
)

test_r2_forest = r2_score(
    y_test,
    y_test_pred_forest
)

mse_forest = mean_squared_error(
    y_test,
    y_test_pred_forest
)

# 19. 用 GroupKFold 比较随机森林不同的 max_depth

# 20. 固定 max_depth=7，比较不同 n_estimators

# print("\n========== Random Forest：不同 n_estimators 的比较 ==========")

# tree_numbers = [50, 100, 200, 300, 500]

# for n_tree in tree_numbers:

#     fold_r2 = []
#     fold_mse = []

#     for fold_train_index, fold_val_index in gkf.split(
#         X_train,
#         y_train,
#         groups=groups_train
#     ):

#         X_fold_train = X_train.iloc[fold_train_index]
#         y_fold_train = y_train.iloc[fold_train_index]

#         X_fold_val = X_train.iloc[fold_val_index]
#         y_fold_val = y_train.iloc[fold_val_index]

#         forest_cv = RandomForestRegressor(
#             n_estimators=n_tree,
#             max_depth=7,
#             random_state=42,
#             n_jobs=-1
#         )

#         forest_cv.fit(X_fold_train, y_fold_train)

#         y_fold_pred = forest_cv.predict(X_fold_val)

#         fold_r2.append(
#             r2_score(y_fold_val, y_fold_pred)
#         )

#         fold_mse.append(
#             mean_squared_error(y_fold_val, y_fold_pred)
#         )

#     mean_r2 = np.mean(fold_r2)
#     std_r2 = np.std(fold_r2)

#     mean_mse = np.mean(fold_mse)
#     std_mse = np.std(fold_mse)

#     print(
#         f"n_estimators={n_tree:3d} | "
#         f"CV R2={mean_r2:.4f} ± {std_r2:.4f} | "
#         f"CV MSE={mean_mse:.2f} ± {std_mse:.2f}"
#     )

# 21. 使用最佳参数建立最终随机森林

final_forest = RandomForestRegressor(
    n_estimators=500,
    max_depth=7,
    random_state=42,
    n_jobs=-1
)

final_forest.fit(X_train, y_train)

# 22. 在训练集和测试集上进行预测

y_train_pred_final = final_forest.predict(X_train)
y_test_pred_final = final_forest.predict(X_test)

# 23. 评估最终随机森林

train_r2_final = r2_score(
    y_train,
    y_train_pred_final
)

test_r2_final = r2_score(
    y_test,
    y_test_pred_final
)

mse_final = mean_squared_error(
    y_test,
    y_test_pred_final
)

# print("\n========== 最终随机森林 ==========")
# print("n_estimators = 500")
# print("max_depth = 7")
# print(f"训练集 R2: {train_r2_final:.4f}")
# print(f"测试集 R2: {test_r2_final:.4f}")
# print(f"测试集 MSE: {mse_final:.2f}")

# 24. 绘制最终随机森林：真实值 vs 预测值

plt.figure(figsize=(7, 7))

plt.scatter(
    y_test,
    y_test_pred_final,
    s=35,
    alpha=0.5
)

min_value = min(y_test.min(), y_test_pred_final.min())
max_value = max(y_test.max(), y_test_pred_final.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=2,
    label="理想预测线 y=x"
)

plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)

plt.xlabel("真实偏差", fontproperties=chinese_font, fontsize=12)
plt.ylabel("预测偏差", fontproperties=chinese_font, fontsize=12)

plt.title(
    "随机森林：真实值 vs 预测值",
    fontproperties=chinese_font,
    fontsize=14
)

plt.text(
    0.05,
    0.88,
    f"R² = {test_r2_final:.4f}\nMSE = {mse_final:.2f}",
    transform=plt.gca().transAxes,
    fontsize=11
)

plt.legend(prop=chinese_font)
plt.grid(True, alpha=0.3)

plt.tight_layout()
# 保存测试集预测结果与模型评估指标。
predictions = clean_data.loc[y_test.index, ["session_id", "日期", "古筝编号", "弦编号"]].copy()
predictions["真实偏差"] = y_test
predictions["预测偏差"] = y_test_pred_final
predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

pd.DataFrame([{
    "train_r2": train_r2_final,
    "test_r2": test_r2_final,
    "test_mse": mse_final,
}]).to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")

plt.savefig(output_dir / "actual_vs_predicted.png", dpi=300, bbox_inches="tight")
print(f"结果已保存至：{output_dir}")
plt.show()

# 25. 查看随机森林的特征重要性

# feature_importance = final_forest.feature_importances_

# print("\n========== 随机森林特征重要性 ==========")

# for name, importance in zip(X.columns, feature_importance):
#     print(f"{name}: {importance:.4f}")

# 26. 查看湿度变化的数据范围

# print("\n湿度变化范围：")
# print("最小值：", clean_data["湿度变化"].min())
# print("最大值：", clean_data["湿度变化"].max())
# print("平均值：", clean_data["湿度变化"].mean())
# print("中位数：", clean_data["湿度变化"].median())

# # 27. 建立湿度变化的连续取值

# humidity_change_grid = np.linspace(
#     X_train["湿度变化"].min(),
#     X_train["湿度变化"].max(),
#     200
# )

# print(humidity_change_grid[:5])
# print(humidity_change_grid[-5:])

# # 28. 计算湿度变化对预测偏差的平均影响

# humidity_response = []

# for humidity_change in humidity_change_grid:

#     X_temp = X_train.copy()

#     X_temp["湿度变化"] = humidity_change

#     y_temp_pred = final_forest.predict(X_temp)

#     humidity_response.append(
#         np.mean(y_temp_pred)
#     )

# # 29. 绘制湿度变化响应曲线

# plt.figure(figsize=(8, 6))

# # 原始数据散点
# plt.scatter(
#     X_train["湿度变化"],
#     y_train,
#     s=20,
#     alpha=0.15,
#     label="实际数据"
# )

# # 随机森林预测响应曲线
# plt.plot(
#     humidity_change_grid,
#     humidity_response,
#     linewidth=2.5,
#     label="随机森林平均预测"
# )

# # 湿度变化为0的参考线
# plt.axvline(
#     x=0,
#     linestyle="--",
#     linewidth=1.5,
#     label="湿度无变化"
# )

# plt.xlabel(
#     "湿度变化",
#     fontproperties=chinese_font,
#     fontsize=12
# )

# plt.ylabel(
#     "调音偏差",
#     fontproperties=chinese_font,
#     fontsize=12
# )

# plt.title(
#     "湿度变化与古筝调音偏差的随机森林响应曲线",
#     fontproperties=chinese_font,
#     fontsize=14
# )

# plt.legend(prop=chinese_font)

# plt.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# # 30. 查看温度变化的数据范围

# print("\n温度变化范围：")
# print("最小值：", clean_data["温度变化"].min())
# print("最大值：", clean_data["温度变化"].max())
# print("平均值：", clean_data["温度变化"].mean())
# print("中位数：", clean_data["温度变化"].median())

# # 31. 建立温度变化的连续取值

# temperature_change_grid = np.linspace(
#     X_train["温度变化"].min(),
#     X_train["温度变化"].max(),
#     200
# )

# # 32. 计算温度变化对预测偏差的平均影响

# temperature_response = []

# for temperature_change in temperature_change_grid:

#     X_temp = X_train.copy()

#     X_temp["温度变化"] = temperature_change

#     y_temp_pred = final_forest.predict(X_temp)

#     temperature_response.append(
#         np.mean(y_temp_pred)
#     )

# # 33. 绘制温度变化响应曲线

# plt.figure(figsize=(8, 6))

# # 原始数据散点
# plt.scatter(
#     X_train["温度变化"],
#     y_train,
#     s=20,
#     alpha=0.15,
#     label="实际数据"
# )

# # 随机森林平均预测响应曲线
# plt.plot(
#     temperature_change_grid,
#     temperature_response,
#     linewidth=2.5,
#     label="随机森林平均预测"
# )

# # 温度变化为0的参考线
# plt.axvline(
#     x=0,
#     linestyle="--",
#     linewidth=1.5,
#     label="温度无变化"
# )

# plt.xlabel(
#     "温度变化（℃）",
#     fontproperties=chinese_font,
#     fontsize=12
# )

# plt.ylabel(
#     "调音偏差",
#     fontproperties=chinese_font,
#     fontsize=12
# )

# plt.title(
#     "温度变化与古筝调音偏差的随机森林响应曲线",
#     fontproperties=chinese_font,
#     fontsize=14
# )

# plt.legend(prop=chinese_font)
# plt.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# 34. 定义古筝调音偏差预测函数

def predict_deviation(
    humidity,
    humidity_change,
    temperature,
    temperature_change,
    string_no,
):

    new_data = pd.DataFrame(
        [[
            humidity,
            humidity_change,
            temperature,
            temperature_change,
            string_no,
        ]],
        columns=[
            "湿度",
            "湿度变化",
            "温度",
            "温度变化",
            "弦编号",
        ]
    )

    prediction = final_forest.predict(new_data)

    return prediction[0]

# 35. 测试预测函数

result = predict_deviation(
    humidity=0.70,
    humidity_change=0.10,
    temperature=26.0,
    temperature_change=1.0,
    string_no=8,
)

print("\n预测调音偏差：")
print(result)

# 36. 一次预测 21 根弦

def predict_all_strings(
    humidity,
    humidity_change,
    temperature,
    temperature_change,
):

    results = []

    for string_no in range(1, 22):

        prediction = predict_deviation(
            humidity=humidity,
            humidity_change=humidity_change,
            temperature=temperature,
            temperature_change=temperature_change,
            string_no=string_no,
        )

        results.append(
            [string_no, prediction]
        )

    result_table = pd.DataFrame(
        results,
        columns=["弦编号", "预测偏差"]
    )

    return result_table

# 37. 测试 21 根弦预测

all_predictions = predict_all_strings(
    humidity=0.70,
    humidity_change=0.10,
    temperature=26.0,
    temperature_change=1.0,
)

print("\n21根弦预测结果：")
print(all_predictions)

all_predictions.to_csv(
    output_dir / "all_strings_predictions.csv", index=False, encoding="utf-8-sig"
)
