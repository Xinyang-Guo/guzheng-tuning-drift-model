# 1. 导入需要的库
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from matplotlib import font_manager
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_squared_error, r2_score

font_path = "/Library/Fonts/Arial Unicode.ttf"
chinese_font = (font_manager.FontProperties(fname=font_path)
                if Path(font_path).is_file() else font_manager.FontProperties())
plt.rcParams["axes.unicode_minus"] = False

# 结果统一保存到项目根目录的 output 下，不受运行目录影响。
project_dir = Path(__file__).resolve().parent.parent
output_dir = project_dir / "output" / "decision_tree"
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
    ["湿度", "湿度变化", "温度", "温度变化", "弦编号", "古筝编号"]
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

# 16. 对特征进行标准化

# scaler = StandardScaler()

# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# 16. 创建并训练决策树回归模型

# tree = DecisionTreeRegressor(
#     max_depth=3,
#     random_state=42
# )

# tree.fit(X_train, y_train)

# # 17. 使用测试集进行预测

# y_pred_tree = tree.predict(X_test)

# # 18. 评估决策树回归模型

# mse_tree = mean_squared_error(y_test, y_pred_tree)
# r2_tree = r2_score(y_test, y_pred_tree)

# # 训练集预测
# y_train_pred_tree = tree.predict(X_train)
# train_r2_tree = r2_score(y_train, y_train_pred_tree)

# print("\n========== 决策树回归 max_depth=3 ==========")
# print(f"训练集 R2: {train_r2_tree:.4f}")
# print(f"测试集 R2: {r2_tree:.4f}")
# print(f"测试集 MSE: {mse_tree:.2f}")

# 19. 比较不同 max_depth 的决策树

print("\n========== 不同 max_depth 的比较 ==========")

for depth in range(1, 11):

    tree_test = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    # 训练
    tree_test.fit(X_train, y_train)

    # 预测训练集和测试集
    y_train_pred = tree_test.predict(X_train)
    y_test_pred = tree_test.predict(X_test)

    # 计算指标
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)

    # print(
    #     f"depth={depth:2d} | "
    #     f"Train R2={train_r2:.4f} | "
    #     f"Test R2={test_r2:.4f} | "
    #     f"MSE={mse:.2f}"
    # )

# 20. 为训练集准备 session 分组

groups_train = groups.iloc[train_index]

gkf = GroupKFold(n_splits=5)

# 21. 用 GroupKFold 比较不同 max_depth

# print("\n========== GroupKFold：不同 max_depth 的比较 ==========")

# depth_results = []

# for depth in range(1, 11):

#     fold_r2 = []
#     fold_mse = []

#     for fold_train_index, fold_val_index in gkf.split(
#         X_train,
#         y_train,
#         groups=groups_train
#     ):

#         # 当前这一折的训练数据
#         X_fold_train = X_train.iloc[fold_train_index]
#         y_fold_train = y_train.iloc[fold_train_index]

#         # 当前这一折的验证数据
#         X_fold_val = X_train.iloc[fold_val_index]
#         y_fold_val = y_train.iloc[fold_val_index]

#         # 建立当前深度的决策树
#         tree_cv = DecisionTreeRegressor(
#             max_depth=depth,
#             random_state=42
#         )

#         # 训练
#         tree_cv.fit(X_fold_train, y_fold_train)

#         # 验证集预测
#         y_fold_pred = tree_cv.predict(X_fold_val)

#         # 保存这一折的评价指标
#         fold_r2.append(
#             r2_score(y_fold_val, y_fold_pred)
#         )

#         fold_mse.append(
#             mean_squared_error(y_fold_val, y_fold_pred)
#         )

#     # 5折取平均
#     mean_r2 = np.mean(fold_r2)
#     mean_mse = np.mean(fold_mse)
#     std_r2 = np.std(fold_r2)
#     std_mse = np.std(fold_mse)

#     depth_results.append(
#         [depth, mean_r2, mean_mse]
#     )

#     print(
#     f"depth={depth:2d} | "
#     f"CV R2={mean_r2:.4f} ± {std_r2:.4f} | "
#     f"CV MSE={mean_mse:.2f} ± {std_mse:.2f}"
# )

# 22. 使用选择出的 max_depth=5 建立最终决策树

final_tree = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

final_tree.fit(X_train, y_train)

# 23. 最终决策树在训练集和测试集上预测

y_train_pred_final = final_tree.predict(X_train)
y_test_pred_final = final_tree.predict(X_test)

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

# print("\n========== 最终决策树 max_depth=5 ==========")
# print(f"训练集 R2: {train_r2_final:.4f}")
# print(f"测试集 R2: {test_r2_final:.4f}")
# print(f"测试集 MSE: {mse_final:.2f}")

# 24. 绘制最终决策树：真实值 vs 预测值

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

plt.xlabel(
    "真实偏差",
    fontproperties=chinese_font,
    fontsize=12
)

plt.ylabel(
    "预测偏差",
    fontproperties=chinese_font,
    fontsize=12
)

plt.title(
    "决策树回归：真实值 vs 预测值（max_depth=5）",
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