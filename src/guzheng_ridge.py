# 1. 导入需要的库
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from matplotlib import font_manager
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score

font_path = "/Library/Fonts/Arial Unicode.ttf"
chinese_font = (font_manager.FontProperties(fname=font_path)
                if Path(font_path).is_file() else font_manager.FontProperties())
plt.rcParams["axes.unicode_minus"] = False

# 结果统一保存到项目根目录的 output 下，不受运行目录影响。
project_dir = Path(__file__).resolve().parent.parent
output_dir = project_dir / "output" / "without_guzheng_id" / "ridge" / "original_config"
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

# 16. 对特征进行标准化

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 17. 创建并训练岭回归模型

ridge = Ridge(alpha=1.0)

ridge.fit(X_train_scaled, y_train)

# 使用测试集进行预测
y_pred_ridge = ridge.predict(X_test_scaled)

# 18. 评估岭回归模型

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)



# 19. 使用 RidgeCV 自动寻找最佳 alpha

alphas = np.logspace(-3, 3, 50)

ridge_cv = RidgeCV(
    alphas=alphas,
    scoring="neg_mean_squared_error",
    cv=5
)

ridge_cv.fit(X_train_scaled, y_train)

best_alpha = ridge_cv.alpha_

# 20. 使用最佳 alpha 的 RidgeCV 进行预测

y_pred_cv = ridge_cv.predict(X_test_scaled)

mse_cv = mean_squared_error(y_test, y_pred_cv)
r2_cv = r2_score(y_test, y_pred_cv)

y_train_pred_cv = ridge_cv.predict(X_train_scaled)

train_r2_cv = r2_score(y_train, y_train_pred_cv)

# 21. 输出 RidgeCV 最佳模型结果

# print("\n========== RidgeCV 最佳模型 ==========")

# print(f"最佳 alpha: {best_alpha}")
# print(f"训练集 R2: {train_r2_cv:.4f}")
# print(f"测试集 R2: {r2_cv:.4f}")
# print(f"测试集 MSE: {mse_cv:.2f}")

# print("\n各特征系数：")

# for name, coef in zip(X.columns, ridge_cv.coef_):
#     print(f"{name}: {coef:.4f}")

# print(f"\n截距: {ridge_cv.intercept_:.4f}")

# 22. 绘制 RidgeCV 最佳模型：真实值 vs 预测值

plt.figure(figsize=(7, 7))

# 画测试集散点
plt.scatter(
    y_test,
    y_pred_cv,
    s=35,
    alpha=0.5
)

# 找到真实值和预测值的整体范围
min_value = min(y_test.min(), y_pred_cv.min())
max_value = max(y_test.max(), y_pred_cv.max())

# 画理想预测线 y = x
plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=2,
    label="理想预测线 y=x"
)

# 让横轴和纵轴范围一致
plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)

# 标题和坐标轴
plt.xlabel("真实偏差", fontproperties=chinese_font, fontsize=12)
plt.ylabel("预测偏差", fontproperties=chinese_font, fontsize=12)

plt.title(
    f"岭回归：真实值 vs 预测值（alpha={best_alpha:.3f}）",
    fontproperties=chinese_font,
    fontsize=14
)

# 在图上显示评价指标
plt.text(
    0.05,
    0.88,
    f"R² = {r2_cv:.4f}\nMSE = {mse_cv:.2f}",
    transform=plt.gca().transAxes,
    fontsize=11
)

plt.legend(prop=chinese_font)
plt.grid(True, alpha=0.3)

plt.tight_layout()
# 保存测试集预测结果与模型评估指标。
predictions = clean_data.loc[y_test.index, ["session_id", "日期", "古筝编号", "弦编号"]].copy()
predictions["真实偏差"] = y_test
predictions["预测偏差"] = y_pred_cv
predictions.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

pd.DataFrame([{
    "train_r2": train_r2_cv,
    "test_r2": r2_cv,
    "test_mse": mse_cv,
}]).to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")

plt.savefig(output_dir / "actual_vs_predicted.png", dpi=300, bbox_inches="tight")
print(f"结果已保存至：{output_dir}")
plt.show()
