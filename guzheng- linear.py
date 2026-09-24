# 1. 导入需要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import font_manager
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score

font_path = "/Library/Fonts/Arial Unicode.ttf"
chinese_font = font_manager.FontProperties(fname=font_path)
plt.rcParams["axes.unicode_minus"] = False

# 2. 读取数据
file_path = "/Users/angela/Desktop/古筝调音数据.xlsx"

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

# 16. 创建并训练线性回归模型

model = LinearRegression()

model.fit(X_train, y_train)


# 17. 用测试集进行预测

y_pred = model.predict(X_test)



mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 18. 评估线性回归模型

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 19. 查看线性回归模型的系数和截距

feature_names = X.columns

for name, coef in zip(feature_names, model.coef_):
    print(f"{name}: {coef:.4f}")

# 20. 绘制真实值 vs 预测值

plt.figure(figsize=(7, 7))

# 散点
plt.scatter(
    y_test,
    y_pred,
    s=35,
    alpha=0.5
)

# 找统一的坐标范围
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

# 理想预测线 y = x
plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=2,
    label="理想预测线 y=x"
)

# 让横纵坐标范围一致
plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)

# 中文标签
plt.xlabel("真实偏差", fontproperties=chinese_font, fontsize=12)
plt.ylabel("预测偏差", fontproperties=chinese_font, fontsize=12)
plt.title(
    "线性回归：真实值 vs 预测值",
    fontproperties=chinese_font,
    fontsize=14
)

# 把模型指标写在图中
plt.text(
    0.05,
    0.90,
    f"R² = {r2:.4f}\nMSE = {mse:.2f}",
    transform=plt.gca().transAxes,
    fontsize=11
)

plt.legend(prop=chinese_font)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 检查训练集和测试集的表现

y_train_pred = model.predict(X_train)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_pred)

print(f"训练集 R2: {train_r2:.4f}")
print(f"测试集 R2: {test_r2:.4f}")