import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 載入資料
data_path = r"E:\MRI\HW2\VBM_IXI_Data.csv"
data = pd.read_csv(data_path)

# 移除NaN資料
data = data.dropna(subset=['AGE'])

# 特徵選擇（只使用 ModulatedGM 或 SmoothedGM）
modulated_gm_data = data[['ModulatedGM', 'AGE']]
smoothed_gm_data = data[['SmoothedGM', 'AGE']]

# 切分訓練集與測試集
X_modulated = modulated_gm_data[['ModulatedGM']]
y = modulated_gm_data['AGE']
X_train_mod, X_test_mod, y_train, y_test = train_test_split(X_modulated, y, test_size=0.2, random_state=42)

X_smoothed = smoothed_gm_data[['SmoothedGM']]
X_train_smooth, X_test_smooth, y_train_smooth, y_test_smooth = train_test_split(X_smoothed, y, test_size=0.2, random_state=42)

# 訓練模型：ModulatedGM
model_modulated = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5, min_samples_split=10)
model_modulated.fit(X_train_mod, y_train)
y_pred_mod = model_modulated.predict(X_test_mod)

# 訓練模型：SmoothedGM
model_smoothed = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5, min_samples_split=10)
model_smoothed.fit(X_train_smooth, y_train_smooth)
y_pred_smooth = model_smoothed.predict(X_test_smooth)

# 計算誤差
mae_mod = mean_absolute_error(y_test, y_pred_mod)
mse_mod = mean_squared_error(y_test, y_pred_mod)
rmse_mod = np.sqrt(mse_mod)

mae_smooth = mean_absolute_error(y_test_smooth, y_pred_smooth)
mse_smooth = mean_squared_error(y_test_smooth, y_pred_smooth)
rmse_smooth = np.sqrt(mse_smooth)

# 打印誤差指標
print(f"ModulatedGM - MAE: {mae_mod}, MSE: {mse_mod}, RMSE: {rmse_mod}")
print(f"SmoothedGM - MAE: {mae_smooth}, MSE: {mse_smooth}, RMSE: {rmse_smooth}")

# 評估模型：ModulatedGM
mae = mean_absolute_error(y_test, y_pred_mod)
r2 = r2_score(y_test, y_pred_mod)
print(f"ModulatedGM - MAE: {mae}")
print(f"ModulatedGM - R² Score: {r2}")

# 評估模型：SmoothedGM
mae_smooth = mean_absolute_error(y_test_smooth, y_pred_smooth)
r2_smooth = r2_score(y_test_smooth, y_pred_smooth)
print(f"SmoothedGM - MAE: {mae_smooth}")
print(f"SmoothedGM - R² Score: {r2_smooth}")

# 繪製比較圖
plt.figure(figsize=(18, 6))

# ModulatedGM 預測結果
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_mod, alpha=0.6, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('ModulatedGM: Predicted vs Actual Age')
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')

# SmoothedGM 預測結果
plt.subplot(1, 3, 2)
plt.scatter(y_test_smooth, y_pred_smooth, alpha=0.6, color='green')
plt.plot([min(y_test_smooth), max(y_test_smooth)], [min(y_test_smooth), max(y_test_smooth)], color='red', linestyle='--')
plt.title('SmoothedGM: Predicted vs Actual Age')
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')

# 預測年齡與實際年齡比較圖
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_mod, alpha=0.6, color='blue', label='ModulatedGM')
plt.scatter(y_test_smooth, y_pred_smooth, alpha=0.6, color='green', label='SmoothedGM')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Predicted vs Actual Age')
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.legend()

# 顯示圖表
plt.tight_layout()
plt.show()
