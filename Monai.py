import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from monai.transforms import (
    Compose,
    ScaleIntensity,
    Resize,
    ToTensor,
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import Regressor
from torch.nn import MSELoss
from torch.optim import AdamW

# 檢查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 將數據轉換為 3D 張量
def simulate_3d_data(value):
    return np.expand_dims(np.ones((96, 96, 96)) * value, axis=0)

# 單模型訓練函數（使用 MONAI）
def train_monai_model(feature_name, train_data, val_data, train_labels, val_labels, save_model_name):
    transforms = Compose([ScaleIntensity(), Resize((96, 96, 96)), ToTensor()])
    train_ds = Dataset(
        data=[{"image": img, "label": label} for img, label in zip(train_data, train_labels)],
        transform=lambda x: {
            "image": transforms(torch.tensor(x["image"], dtype=torch.float32)),
            "label": torch.tensor(x["label"], dtype=torch.float32),
        },
    )
    val_ds = Dataset(
        data=[{"image": img, "label": label} for img, label in zip(val_data, val_labels)],
        transform=lambda x: {
            "image": transforms(torch.tensor(x["image"], dtype=torch.float32)),
            "label": torch.tensor(x["label"], dtype=torch.float32),
        },
    )
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    model = Regressor(in_shape=(1, 96, 96, 96), out_shape=1, channels=(16, 32, 64, 128), strides=(2, 2, 2, 2))
    model.to(device)
    loss_function = MSELoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_rmse = float("inf")
    val_labels_final, val_preds_final = [], []

    for epoch in range(20):
        model.train()
        train_loss = 0
        for batch_data in train_loader:
            images = batch_data["image"].to(device)
            labels = batch_data["label"].to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_labels, val_preds = [], []
        with torch.no_grad():
            for val_data in val_loader:
                val_images = val_data["image"].to(device)
                val_labels_batch = val_data["label"].to(device).unsqueeze(1)
                val_outputs = model(val_images)
                val_labels.extend(val_labels_batch.cpu().numpy())
                val_preds.extend(val_outputs.cpu().numpy())
        val_labels = np.array(val_labels)
        val_preds = np.array(val_preds).squeeze()

        if rmse := np.sqrt(mean_squared_error(val_labels, val_preds)) < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), save_model_name)
        val_labels_final, val_preds_final = val_labels, val_preds

    return val_labels_final, val_preds_final

# 數據加載與處理
if __name__ == "__main__":
    data_path = r"E:\MRI\HW2\VBM_IXI_Data.csv"
    data = pd.read_csv(data_path).dropna(subset=["AGE", "ModulatedGM", "SmoothedGM"])

    modulated_values = data["ModulatedGM"].values
    smoothed_values = data["SmoothedGM"].values
    ages = data["AGE"].values

    modulated_3d_data = np.array([simulate_3d_data(v) for v in modulated_values])
    smoothed_3d_data = np.array([simulate_3d_data(v) for v in smoothed_values])

    train_data_mod, val_data_mod, train_labels_mod, val_labels_mod = train_test_split(
        modulated_3d_data, ages, test_size=0.2, random_state=42
    )
    train_data_smooth, val_data_smooth, train_labels_smooth, val_labels_smooth = train_test_split(
        smoothed_3d_data, ages, test_size=0.2, random_state=42
    )

    y_test_mod, y_pred_mod = train_monai_model("ModulatedGM", train_data_mod, val_data_mod, train_labels_mod, val_labels_mod, "best_model_modulated.pth")
    y_test_smooth, y_pred_smooth = train_monai_model("SmoothedGM", train_data_smooth, val_data_smooth, train_labels_smooth, val_labels_smooth, "best_model_smoothed.pth")

    # 計算誤差
    mae_mod = mean_absolute_error(y_test_mod, y_pred_mod)
    mse_mod = mean_squared_error(y_test_mod, y_pred_mod)
    rmse_mod = np.sqrt(mse_mod)

    mae_smooth = mean_absolute_error(y_test_smooth, y_pred_smooth)
    mse_smooth = mean_squared_error(y_test_smooth, y_pred_smooth)
    rmse_smooth = np.sqrt(mse_smooth)
    # 打印誤差指標
    print(f"ModulatedGM - MAE: {mae_mod}, MSE: {mse_mod}, RMSE: {rmse_mod}")
    print(f"SmoothedGM - MAE: {mae_smooth}, MSE: {mse_smooth}, RMSE: {rmse_smooth}")

    # 評估模型：ModulatedGM
    mae = mean_absolute_error(y_test_mod, y_pred_mod)
    r2 = r2_score(y_test_mod, y_pred_mod)
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
    plt.scatter(y_test_mod, y_pred_mod, alpha=0.6, color='blue')
    plt.plot([min(y_test_mod), max(y_test_mod)], [min(y_test_mod), max(y_test_mod)], color='red', linestyle='--')
    plt.title("ModulatedGM: Predicted vs Actual Age")
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    # SmoothedGM 預測結果
    plt.subplot(1, 3, 2)
    plt.scatter(y_test_smooth, y_pred_smooth, alpha=0.6, color='green')
    plt.plot([min(y_test_smooth), max(y_test_smooth)], [min(y_test_smooth), max(y_test_smooth)], color='red', linestyle='--')
    plt.title("SmoothedGM: Predicted vs Actual Age")
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")

    # 預測年齡與實際年齡比較圖
    plt.subplot(1, 3, 3)
    plt.scatter(y_test_mod, y_pred_mod, alpha=0.6, color='blue', label='ModulatedGM')
    plt.scatter(y_test_smooth, y_pred_smooth, alpha=0.6, color='green', label='SmoothedGM')
    plt.plot([min(y_test_mod), max(y_test_mod)], [min(y_test_mod), max(y_test_mod)], color='red', linestyle='--')
    plt.title('Predicted vs Actual Age')
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.legend()
    # 顯示圖表
    plt.tight_layout()
    plt.show()
