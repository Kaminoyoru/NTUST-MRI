#訓練模型
import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd

import monai
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
)
from monai.data import DataLoader, ImageDataset
from monai.networks.nets import Regressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    # 設定裝置
    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入灰質體積與年齡數據
    # volume_csv_path = r"E:\MRI\HW2\VBM\Monai\SmoothedGM\GrayMatterVolumesWithSpacing_IXI.csv"
    volume_csv_path = r"E:\MRI\HW2\VBM\Monai\SmoothedGM\GrayMatterDensityStats_IXI.csv"
    ixi_csv_path = r"E:\MRI\HW2\VBM\Monai\IXI.csv"

    # 載入 CSV 數據
    volumes_df = pd.read_csv(volume_csv_path)
    ixi_df = pd.read_csv(ixi_csv_path)

    # 合併體積與年齡
    merged_df = volumes_df.merge(
        ixi_df[['IXI_ID', 'AGE']],
        on='IXI_ID',  # 依據 IXI_ID 合併
        how='inner'
    )
    merged_df = merged_df.dropna()

    # 準備影像和年齡標籤
    image_files = merged_df['FileName'].apply(
        lambda x: os.path.join(r"E:\MRI\HW2\VBM\Monai\SmoothedGM", x)
    ).tolist()
    ages = merged_df['AGE'].values

    # 切分訓練集與測試集
    image_files_train, image_files_val, ages_train, ages_val = train_test_split(
        image_files, ages, test_size=0.2, random_state=42
    )

    # 定義數據增強和轉換
    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()])
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])

    # 創建數據集和 DataLoader
    train_ds = ImageDataset(image_files=image_files_train, labels=ages_train, transform=train_transforms)
    val_ds = ImageDataset(image_files=image_files_val, labels=ages_val, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=0, pin_memory=pin_memory)

    # 創建模型
    model = Regressor(in_shape=[1, 96, 96, 96], out_shape=1, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))
    model = model.to(device)

    # 損失函數和優化器
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    # 訓練參數
    val_interval = 2
    max_epochs = 50
    lowest_rmse = sys.float_info.max
    highest_rmse = -sys.float_info.max  # 用於追踪最差模型
    lowest_rmse_epoch = -1
    highest_rmse_epoch = -1

    # 訓練模型
    writer = SummaryWriter()
    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= step
        print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        writer.add_scalar("train_loss", epoch_loss, epoch + 1)

        # 驗證模型
        if (epoch + 1) % val_interval == 0:
            model.eval()
            all_labels = []
            all_val_outputs = []
            with torch.no_grad():
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    val_outputs = model(val_images)
                    all_labels.extend(val_labels.cpu().numpy())
                    all_val_outputs.extend(val_outputs.cpu().numpy().flatten())

            # 計算指標
            mse = mean_squared_error(all_labels, all_val_outputs)
            rmse = np.sqrt(mse)

            # 保存最好的模型
            if rmse < lowest_rmse:
                lowest_rmse = rmse
                lowest_rmse_epoch = epoch + 1
                torch.save(model.state_dict(), r"E:\MRI\HW2\VBM\Monai\best_model.pth")
                print("Saved new best model!")

            # 保存最差的模型
            if rmse > highest_rmse:
                highest_rmse = rmse
                highest_rmse_epoch = epoch + 1
                torch.save(model.state_dict(), r"E:\MRI\HW2\VBM\Monai\worst_model.pth")
                print("Saved new worst model!")

            print(f"Validation RMSE: {rmse:.4f}")
            writer.add_scalar("val_rmse", rmse, epoch + 1)

    print(f"Training completed")
    print(f"Lowest RMSE: {lowest_rmse:.4f} at epoch: {lowest_rmse_epoch}")
    print(f"Highest RMSE: {highest_rmse:.4f} at epoch: {highest_rmse_epoch}")
    writer.close()


if __name__ == "__main__":
    main()
    
#使用訓練好的模型
#import os
# import torch
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# from monai.transforms import (
#     EnsureChannelFirst,
#     Compose,
#     Resize,
#     ScaleIntensity,
# )
# from monai.data import DataLoader, ImageDataset
# from monai.networks.nets import Regressor


# def generate_scatter_plot(actual_ages, predictions, title, save_path):
#     """
#     繪製實際年齡與預測年齡的散布圖
#     """
#     plt.figure(figsize=(8, 6))
#     plt.scatter(actual_ages, predictions, alpha=0.7, color="blue", label="Predictions")
#     plt.plot(
#         [min(actual_ages), max(actual_ages)],
#         [min(actual_ages), max(actual_ages)],
#         color="red",
#         linestyle="--",
#         label="Ideal Fit",
#     )
#     plt.xlabel("Actual Age")
#     plt.ylabel("Predicted Age")
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(save_path)
#     plt.show()
#     print(f"Scatter plot saved to {save_path}")


# def predict_and_evaluate(model_path, actual_ages, image_files, device, title, scatter_plot_path):
#     """
#     使用模型進行預測、評估並繪製散布圖
#     """
#     # 定義數據轉換
#     predict_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])

#     # 創建數據集和 DataLoader
#     predict_ds = ImageDataset(image_files=image_files, labels=None, transform=predict_transforms)
#     predict_loader = DataLoader(predict_ds, batch_size=1, num_workers=0)

#     # 載入模型
#     model = Regressor(in_shape=[1, 96, 96, 96], out_shape=1, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model = model.to(device)
#     model.eval()

#     # 預測影像年齡
#     predictions = []
#     with torch.no_grad():
#         for batch_images in predict_loader:
#             batch_images = batch_images.to(device)
#             batch_predictions = model(batch_images)
#             predictions.extend(batch_predictions.cpu().numpy().flatten())

#     # 檢查並移除 NaN 值
#     predictions = np.array(predictions)
#     actual_ages = np.array(actual_ages)
#     valid_mask = ~np.isnan(predictions) & ~np.isnan(actual_ages)
#     predictions = predictions[valid_mask]
#     actual_ages = actual_ages[valid_mask]

#     # 確認處理後是否仍有數據
#     if len(predictions) == 0 or len(actual_ages) == 0:
#         raise ValueError("No valid data available for evaluation after removing NaNs.")

#     # 計算評估指標
#     mae = mean_absolute_error(actual_ages, predictions)
#     mse = mean_squared_error(actual_ages, predictions)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(actual_ages, predictions)

#     print(f"{title} Evaluation Metrics:")
#     print(f"MAE: {mae:.4f}")
#     print(f"MSE: {mse:.4f}")
#     print(f"RMSE: {rmse:.4f}")
#     print(f"R²: {r2:.4f}")

#     # 繪製散布圖
#     generate_scatter_plot(actual_ages, predictions, title, scatter_plot_path)


# def main():
#     # 設定裝置
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 模型檔案路徑
#     best_model_path = r"E:\MRI\HW2\VBM\Monai\ModulatedGM\best_model.pth"
#     worst_model_path = r"E:\MRI\HW2\VBM\Monai\ModulatedGM\worst_model.pth"

#     # 載入灰質體積與年齡數據
#     volume_csv_path = r"E:\MRI\HW2\VBM\Monai\ModulatedGM\GrayMatterVolumesWithStats_IXI.csv"
#     ixi_csv_path = r"E:\MRI\HW2\VBM\Monai\IXI.csv"

#     # 載入 CSV 並合併數據
#     volumes_df = pd.read_csv(volume_csv_path)
#     ixi_df = pd.read_csv(ixi_csv_path)

#     # 合併數據，確保年齡資料無遺漏
#     merged_df = volumes_df.merge(
#         ixi_df[['IXI_ID', 'AGE']],
#         on='IXI_ID',
#         how='inner'
#     ).dropna()

#     # 準備影像路徑和實際年齡
#     image_files = merged_df['FileName'].apply(
#         lambda x: os.path.join(r"E:\MRI\HW2\VBM\Monai\ModulatedGM", x)
#     ).tolist()
#     actual_ages = merged_df['AGE'].values

#     # 分割訓練集和測試集，確保無重疊
#     image_files_train, image_files_test, ages_train, ages_test = train_test_split(
#         image_files, actual_ages, test_size=0.2, random_state=42
#     )

#     # 預測並繪製最佳模型的散布圖
#     best_scatter_plot_path = r"E:\MRI\HW2\VBM\Monai\ModulatedGM\best_model_scatter_plot.png"
#     predict_and_evaluate(
#         model_path=best_model_path,
#         actual_ages=ages_test,
#         image_files=image_files_test,
#         device=device,
#         title="Best Model: Actual vs Predicted Ages",
#         scatter_plot_path=best_scatter_plot_path,
#     )

#     # 預測並繪製最差模型的散布圖
#     worst_scatter_plot_path = r"E:\MRI\HW2\VBM\Monai\ModulatedGM\worst_model_scatter_plot.png"
#     predict_and_evaluate(
#         model_path=worst_model_path,
#         actual_ages=ages_test,
#         image_files=image_files_test,
#         device=device,
#         title="Worst Model: Actual vs Predicted Ages",
#         scatter_plot_path=worst_scatter_plot_path,
#     )


# if __name__ == "__main__":
#     main()
