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
