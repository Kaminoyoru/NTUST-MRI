import os
import nibabel as nib
import cupy as cp
import pandas as pd
from scipy.stats import pearsonr
import tigerbx

# 設定輸入與輸出路徑
input_dir = r'E:\MRI\HW2\IXI-T1_decompressed'
output_dir = r'E:\MRI\HW2\IXI-T1_output'
csv_output_path = r'E:\MRI\HW2\IXI_volumes.csv'

# 使用 TigerBx 進行大腦分割
# try:
#     tigerbx.run('d', input_dir, output_dir)  # 使用 GPU 進行大腦分割
#     print("大腦影像分割完成！")
# except Exception as e:
#     print(f"分割過程中發生錯誤：{e}")

# 定義分割標籤對應表

# Label mapping for deep gray-matter structures
label_mapping = {
    1: 'Left-Thalamus-Proper', 2: 'Right-Thalamus-Proper',
    3: 'Left-Caudate', 4: 'Right-Caudate',
    5: 'Left-Putamen', 6: 'Right-Putamen',
    7: 'Left-Pallidum', 8: 'Right-Pallidum',
    9: 'Left-Hippocampus', 10: 'Right-Hippocampus',
    11: 'Left-Amygdala', 12: 'Right-Amygdala'
}

def calculate_volume_gpu(nifti_img, voxel_volume):
    data = cp.asarray(nifti_img.get_fdata())
    unique_labels = cp.unique(data)
    # Convert labels to integers and filter by label_mapping
    volumes = {label_mapping[int(label)]: float(cp.sum(data == label) * voxel_volume)
               for label in unique_labels.get() if int(label) in label_mapping}
    return volumes

# 讀取分割後的影像並計算每個區域的體積
output_volumes = {}
for filename in os.listdir(output_dir):
    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
        filepath = os.path.join(output_dir, filename)
        nifti_img = nib.load(filepath)
        print(f"Processing file: {filename}")  # 確認檔案讀取

        voxel_volume = cp.prod(cp.asarray(nifti_img.header.get_zooms()))
        subject_id = filename.split('.')[0]
        volumes = calculate_volume_gpu(nifti_img, voxel_volume)
        output_volumes[subject_id] = volumes
        print(f"Calculated volumes for {subject_id}: {volumes}")  # 確認體積計算結果

# 將體積字典轉換成 DataFrame
volumes_df = pd.DataFrame(output_volumes).T
volumes_df.index.name = 'IXI_ID'
print("Volumetric Data (volumes_df):")
print(volumes_df.head())  # 顯示體積資料前幾行

# 將體積資料存為 CSV 檔案
volumes_df.to_csv(csv_output_path)
print(f"體積資料已儲存至 {csv_output_path}")

# 載入並合併人口資料
demographics_df = pd.read_csv(r'E:\MRI\HW2\IXI.csv')
demographics_df['IXI_ID'] = demographics_df['IXI_ID'].astype(str)
volumes_df.index = volumes_df.index.astype(str)
combined_df = pd.merge(demographics_df, volumes_df, on='IXI_ID', how='inner')

# 進行相關性分析
age_volumes_corr = {}
for region in volumes_df.columns:
    if combined_df['AGE'].notnull().sum() >= 2 and combined_df[region].notnull().sum() >= 2:
        correlation, p_value = pearsonr(combined_df['AGE'], combined_df[region])
        age_volumes_corr[region] = (correlation, p_value)
    else:
        print(f"Skipping region {region} due to insufficient data.")

# 顯示相關性分析結果
print("\n相關性分析結果：")
for region, (correlation, p_value) in age_volumes_corr.items():
    print(f"Region {region} - Age Correlation: {correlation:.2f}, p-value: {p_value:.4f}")


# #將體積csv與IXI.csv合併
# import pandas as pd

# # 載入人口統計數據
# demographics_df = pd.read_csv(r'E:\MRI\HW2\IXI.csv')

# # 載入體積數據
# volumes_df = pd.read_csv(r'E:\MRI\HW2\IXI_volumes.csv')

# # 從體積數據中的 IXI_ID 提取數字ID，並確保為整數格式
# volumes_df['IXI_ID'] = volumes_df['IXI_ID'].str.extract('(\d+)').astype(int)

# # 確保人口統計數據中的 IXI_ID 也是整數格式，以保持一致性
# demographics_df['IXI_ID'] = demographics_df['IXI_ID'].astype(int)

# # 使用外部連接合併數據集，保證所有數據都被包含
# combined_df = pd.merge(demographics_df, volumes_df, on='IXI_ID', how='outer')

# # 按 IXI_ID 對合併後的數據進行排序
# combined_df = combined_df.sort_values(by='IXI_ID')

# # 保存修正後的合併數據框到新的 CSV 文件
# combined_df.to_csv(r'E:\MRI\HW2\Combined_IXI_Data.csv', index=False)

# # 顯示預覽以確認合併成功
# print(combined_df.head())
