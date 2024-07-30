import numpy as np # type: ignore

def load_pref_datasets(file_path):

    data = np.load(file_path)

    return data


file_path = 'pref_datasets.npz'  # Đường dẫn tới tệp .npz đã lưu
pref_datasets = load_pref_datasets(file_path)
print("Keys in the .npz file:")
print(pref_datasets.files)
# In thông tin để kiểm tra
print("obs_1 shape:", pref_datasets['obs_1'].shape)
print("obs_2 shape:", pref_datasets['obs_2'].shape)
print("action_1 shape:", pref_datasets['action_1'].shape)
print("action_2 shape:", pref_datasets['action_2'].shape)
print("label shape:", pref_datasets['label'].shape)
print("label data:", pref_datasets['label'])
print("label data:", pref_datasets['action_1'])
