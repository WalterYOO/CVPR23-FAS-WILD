# %%
import torch
import torchvision.transforms as transforms
from dataset import FASWildDataset, FASWildZipDataset

# %%
annotations_file = '/CVPR23-FAS-WILD/anti_spoof/data_local/CVPR23-FAS-WILD/train/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/train.csv'
img_dir = '/CVPR23-FAS-WILD/anti_spoof/data_local/CVPR23-FAS-WILD/train/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209'
# %%
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Resize((224, 224)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
# %%
trainset = FASWildDataset(annotations_file, img_dir, transform=transform, is_cutface=True, keep_ratio=True)

# %%
try:
    for i, (img, label) in enumerate(trainset):
        print(f'{i}/{len(trainset)}', end='\r', flush=True)
except Exception as e:
    print(e, end='\n')
# %%
print(img.shape)
# %%
print(label)
# %%
print(img.max())
# %%
# zip_file = '/CVPR23-FAS-WILD/anti_spoof/data/CVPR23-FAS-WILD/dev/CVPR2023-Anti_Spoof-Challenge-ReleaseData-Dev-20230211/CVPR2023-Anti_Spoof-Challenge-ReleaseData-Dev-20230211.zip'
# zip_passwd = b'f704dd568338'
zip_file = '/CVPR23-FAS-WILD/anti_spoof/data_local/CVPR23-FAS-WILD/dev/CVPR2023-Anti_Spoof-Challenge-ReleaseData-Dev-20230211/CVPR2023-Anti_Spoof-Challenge-ReleaseData-Dev-20230211-r0.zip'
zip_passwd = None
# %%
evalset = FASWildZipDataset(zip_file=zip_file, zip_passwd=zip_passwd, transform=transform, is_train=False)
# %%
try:
    for i, (img, name) in enumerate(evalset):
        print(f'{i}/{len(evalset)}', end='\r', flush=True)
except Exception as e:
    print(e, end='\n')
# %%
print(img.shape)
# %%
print(label)
# %%
print(img.max())
# %%
trainset = FASWildDataset(annotations_file, img_dir, is_cutface=True)

# %%
for image, label in trainset:
    break
# %%
