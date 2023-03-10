# %%
import os
import pandas as pd
from PIL import Image
from PIL import ImageOps

# %%
annotations_file = '/CVPR23-FAS-WILD/anti_spoof/data_local/CVPR23-FAS-WILD/train/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/train.csv'
img_dir = '/CVPR23-FAS-WILD/anti_spoof/data_local/CVPR23-FAS-WILD/train/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209'
# %%
img_labels = pd.read_csv(annotations_file, header=None)
img_dir = img_dir
# %%
for idx in range(len(img_labels)):
    print(f'{idx}/{len(img_labels)}', end='\r', flush=True)
    img_path = os.path.join(img_dir, img_labels.iloc[idx, 0])
    try:
        image = Image.open(img_path)
        try:
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            print(e)
        image = image.convert('RGB')
    except Exception as e:
        print(e)
        print(f'{idx}: {img_labels.iloc[idx, 0]}')
# %%
annotations_file = '/CVPR23-FAS-WILD/anti_spoof/data_local/CVPR23-FAS-WILD/dev/CVPR2023-Anti_Spoof-Challenge-ReleaseData-Dev-20230211/Dev.txt'
img_dir = '/CVPR23-FAS-WILD/anti_spoof/data_local/CVPR23-FAS-WILD/dev/CVPR2023-Anti_Spoof-Challenge-ReleaseData-Dev-20230211/data'
# %%
img_labels = pd.read_csv(annotations_file, header=None)
img_dir = img_dir
# %%
for idx in range(len(img_labels)):
    print(f'{idx}/{len(img_labels)}', end='\r', flush=True)
    img_path = os.path.join(img_dir, img_labels.iloc[idx, 0])
    try:
        image = Image.open(img_path)
        try:
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            print(e)
        image = image.convert('RGB')
    except Exception as e:
        print(e)
        print(f'{idx}: {img_labels.iloc[idx, 0]}')
# %%