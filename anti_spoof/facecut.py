# %%
import os
import pandas as pd
# %%
from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# %%
annotations_file = '/CVPR23-FAS-WILD/anti_spoof/data_local/CVPR23-FAS-WILD/train/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/train.csv'
img_dir = '/CVPR23-FAS-WILD/anti_spoof/data_local/CVPR23-FAS-WILD/train/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209'
img_labels = pd.read_csv(annotations_file, header=None)
# %%
idx = 0
img_path = os.path.join(img_dir, img_labels.iloc[idx, 0])
image = Image.open(img_path)
try:
    image = ImageOps.exif_transpose(image)
except Exception as e:
    print(e)
image = image.convert('RGB')
# %%
pt_path = os.path.join(img_dir, os.path.splitext(img_labels.iloc[idx, 0])[0] + '.txt')
# %%
image.show()
# %%
pts = pd.read_csv(pt_path, header=None, sep=' ')
# %%
draw = ImageDraw.Draw(image)
# %%
colors = [
    (224, 0, 0), # red
    (0, 224, 0), # green
    (0, 0, 224), # blue
    (224, 224, 0), # yellow
    (224, 0, 224), # pink
    (0, 224, 224), # cyan
    (224, 224, 224), # white
]
for idx in range(len(pts)):
    draw.ellipse((pts.iloc[idx, 0] - 5, pts.iloc[idx, 1] - 5, pts.iloc[idx, 0] + 5, pts.iloc[idx, 1] + 5), fill=colors[idx])
# %%
image.show()
# %%
face = image.crop((pts.iloc[0, 0], pts.iloc[0, 1], pts.iloc[1, 0], pts.iloc[1, 1]))
face.show()
# %%
x1, y1 = pts.iloc[0]
x2, y2 = pts.iloc[1]
x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
w = x2 - x1
h = y2 - y1
# %%
expand_ratio = 1.0
face = image.crop((x1 - w * expand_ratio, y1 - h * expand_ratio, x2 + w * expand_ratio, y2 + h * expand_ratio))
face.show()
# %%
face.size
# %%
