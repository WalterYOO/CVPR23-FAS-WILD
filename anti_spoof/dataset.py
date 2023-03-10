import os
import random
import pandas as pd
from torch.utils.data import Dataset

from io import BytesIO
from zipfile import ZipFile

from PIL import Image
from PIL import ImageOps
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class FASWildDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, is_train=True, is_cutface=False, keep_ratio=False, balance_sampling=False):
        self.is_train = is_train
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_labels.columns = ['path', 'label', 'sublabel'] if self.is_train else ['path']
        self.img_dir = img_dir
        self.transform = transform
        self.is_cutface = is_cutface
        self.keep_ratio = keep_ratio
        self.balance_sampling = balance_sampling
        if self.is_train and self.balance_sampling:
            self._balance_sampling()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        try:
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            print(e)
        image = image.convert('RGB')
        if self.is_cutface:
            expand_ratio = random.random() * .1
            image = self._cutout_face(image, img_path, expand_ratio=expand_ratio)
        if self.keep_ratio:
            w, h = image.size
            edge_length = max(w, h)
            image = ImageOps.pad(image, (edge_length, edge_length))
        if self.transform:
            image = self.transform(image)
        if self.is_train:
            label = self.img_labels.iloc[idx, 1]
            sublabel = self.img_labels.iloc[idx, 2]
            return image, label, sublabel
        else:
            return image, self.img_labels.iloc[idx, 0]

    def _cutout_face(self, image, image_path, expand_ratio=0):
        pt_path = os.path.splitext(image_path)[0] + '.txt'
        pts = pd.read_csv(pt_path, header=None, sep=' ')
        x1, y1 = pts.iloc[0]
        x2, y2 = pts.iloc[1]
        x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
        y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
        w = x2 - x1
        h = y2 - y1
        face = image.crop((x1 - w * expand_ratio, y1 - h * expand_ratio, x2 + w * expand_ratio, y2 + h * expand_ratio))
        return face

    def _balance_sampling(self):
        img_labels_count = self.img_labels.groupby('sublabel').count()
        quantile = img_labels_count.label.quantile(0.25)
        over_sample_idx = img_labels_count.query(f'label <= {quantile // 2}').index
        over_samples = list()
        for i in over_sample_idx:
            over_samples.extend([self.img_labels.query(f'sublabel == {i}')] * int(quantile // img_labels_count.iloc[i].label - 1))
        self.img_labels = self.img_labels.append(over_samples)


class FASWildZipDataset(Dataset):
    def __init__(self, zip_file, zip_passwd=None, annotations_file=None, transform=None, is_train=True):
        if annotations_file:
            self.img_labels = pd.read_csv(annotations_file, header=None)
        self.zip_file = ZipFile(zip_file, 'r')
        self.zip_passwd = zip_passwd
        self.zip_file.setpassword(self.zip_passwd)
        namelist = self.zip_file.namelist()
        self.img_list = [x for x in namelist if 'jpg' in x]
        self.pt_list = [x for x in namelist if 'txt' in x]
        self.transform = transform
        self.is_train = is_train

    def __del__(self):
        self.zip_file.close()
        self.zip_file = None

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        data = self.zip_file.read(img_name)
        image = Image.open(BytesIO(data))
        try:
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            print(e)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.is_train:
            label = 1 if 'living' in img_name else 0
            return image, label
        else:
            return image, os.path.split(img_name)[-1]