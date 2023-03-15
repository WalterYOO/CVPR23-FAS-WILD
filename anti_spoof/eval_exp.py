# %%
import os
import json
# %%
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import timm
# %%
from tqdm import tqdm
# %%
from dataset import FASWildDataset, FASWildZipDataset
# %%
import argparse
parser = argparse.ArgumentParser(description='Config substitute')
parser.add_argument('--ckpt', default='',type=str, help='Checkpoint file path')
parser.add_argument('--model-name', default='',type=str, help='Model name')
parser.add_argument('--image-size', default=224,type=int, help='Input image size')
parser.add_argument('--batch-size', default=128,type=int, help='Input batch size')
parser.add_argument('--number-classes', default=10,type=int, help='Number of classes')
args = parser.parse_args()
# %%
# annotations_file = '/CVPR23-FAS-WILD/anti_spoof/data/CVPR23-FAS-WILD/train/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/train.csv'
# img_dir = '/CVPR23-FAS-WILD/anti_spoof/data/CVPR23-FAS-WILD/train/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209'
annotations_file = '/CVPR23-FAS-WILD/anti_spoof/data_ssd/CVPR23-FAS-WILD/dev/CVPR2023-Anti_Spoof-Challenge-ReleaseData-Dev-20230211/Dev.txt'
img_dir = '/CVPR23-FAS-WILD/anti_spoof/data_ssd/CVPR23-FAS-WILD/dev/CVPR2023-Anti_Spoof-Challenge-ReleaseData-Dev-20230211/data'
zip_file = '/CVPR23-FAS-WILD/anti_spoof/data_local/CVPR23-FAS-WILD/dev/CVPR2023-Anti_Spoof-Challenge-ReleaseData-Dev-20230211/CVPR2023-Anti_Spoof-Challenge-ReleaseData-Dev-20230211-r0.zip'
# %%
subset = 'dev'
output_prefix = f'/CVPR23-FAS-WILD/anti_spoof/{subset}out'
# %%
if not os.path.exists(output_prefix):
    os.makedirs(output_prefix)
if not os.path.isdir(output_prefix):
    print(f'{output_prefix} is not a directory!')
    exit()
# find the non-exist exp number which close to 0
dirnames = [x for x in os.listdir(output_prefix) if 'exp' in x and os.path.isdir(os.path.join(output_prefix, x))]
def find_last_number(string):
    import re
    all_number = re.findall(r'\d+', string)
    if all_number:
        return int(all_number[-1])
    else:
        return -1
dirnames.sort(key=find_last_number)
exp_number = 0
for i in range(len(dirnames)):
    if i < find_last_number(dirnames[i]):
        break
    exp_number = i + 1
output = f'{output_prefix}/exp{exp_number}'
# %%
if not os.path.exists(output):
    os.makedirs(output)
# %%
input_size = (args.image_size, args.image_size)
# transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Resize(input_size),
    ])
# %%
is_cutface = True
keep_ratio = True
evalset = FASWildDataset(annotations_file, img_dir, transform, is_train=False, is_cutface=is_cutface, keep_ratio=keep_ratio)
# evalset = FASWildZipDataset(zip_file=zip_file, transform=transform, is_train=False)
# %%
batch_size = args.batch_size
evalloader = torch.utils.data.DataLoader(evalset, batch_size=batch_size,
                                          shuffle=False, num_workers=16)
# %%
ckpt_filename = '/CVPR23-FAS-WILD/anti_spoof/output/exp50/anti_spoof_mobilenetv3_large_100-23.pth' if not args.ckpt else args.ckpt
# %%
ckpt = torch.load(ckpt_filename)
# %%
model_name = 'mobilenetv3_large_100' if not args.model_name else args.model_name
if hasattr(timm.models, model_name):
    model_func = getattr(timm.models, model_name)
else:
    print(f'There is no model {model_name} in timm')
    exit()
num_classes = args.number_classes
net = model_func(num_classes=num_classes)
net.load_state_dict(ckpt['net'])
net = net.cuda()
# %%
config = {
    'annotations_file': annotations_file,
    'img_dir': img_dir,
    'zip_file': zip_file,
    'input_size': input_size,
    'is_cutface': is_cutface,
    'keep_ratio': keep_ratio,
    'num_classes': num_classes,
    'model_name': model_name,
    'output': output,
    'ckpt_filename': ckpt_filename,
}
with open(f'{output}/config.json', 'w') as fd:
    json.dump(config, fd, indent=4)
# %%
with open(os.path.join(output, 'predict.txt'), 'w') as fd:
    with torch.no_grad():
        for data in tqdm(evalloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, names = data
            inputs = inputs.cuda()

            # forward + backward + optimize
            outputs = net(inputs)

            outputs = F.softmax(outputs, dim=1)

            for name, prob in zip(names, outputs):
                fd.write(f'{subset}/{name} {prob[0]:.8f}\n')

        print('Finished Eval')
        print(f'Predict results write into {os.path.join(output, "predict.txt")}')
# %%
