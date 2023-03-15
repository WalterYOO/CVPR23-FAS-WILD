# %%
import os
import json
# %%
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler

import timm
# %%
from tqdm import tqdm
# %%
from dataset import FASWildDataset
# %%
from tensorboardX import SummaryWriter
# %%
import logging
logging.basicConfig(level=logging.INFO)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
logger = logging.getLogger(__file__)
# hstdout = logging.StreamHandler(sys.stdout)
# hstdout.setFormatter(formatter)
# logger.addHandler(hstdout)
# %%
annotations_file = '/CVPR23-FAS-WILD/anti_spoof/data/CVPR23-FAS-WILD/train/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/train10.csv'
img_dir = '/CVPR23-FAS-WILD/anti_spoof/data/CVPR23-FAS-WILD/train/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209'
# %%
output_prefix = '/CVPR23-FAS-WILD/anti_spoof/output'
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
hfile = logging.FileHandler(f'{output}/train.log')
hfile.setFormatter(formatter)
logger.addHandler(hfile)
tbwriter = SummaryWriter(log_dir=output)
# %%
input_size = (640, 640)
use_imagenet_norm = False
if use_imagenet_norm:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        ])
# %%
is_cutface = True
keep_ratio = True
balance_sampling = False
trainset = FASWildDataset(annotations_file, img_dir, transform, is_cutface=is_cutface, keep_ratio=keep_ratio, balance_sampling=balance_sampling)
# %%
base_batch_size = 128
batch_size = 64
batch_size_scale = base_batch_size / batch_size
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=16)
# %%
resume = True
reset_optim_state_dict = True
ckpt_filename = '/CVPR23-FAS-WILD/anti_spoof/output/exp37/anti_spoof_mobilenetv3_large_100-23.pth'
# %%
if resume:
    ckpt = torch.load(ckpt_filename)
# %%
model_name = 'mobilenetv3_large_100'
if hasattr(timm.models, model_name):
    model_func = getattr(timm.models, model_name)
else:
    print(f'There is no model {model_name} in timm')
    exit()
num_classes = 2
net = model_func(num_classes=num_classes)
if resume:
    if num_classes != ckpt['net']['classifier.bias'].shape[0]:
        logger.info(f'num_classes {num_classes} mismatch classifier.bias {ckpt["net"]["classifier.bias"].shape[0]}')
        del ckpt['net']['classifier.weight']
        del ckpt['net']['classifier.bias']
    net.load_state_dict(ckpt['net'], strict=False)
net = net.cuda()
# %%
use_focal = False
base_lr = 0.1
criterion = nn.CrossEntropyLoss(reduction='none' if use_focal else 'mean')
optimizer_type = 'SGD'
if optimizer_type == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9)
elif optimizer_type == 'AdamW':
    optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=0.0001)
if resume and not reset_optim_state_dict:
    optimizer.load_state_dict(ckpt['optimizer'])
# %%
reset_epoch = True
epochs = 24
start_epoch = 0 if not resume or reset_epoch else ckpt['epoch'] + 1
# %%
scheduler_type = 'CosineAnnealingLR'
if scheduler_type == 'MultiStepLR':
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 13, 17, 21], gamma=0.1)
elif scheduler_type == 'CosineAnnealingLR':
    t_total = len(trainloader) * epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_total, eta_min=1e-8)
if resume and not reset_optim_state_dict:
    scheduler.load_state_dict(ckpt['scheduler'])
# %%
def save_checkpoint(net, optimizer, scheduler, epoch, prefix):
    ckpt = {
        'epoch': epoch,
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(ckpt, f'{prefix}-{epoch}.pth')
# %%
config = {
    'annotations_file': annotations_file,
    'img_dir': img_dir,
    'input_size': input_size,
    'is_cutface': is_cutface,
    'keep_ratio': keep_ratio,
    'balance_sampling': balance_sampling,
    'use_imagenet_norm': use_imagenet_norm,
    'num_classes': num_classes,
    'model_name': model_name,
    'base_lr': base_lr,
    'use_focal': use_focal,
    'optimizer_type': optimizer_type,
    'scheduler_type': scheduler_type,
    'reset_optim_state_dict': reset_optim_state_dict,
    'output': output,
    'resume': resume,
    'reset_epoch': reset_epoch,
    'ckpt_filename': ckpt_filename,
    'epochs': epochs,
    'start_epoch': start_epoch,
}
with open(f'{output}/config.json', 'w') as fd:
    json.dump(config, fd, indent=4)
# %%
scaler = GradScaler()
for epoch in range(start_epoch, epochs):  # loop over the dataset multiple times
    tbwriter.add_scalar('lr', scheduler.get_last_lr()[0], epoch * len(trainloader) // batch_size_scale)

    running_loss = 0.0
    batch_count = 0
    for data in tqdm(trainloader):
        batch_count += 1
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels, sublabels = data
        inputs = inputs.cuda()
        if num_classes == 2:
            labels = labels.cuda()
        elif num_classes == 10:
            labels = sublabels.cuda()

        # write images sample in tensorboard
        if epoch == start_epoch and batch_count == 1:
            tbwriter.add_image('inputs_sample', torchvision.utils.make_grid(inputs), epoch)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        if use_focal:
            loss = ((1 - torch.exp(-loss))**2 * loss).mean()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if batch_count % 200 == 0:    # print every 200 mini-batches
            # tbwriter.add_scalar('train_loss', running_loss / 200, batch_count + epoch * len(trainloader))
            tbwriter.add_scalar('train_loss', running_loss / 200, batch_count // batch_size_scale + epoch * len(trainloader) // batch_size_scale)
            logger.info(f'[{epoch}, {batch_count:5d}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

        scheduler.step()

    save_checkpoint(net, optimizer, scheduler, epoch, f'{output}/anti_spoof_{model_name}')

logger.info('Finished Training')
# %%
