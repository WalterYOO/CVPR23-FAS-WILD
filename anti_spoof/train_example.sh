CUDA_VISIBLE_DEVICES=0 python train_exp.py --resume --ckpt /CVPR23-FAS-WILD/anti_spoof/output/exp0/anti_spoof_mobilenetv3_large_100-23.pth --model-name mobilenetv3_large_100 --image-size 448 --number-classes 10 --batch-size 64 --use-imagenet-norm
