import torch
import timm
from ptflops import get_model_complexity_info


def main():
    # for net in [timm.models.swin_tiny_patch4_window7_224(), timm.models.swin_small_patch4_window7_224(), timm.models.swin_base_patch4_window7_224(), timm.models.swin_large_patch4_window7_224(), timm.models.swinv2_cr_tiny_224(), timm.models.swinv2_cr_small_224()]:
        # get_flops(net, input_size)
    for input_size in [(3, 224, 224), (3, 112, 112)]:
        net = timm.models.swin_tiny_patch4_window7_224(num_classes=2, input_size=input_size)
        get_flops(net, input_size)

def get_flops(net, input_size):
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(net, input_size, as_strings=True,
                                                print_per_layer_stat=False, verbose=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


if __name__ == "__main__":
    main()