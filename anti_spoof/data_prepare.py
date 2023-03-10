import os

label_dict = {
    'living': 0,
    'spoof': 1
}

sub_label_dict = {
    'living': 0,
    '2D-Display-Pad': 1,
    '2D-Display-Phone': 2,
    '2D-Print-Album': 3,
    '2D-Print-Newspaper': 4,
    '2D-Print-Photo': 5,
    '2D-Print-Poster': 6,
    '3D-AdultDoll': 7,
    '3D-GarageKit': 8,
    '3D-Mask': 9
}

root = '/CVPR23-FAS-WILD/anti_spoof/data_ssd/CVPR23-FAS-WILD/train/CVPR2023-Anti_Spoof-Challenge-Release-Data-20230209/'
with open(os.path.join(root, 'train10.csv'), 'w') as fdc:
    with open(os.path.join(root, 'train.list'), 'r') as fdl:
        for line in fdl.readlines():
            line = line.strip()
            marks = line.split('/')
            if marks[1] == 'living':
                fdc.write(f'{line.strip()},0,0\n')
            elif marks[1] == 'spoof':
                fdc.write(f'{line.strip()},1,{sub_label_dict[marks[2]]}\n')

