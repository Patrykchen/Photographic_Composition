"""
Created on Tue April 19 2022
@author: Wang Zhicheng
"""

import torchvision.transforms as transforms
from pc_datasets.KU_PCP.KU import KU_PCPDataset
import os

def loading_data(data_root,augmentation):
    if augmentation == True:
        transform = transforms.Compose([
            # 数据增强，反转与亮度变换
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    train_path = 'train_img'
    test_path = 'test_img'
    train_label = 'train_label.txt'
    test_label = 'test_label.txt'
    train_path = os.path.join(data_root, train_path)
    test_path = os.path.join(data_root, test_path)
    train_list = os.path.join(data_root,train_label)
    test_list = os.path.join(data_root, test_label)

    input_size = [224,224]

    train_set = KU_PCPDataset(train_path, train_list, input_size, MDC =False,transform=transform)
    test_set = KU_PCPDataset(test_path, test_list, input_size, MDC = False ,transform=transform)

    return train_set,test_set

