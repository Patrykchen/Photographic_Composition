# 用来观测VGG不同层次中的输出结果 SGD Augmentation off

import argparse
import datetime
import random
import time
import os
import numpy as np
import cv2 as cv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import vgg_internal_output.model_internal as model_internal
from torchvision import transforms

from models import build_model
from pc_datasets import build_datasets
from tensorboardX import SummaryWriter
import warnings

warnings.filterwarnings('ignore')
def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training PC_classfier', add_help=False)
    # SGD使用1e-2开始训练
    # ADAM使用1e-4开始训练
    parser.add_argument('--lr',default = 1e-2,type=float)
    parser.add_argument('--lr_backbone', default=1e-2,type=float)
    parser.add_argument('--lr_drop', default=30, type=int)
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=80, type=int)
    # backbone选择,alexnet论文原模型，vgg16_bn,resnet(101层),densenet(201层)
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")
    # 随机种子选择
    parser.add_argument('--seed',default=19,type=int,
                        help='set the seed for reproducing the result')

    # 数据
    parser.add_argument('--data_root', default='./DATA',
                        help='path where to get the data')
    parser.add_argument('--dataset_file', default='KU',
                        help='dataset to use')
    parser.add_argument('--augmentation', default=False,
                        help='if use augmentation')

    # 结果保存
    parser.add_argument('--output_dir', default='./log',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoints_dir', default='./ckpt',
                        help='path where to save checkpoints, empty for no saving')
    parser.add_argument('--tensorboard_dir', default='./runs',
                        help='path where to save, empty for no saving')

    # 运行选项
    parser.add_argument('--eval_freq',default = 5,type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--resume',default='./ckpt/best_loss_vgg16_bn_Adam_aug_on.pth',help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--gpu_id', default=1, type=int, help='the gpu used for training')
    parser.add_argument('--num_workers', default=8, type=int)

    return parser

def main(args):
    torch.multiprocessing.set_sharing_strategy('file_system')
    # setting the gui_id
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    # setting gpu training

    device = torch.device('cuda')
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    # 建立模型并且读取
    #****************************#
    model, criterion = build_model(args, training=False)
    # move to GPU
    model.to(device)
    criterion.to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        #args.start_epoch = checkpoint['epoch'] + 1
    model_layer1 = model_internal.VGG_0(model)
    #*****************************#
    # 构造用来测试的测试图
    # create the dataset
    loading_data = build_datasets(args=args)
    # create the training and testing set
    # 得到训练集和验证集
    train_set, test_set = loading_data(args.data_root, args.augmentation)
    # create the sampler used during training
    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_test = torch.utils.data.SequentialSampler(test_set)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    # the dataloader for training
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
                                   num_workers=args.num_workers)

    data_loader_test = DataLoader(test_set, 1, sampler=sampler_test,
                                  drop_last=False, num_workers=args.num_workers)
    #*****************************#
    model_layer1.eval()
    n = 0
    with torch.no_grad():
        for index, data in enumerate(data_loader_test):
            n = n + 1
            input = data['image']
            label = data['label']
            input = input.to(device)
            label = label.to(device)
            output = model_layer1(input)
            unloader = transforms.ToPILImage()
            image = output.cpu().clone()  # clone the tensor
            image = image.squeeze(0)  # remove the fake batch dimension
            image = image[0,:,:] # 取第一张图片显示
            image = unloader(image)
            image.save('example.jpg')
            print('ok')





if __name__ == '__main__':
    parser = argparse.ArgumentParser('PCP_model training and testing', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)