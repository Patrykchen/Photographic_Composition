import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models import build_model
from pc_datasets import build_datasets


def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training PC_classfier', add_help=False)
    parser.add_argument('--backbone', default='resnet', type=str,
                        help="Name of the convolutional backbone to use")

    # 随机种子选择
    parser.add_argument('--seed', default=19, type=int,
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
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--gpu_id', default=1, type=int, help='the gpu used for training')
    parser.add_argument('--num_workers', default=0, type=int)

    return parser


def load_model(info, args):
    device = torch.device('cuda')
    model_list = []

    for name in info.keys():
        args.backbone = name
        path = './voters/' + name + '.pth'

        model, criterion = build_model(args)
        model.to(device)
        criterion.to(device)

        ckpt = torch.load(path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        model.eval()
        model_list.append(model)

    return model_list

def voter(data,model):

    device = torch.device('cuda')

    input = data
    input.to(device)

    output = model(input)
    logits = output
    max_score, max_index = torch.max(output, 1)
    ones = torch.ones_like(output[0])
    zeros = torch.zeros_like(output[0])
    for i in range(len(output)):
        output[i] = torch.where(output[i] > 0.8 * max_score[i], ones, zeros)

    return output

def voters(data):
    info = {'alexnet': 0.24, 'vgg16_bn': 0.24, 'resnet': 0.255, 'densenet': 0.265}
    parser = argparse.ArgumentParser('PCP_model training and testing', parents=[get_args_parser()])
    args = parser.parse_args()

    device = torch.device('cuda')
    model_list = load_model(info, args)


    input = data
    input.to(device)
    for name, model in zip(info.keys(), model_list):
        output = model(input)
        logits = output
        max_score, max_index = torch.max(output, 1)
        ones = torch.ones_like(output)
        zeros = torch.zeros_like(output)
        output = torch.where(output > 0.8 * max_score, ones, zeros)
        if name == 'alexnet':
            result = info[name] * output
            logits = info[name] * logits
        else:
            result += info[name] * output
            logits += info[name] * logits
    max_score, max_index = torch.max(result, 1)
    ones = torch.ones_like(result)
    zeros = torch.zeros_like(result)
    predict = torch.where(result > 0.8 * max_score, ones, zeros)

    return predict


if __name__ == '__main__':
    info = {'alexnet': 0.24, 'vgg16_bn': 0.24, 'resnet': 0.255, 'densenet': 0.265}
    parser = argparse.ArgumentParser('PCP_model training and testing', parents=[get_args_parser()])
    args = parser.parse_args()


    loading_data = build_datasets(args=args)
    train_set, test_set = loading_data(args.data_root, args.augmentation)
    sampler_test = torch.utils.data.SequentialSampler(test_set)
    data_loader_test = DataLoader(test_set, 1, sampler=sampler_test, drop_last=False, num_workers=args.num_workers)

    device = torch.device('cuda')
    model_list = load_model(info, args)

    acc_test = 0
    n = 0
    with torch.no_grad():
        for index, data in enumerate(data_loader_test):
            n = n + 1
            input = data['image']
            label = data['label']
            input = input.to(device)
            label = label.to(device)
            for name, model in zip(info.keys(), model_list):
                output = model(input)
                max_score, max_index = torch.max(output, 1)
                ones = torch.ones_like(output)
                zeros = torch.zeros_like(output)
                temp = torch.where(output > 0.8 * max_score, ones, zeros)
                if name == 'alexnet':
                    predict = info[name] * temp
                else:
                    predict += info[name] * temp
            max_score, max_index = torch.max(predict, 1)
            predict = torch.where(predict > 0.8 * max_score, ones, zeros)
            if torch.sum(label) > 0:
                if torch.sum(predict * label) > 0:
                    acc_item = 1.0
                else:
                    acc_item = 0.0
            else:
                acc_item = 0.0
            acc_test += acc_item
            print(acc_test / n)

    acc_test /= n
    print(acc_test)
