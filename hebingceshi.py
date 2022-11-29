"""
Created on Mon April 18 2022
@author: Wang Zhicheng
"""

import argparse
import datetime
import random
import time
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler


from models import build_model
from pc_datasets import build_datasets
from tensorboardX import SummaryWriter
import warnings

warnings.filterwarnings('ignore')
def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training PC_classfier', add_help=False)
    # SGD使用1e-2开始训练
    # ADAM使用1e-4开始训练
    # Adam,SGD
    parser.add_argument('--lr',default = 1e-2,type=float)
    parser.add_argument('--lr_backbone', default=1e-2,type=float)
    parser.add_argument('--lr_drop', default=30, type=int)
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=80, type=int)
    # backbone选择,alexnet论文原模型，vgg16_bn,resnet(101层),densenet(201层)
    parser.add_argument('--backbone', default='alexnet', type=str,
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
    parser.add_argument('--resume',default='./ckpt/best_loss_alexnet_Adam_aug_on.pth',help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--gpu_id', default=3, type=int, help='the gpu used for training')
    parser.add_argument('--num_workers', default=8, type=int)

    return parser
def main(args):
    torch.multiprocessing.set_sharing_strategy('file_system')
    # setting the gui_id
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    # create the logging file
    if args.augmentation:
        filename_tail = args.backbone+'_'+args.optimizer+'_aug_on'
    else:
        filename_tail = args.backbone + '_' + args.optimizer + '_aug_off'
    run_log_name = 'run_log_'+filename_tail+'.txt'
    run_log_name = os.path.join(args.output_dir,run_log_name)
    with open(run_log_name,'a') as log_file:
        log_file.write('Eval Log %s\n' % time.strftime("%c"))
    print(args)
    with open(run_log_name,'a') as log_file:
        log_file.write("{}".format(args))
    # setting gpu training
    device = torch.device('cuda')
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)

    model, criterion = build_model(args, training = True)
    # move to GPU
    model.to(device)

    args.backbone = 'vgg16_bn'
    path = './ckpt/best_loss_' + 'vgg16_bn' + '_Adam_aug_on.pth'
    model1, criterion = build_model(args, training=True)
    ckpt = torch.load(path, map_location='cpu')
    model1.load_state_dict(ckpt['model'])
    # move to GPU
    model1.to(device)

    args.backbone = 'resnet'
    path = './ckpt/best_loss_' + 'resnet' + '_Adam_aug_on.pth'
    model2, criterion = build_model(args, training=True)
    ckpt = torch.load(path, map_location='cpu')
    model2.load_state_dict(ckpt['model'])
    # move to GPU
    model2.to(device)

    args.backbone = 'densenet'
    path = './ckpt/best_loss_' + 'densenet' + '_Adam_aug_on.pth'
    model3, criterion = build_model(args, training=True)
    ckpt = torch.load(path, map_location='cpu')
    model3.load_state_dict(ckpt['model'])
    # move to GPU
    model3.to(device)

    # move to GPU
    criterion.to(device)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    t = model_without_ddp.named_parameters()
    # use different optimation params for different parts of the model
    # 使用p for n, p in model_without_ddp.named_parameters()读取参数名字与参数值，对于backbone设定一定的参数，且要其要grad
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    # SGD is used by default
    if args.optimizer=='SGD' or args.optimizer==None:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr)
    if args.optimizer=='Adam':
        optimizer = torch.optim.Adam(param_dicts, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # create the dataset
    loading_data = build_datasets(args=args)
    # create the training and testing set
    # 得到训练集和验证集
    train_set, test_set = loading_data(args.data_root,args.augmentation)
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

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        #args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    # save the performance during the training
    writer = SummaryWriter(args.tensorboard_dir)

    loss_test_list = []
    loss_epoch = []
    step = 0
    writer = SummaryWriter(args.tensorboard_dir)
    for epoch in range(args.start_epoch, args.epochs):
        if epoch % args.eval_freq == 0 and epoch != 0:
            t1 = time.time()
            model.eval()
            model1.eval()
            model2.eval()
            model3.eval()
            loss_test = 0
            n = 0
            acc_test = 0
            w = [0,0,0,1]
            with torch.no_grad():
                for index, data in enumerate(data_loader_test):
                    n = n+1
                    input = data['image']
                    label = data['label']
                    input = input.to(device)
                    label = label.to(device)
                    output = model(input)
                    output1 = model1(input)
                    output2 = model2(input)
                    output3 = model3(input)

                    max_score, max_index = torch.max(output,1)
                    max_score1, max_index1 = torch.max(output1, 1)
                    max_score2, max_index2 = torch.max(output2, 1)
                    max_score3, max_index3 = torch.max(output3, 1)

                    ones = torch.ones_like(output)
                    zeros = torch.zeros_like(output)

                    predict0 = torch.where(output > 0.8*max_score, ones, zeros)
                    predict1 = torch.where(output1 > 0.8 * max_score1, ones, zeros)
                    predict2 = torch.where(output2 > 0.8 * max_score2, ones, zeros)
                    predict3 = torch.where(output3 > 0.8 * max_score3, ones, zeros)

                    predict = w[0]*predict0+w[1]*predict1+w[2]*predict2+w[3]*predict3

                    max_score, max_index = torch.max(predict, 1)
                    predict = torch.where(predict > 0.8 * max_score, ones, zeros)

                    if torch.sum(label)>0:
                        #acc = torch.sum(predict * label)/torch.sum(label)
                        if torch.sum(predict * label)>0:
                            acc_item = 1.0
                        else:
                            acc_item = 0.0
                    else:
                        acc_item = 0.0
                        #acc = torch.sum(predict * label)
                    #acc_item = acc.item()
                    loss = criterion(output, label)
                    loss_test += loss
                    acc_test += acc_item
                    print(acc_test/n)
            acc_test = acc_test/n
            loss_test = loss_test/n
            loss_test = loss_test.item()
            loss_test_list.append(loss_test)
            t2 = time.time()
            print('=======================================test=======================================')
            print("loss_test:", loss_test,"acc_test:",acc_test,"time:", t2 - t1, "min loss_test:", min(loss_test_list))
            with open(run_log_name, "a") as log_file:
                log_file.write("loss_test:{},acc_test:{},time:{}, min loss_test:{}\n".format(loss_test,acc_test,t2 - t1, min(loss_test_list)))
            print('=======================================test=======================================')
            if writer is not None:
                with open(run_log_name, "a") as log_file:
                    log_file.write("loss@{}: {},acc_test@{}: {}\n".format(step, loss_test,step,acc_test))
                writer.add_scalar('loss', loss_test, step)
                writer.add_scalar('acc', acc_test, step)
                step += 1

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PCP_model training and testing', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
