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
import cv2
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler


from models import build_model
from pc_datasets import build_datasets
from tensorboardX import SummaryWriter
import warnings

warnings.filterwarnings('ignore')

def save_picture(path,img_original):
    img = img_original.cpu().clone()
    aaa = img.size()
    img = img.squeeze(0)
    img = img[0, :, :]
    img = np.array(img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(path, img)

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training PC_classfier', add_help=False)
    # SGD使用1e-2开始训练
    # ADAM使用1e-4开始训练
    # Adam,SGD
    parser.add_argument('--lr',default = 1e-4,type=float)
    parser.add_argument('--lr_backbone', default=1e-4,type=float)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--optimizer', default='Adam', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=100, type=int)
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
    parser.add_argument('--backbone_saliency_path', default='pre_checkpoints/saliency.pth', type=str,
                        help='set the seed for reproducing the result')
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
    parser.add_argument('--resume',default='./ckpt/latest_vgg16_bn_Adam_aug_on.pth',help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--gpu_id', default=2, type=int, help='the gpu used for training')
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
    run_log_name = 'run_log_sal_'+filename_tail+'.txt'
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
    model, criterion = build_model(args, training=True)
    # move to GPU
    model.to(device)
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
        }
        # {
        #     "params": [p for n, p in model_without_ddp.named_parameters() if "body" in n and p.requires_grad],
        #     "lr": args.lr_backbone_saliency,
        # },
        # {
        #     "params": [p for n, p in model_without_ddp.named_parameters() if "encoder" in n and p.requires_grad],
        #     "lr": args.lr_backbone_saliency,
        # }
    ]
    # SGD is used by default
    if args.optimizer=='SGD' or args.optimizer == None:
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
        # t1 = time.time()
        # running_loss = 0
        # n = 0
        # model.train()
        # for index,data in enumerate(data_loader_train):
        #     n = n+1
        #     input = data['image']
        #     label = data['label']
        #     input = input.to(device)
        #     label = label.to(device)
        #     output = model(input)
        #     #loss = criterion(output, label,model)
        #     loss = criterion(output,label)
        #
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     running_loss += loss.item()
        # loss_epoch.append(running_loss/n)
        # t2 = time.time()
        # if writer is not None:
        #     with open(run_log_name,'a') as log_file:
        #         log_file.write("{} epoch loss:{:.5f}, cost time: {:.2f}s \n".format(epoch,loss_epoch[epoch],t2-t1))
        #     writer.add_scalar('loss/loss',loss_epoch[epoch], epoch)
        # print("{} epoch loss:{:.5f}, cost time: {} s ".format(epoch,loss_epoch[epoch],t2-t1))
        # lr_scheduler.step()
        # checkpoint_latest_path = 'latest_sal_' + filename_tail + '.pth'
        # checkpoint_latest_path = os.path.join(args.checkpoints_dir,checkpoint_latest_path)
        # torch.save({
        #     'model': model.state_dict(),
        #     'epoch': epoch
        # }, checkpoint_latest_path)

        if epoch % args.eval_freq == 0 and epoch != 0:
            t1 = time.time()
            model.eval()
            loss_test = 0
            n = 0
            acc_test = 0
            with torch.no_grad():
                base = torch.zeros((1,9)).to(device)
                result = torch.zeros((1,9)).to(device)
                for index, data in enumerate(data_loader_test):
                    n = n+1
                    input = data['image']
                    label = data['label']
                    input = input.to(device)
                    label = label.to(device)
                    base = base + label
                    output = model(input)
                    x1,x2,x3,x4 = model.feature_get(input)
                    save_picture('aaaaa.png',x1)
                    max_score, max_index = torch.max(output,1)
                    ones = torch.ones_like(output)
                    zeros = torch.zeros_like(output)
                    predict = torch.where(output>0.8*max_score, ones, zeros)
                    if torch.sum(label)>0:
                        #acc = torch.sum(predict * label)/torch.sum(label)
                        r = predict * label
                        result = result+r
                        if torch.sum(predict * label)>0:
                            acc_item = 1.0
                        else:
                            acc_item = 0.0
                    else:
                        acc_item = 0.0
                        #acc = torch.sum(predict * label)
                    #acc_item = acc.item()
                    loss = criterion(output, label)
                    #loss = criterion(output, label,model)
                    loss_test += loss
                    acc_test += acc_item
                    # print(base)
                    # print(r)
                    # print(result/(base+0.00000001))
                    # print(acc_test/n)
            acc = (result / (base + 0.00000001))*100
            acc_test = acc_test/n
            loss_test = loss_test/n
            loss_test = loss_test.item()
            loss_test_list.append(loss_test)
            t2 = time.time()
            print('=======================================test=======================================')
            print("loss_test:", loss_test,"acc_test:",acc_test,"time:", t2 - t1, "min loss_test:", min(loss_test_list))
            print("不同构图的正确率比较:")
            print('rule of thirds(RoT):{}%\n'
                  'vertical:{:.2f}%\n'
                  'horizontal:{:.2f}%\n'
                  'diagonal:{:.2f}%\n'
                  'curved:{:.2f}%\n'
                  'triangle:{:.2f}%\n'
                  'center:{:.2f}\n'
                  'symmetric:{:.2f}\n'
                  'pattern:{:.2f}%'.format(acc[0,0],acc[0,1],acc[0,2],acc[0,3],acc[0,4],acc[0,5],acc[0,6],acc[0,7],acc[0,8]))
            with open(run_log_name, "a") as log_file:
                log_file.write("loss_test:{},acc_test:{},time:{}, min loss_test:{}\n".format(loss_test,acc_test,t2 - t1, min(loss_test_list)))
            print('=======================================test=======================================')
            if writer is not None:
                with open(run_log_name, "a") as log_file:
                    log_file.write("loss@{}: {},acc_test@{}: {}\n".format(step, loss_test,step,acc_test))
                writer.add_scalar('loss', loss_test, step)
                writer.add_scalar('acc', acc_test, step)
                step += 1

            # save the bese model since begining
            # if abs(min(loss_test_list)-loss_test)<0.01:
            #     checkpoint_latest_path = 'best_loss_sal_' + filename_tail + '.pth'
            #     checkpoint_best_path = os.path.join(args.checkpoints_dir, checkpoint_latest_path)
            #     torch.save({
            #         'model': model.state_dict(),
            #     }, checkpoint_best_path)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PCP_model training and testing', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
