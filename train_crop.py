import os
import sys
import numpy as np
from tensorboardX import SummaryWriter
import torch
import time
import datetime
import csv
import shutil
import random
import torch.utils.data as data
import cv2

from models import build_model
from pc_datasets import build_datasets
import argparse

from Cropping_dataset import FCDBDataset
from config_cropping import cfg
from cropper_test import evaluate_on_FCDB_and_FLMS
from cropping_net import ImgCropper
from ensemble import voter
import warnings

warnings.filterwarnings("ignore")
from CACNet import CACNet

device = torch.device('cuda:{}'.format(cfg.gpu_id))
# device = torch.device('cpu')

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


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


def create_dataloader():
    crop_dataset = FCDBDataset(split='train', keep_aspect_ratio=cfg.keep_aspect_ratio)
    crop_loader = torch.utils.data.DataLoader(crop_dataset, batch_size=cfg.crop_batch_size,
                                              shuffle=True, num_workers=cfg.num_workers,
                                              drop_last=False, worker_init_fn=random.seed(SEED))
    print('FCDB training set has {} samples, batch_size={}, total {} batches'.format(
        len(crop_dataset), cfg.crop_batch_size, len(crop_loader)))

    return crop_loader


class Trainer:
    def __init__(self, model):
        self.model = model
        self.epoch = 0
        self.iters = 0
        self.max_epoch = cfg.max_epoch
        self.writer = SummaryWriter(log_dir=cfg.log_dir)
        self.optimizer, self.lr_scheduler = self.get_optimizer()
        self.crop_loader = create_dataloader()
        self.eval_results = []
        self.best_results = {'FCDB_iou': 0., 'FCDB_disp': 1.,
                             'FLMS_iou': 0., 'FLMS_disp': 1.}
        self.crop_criterion = torch.nn.SmoothL1Loss(reduction='mean')
        self.visual_path = os.path.join(cfg.exp_path, 'visualized_results')

    def get_optimizer(self):
        # params = [
        #     {'params': self.model.cropping_module.parameters(), 'lr': cfg.lr},
        #     {'params': self.model.composition_module.parameters(), 'lr': 1e-4},
        #     {'params': self.model.backbone.parameters(), 'lr': 1e-4}
        # ]
        optim = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim, milestones=cfg.lr_decay_epoch, gamma=cfg.lr_decay
        )
        return optim, lr_scheduler

    def run(self):
        print(("========  Begin Training  ========="))
        for epoch in range(1, self.max_epoch + 1):
            self.epoch = epoch
            self.train()
            if epoch % cfg.eval_freq == 0:
                self.eval()
                self.record_eval_results()
            self.lr_scheduler.step()

    def train(self):
        self.model.train()
        start = time.time()
        batch_crop_loss = 0
        total_batch = len(self.crop_loader)

        cacnet = CACNet(loadweights=False)
        cacnet.load_state_dict(torch.load('./cac_model/best-FLMS_iou.pth', map_location=torch.device('cpu')))
        cacnet = cacnet.to(device).eval()

        parser = argparse.ArgumentParser('PCP_model training and testing', parents=[get_args_parser()])
        args = parser.parse_args()

        args.backbone = 'densenet'
        path = './voters/' + 'densenet' + '.pth'
        dense, criterion = build_model(args)
        dense.to(device)
        criterion.to(device)
        ckpt = torch.load(path, map_location='cpu')
        dense.load_state_dict(ckpt['model'])
        dense.eval()

        for batch_idx, batch_data in enumerate(self.crop_loader):
            # ================ training on cropping task ===============
            self.iters += 1
            im = batch_data[0].to(device)
            crop = batch_data[1].to(device).squeeze(1)
            width = batch_data[2].to(device)
            height = batch_data[3].to(device)

            logits, kcm = cacnet(im, only_classify=True)
            origin = voter(im, dense)

            crop[:, 0::2] = crop[:, 0::2] / width[:, None] * im.shape[-1]
            crop[:, 1::2] = crop[:, 1::2] / height[:, None] * im.shape[-2]
            pred_crop = self.model(im, kcm).int()

            flag = True
            cnt = 0
            dif = 0
            croped_img = im
            for i in range(len(pred_crop)):
                if pred_crop[i][3] - pred_crop[i][1] <= 3 or pred_crop[i][2] - pred_crop[i][0] <= 3:
                    gap = 1
                    flag = False
                    break
                else:
                    croped_img[i] = im[i][pred_crop[1]:pred_crop[3], pred_crop[0]:pred_crop[2]]

            if flag:
                croped_com = voter(croped_img)
                for i in range(len(croped_com)):
                    for j in range(9):
                        cnt += 1
                        if croped_com[i, j] != origin[i, j]:
                            dif += 1
                gap = dif / cnt

            crop_loss = self.crop_criterion(pred_crop, crop) * (1 + gap)

            # print('gt {} v.s. pre {}'.format(crop.shape, pre_crop.shape))
            # print(crop, pre_crop)
            batch_crop_loss += crop_loss.item()
            crop_loss *= cfg.crop_loss_factor
            self.optimizer.zero_grad()
            crop_loss.requires_grad_(True)
            crop_loss.backward()

            # if self.iters % cfg.save_image_freq == 0:
            #     self.visualize_com_prediction(image_path, logits, kcm, labels)

            if batch_idx > 0 and batch_idx % cfg.display_freq == 0:
                avg_crop_loss = batch_crop_loss / (1 + batch_idx)

                cur_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/crop_loss', avg_crop_loss, self.iters)
                self.writer.add_scalar('train/lr', cur_lr, self.iters)

                time_per_batch = (time.time() - start) / (batch_idx + 1.)
                last_batches = (self.max_epoch - self.epoch - 1) * total_batch + (total_batch - batch_idx - 1)
                last_time = int(last_batches * time_per_batch)
                time_str = str(datetime.timedelta(seconds=last_time))

                print(
                    '=== epoch:{}/{}, step:{}/{} | Crop_Loss:{:.4f} | lr:{:.6f} | estimated last time:{} ==='.format(
                        self.epoch, self.max_epoch, batch_idx, total_batch, avg_crop_loss, cur_lr, time_str
                    ))

    def eval(self):
        FCDB_iou, FCDB_disp = evaluate_on_FCDB_and_FLMS(self.model, dataset='FCDB')
        FLMS_iou, FLMS_disp = evaluate_on_FCDB_and_FLMS(self.model, dataset='FLMS')
        self.eval_results.append([self.epoch, FCDB_iou, FCDB_disp, FLMS_iou, FLMS_disp])
        epoch_result = {'FCDB_iou': FCDB_iou, 'FCDB_disp': FCDB_disp,
                        'FLMS_iou': FLMS_iou, 'FLMS_disp': FLMS_disp}
        for m in self.best_results.keys():
            update = False
            if ('disp' not in m) and (epoch_result[m] > self.best_results[m]):
                update = True
            elif ('disp' in m) and (epoch_result[m] < self.best_results[m]):
                update = True
            if update:
                self.best_results[m] = epoch_result[m]
                checkpoint_path = os.path.join(cfg.checkpoint_dir, 'best-{}.pth'.format(m))
                torch.save(self.model.state_dict(), checkpoint_path)
                print('Update best {} model, best {}={:.4f}'.format(m, m, self.best_results[m]))
            if m in ['FCDB_iou', 'FLMS_iou']:
                self.writer.add_scalar('test/{}'.format(m), epoch_result[m], self.epoch)
                self.writer.add_scalar('test/best-{}'.format(m), self.best_results[m], self.epoch)

        if self.epoch > 0 and self.epoch % cfg.save_freq == 0:
            checkpoint_path = os.path.join(cfg.checkpoint_dir, 'epoch-{}.pth'.format(self.epoch))
            torch.save(self.model.state_dict(), checkpoint_path)

    def record_eval_results(self):
        csv_path = os.path.join(cfg.exp_path, '..', '{}.csv'.format(cfg.exp_name))
        header = ['epoch', 'FCDB_iou', 'FCDB_disp', 'FLMS_iou', 'FLMS_disp']
        rows = [header]
        for i in range(len(self.eval_results)):
            new_results = []
            for j in range(len(self.eval_results[i])):
                if header[j] == 'epoch':
                    new_results.append(self.eval_results[i][j])
                else:
                    new_results.append(round(self.eval_results[i][j], 4))
            self.eval_results[i] = new_results
        rows += self.eval_results
        metrics = [[] for i in header]
        for result in self.eval_results:
            for i, r in enumerate(result):
                metrics[i].append(r)
        for name, m in zip(header, metrics):
            if name == 'epoch':
                continue
            index = m.index(max(m))
            if 'disp' in name:
                index = m.index(min(m))
            title = 'best {}(epoch-{})'.format(name, index)
            row = [l[index] for l in metrics]
            row[0] = title
            rows.append(row)
        with open(csv_path, 'w') as f:
            cw = csv.writer(f)
            cw.writerows(rows)
        print('Save result to ', csv_path)


if __name__ == '__main__':
    cfg.create_path()
    for file in os.listdir('./'):
        if file.endswith('.py'):
            shutil.copy(file, cfg.exp_path)
            print('backup', file)
    net = ImgCropper(loadweights=True).to(device)
    trainer = Trainer(net)
    trainer.run()
