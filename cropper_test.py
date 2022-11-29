import os
import numpy as np
import time
import torch
from tqdm import tqdm
import cv2
import json
from Cropping_dataset import FCDBDataset, FLMSDataset
from config_cropping import cfg
from cropping_net import ImgCropper
import warnings
warnings.filterwarnings("ignore")

from CACNet import CACNet


device = torch.device('cuda:{}'.format(cfg.gpu_id))
# device = torch.device('cpu')

# results_dir = './results/' + time.strftime('%m_%d_%M')
# os.makedirs(results_dir, exist_ok=True)

def compute_iou_and_disp(gt_crop, pre_crop, im_w, im_h):
    ''''
    :param gt_crop: [[x1,y1,x2,y2]]
    :param pre_crop: [[x1,y1,x2,y2]]
    :return:
    '''
    gt_crop = gt_crop[gt_crop[:,0] >= 0]
    zero_t  = torch.zeros(gt_crop.shape[0])
    over_x1 = torch.max(gt_crop[:,0], pre_crop[:,0])
    over_y1 = torch.max(gt_crop[:,1], pre_crop[:,1])
    over_x2 = torch.min(gt_crop[:,2], pre_crop[:,2])
    over_y2 = torch.min(gt_crop[:,3], pre_crop[:,3])
    over_w  = torch.max(zero_t, over_x2 - over_x1)
    over_h  = torch.max(zero_t, over_y2 - over_y1)
    inter   = over_w * over_h
    area1   = (gt_crop[:,2] - gt_crop[:,0]) * (gt_crop[:,3] - gt_crop[:,1])
    area2   = (pre_crop[:,2] - pre_crop[:,0]) * (pre_crop[:,3] - pre_crop[:,1])
    union   = area1 + area2 - inter
    iou     = inter / union
    disp    = (torch.abs(gt_crop[:, 0] - pre_crop[:, 0]) + torch.abs(gt_crop[:, 2] - pre_crop[:, 2])) / im_w + \
              (torch.abs(gt_crop[:, 1] - pre_crop[:, 1]) + torch.abs(gt_crop[:, 3] - pre_crop[:, 3])) / im_h
    iou_idx = torch.argmax(iou, dim=-1)
    dis_idx = torch.argmin(disp, dim=-1)
    index   = dis_idx if (iou[iou_idx] == iou[dis_idx]) else iou_idx
    return iou[index].item(), disp[index].item()

def evaluate_on_FCDB_and_FLMS(model, dataset, save_results=False):
    results_dir = './results/' + time.strftime('%m_%d_%M')
    os.makedirs(results_dir, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device
    accum_disp = 0
    accum_iou  = 0
    crop_cnt = 0
    alpha = 0.75
    alpha_cnt = 0
    cnt = 0

    if save_results:
        save_file = os.path.join(results_dir, dataset + '.json')
        crop_dir  = os.path.join(results_dir, dataset)
        os.makedirs(crop_dir, exist_ok=True)
        test_results = dict()

    cacnet = CACNet(loadweights=False)
    cacnet.load_state_dict(torch.load('./cac_model/best-FLMS_iou.pth', map_location=torch.device('cpu')))
    cacnet = cacnet.to(device).eval()

    print('=' * 5, f'Evaluating on {dataset}', '=' * 5)
    with torch.no_grad():
        if dataset == 'FCDB':
            test_set = [FCDBDataset]
        elif dataset == 'FLMS':
            test_set = [FLMSDataset]
        else:
            raise Exception('Undefined test set ', dataset)
        for dataset in test_set:
            test_dataset= dataset(split='test',
                                  keep_aspect_ratio=cfg.keep_aspect_ratio)
            test_loader = torch.utils.data.DataLoader(test_dataset,  batch_size=1,
                                                      shuffle=False, num_workers=cfg.num_workers,
                                                      drop_last=False)
            for batch_idx, batch_data in enumerate(tqdm(test_loader)):
                im = batch_data[0].to(device)
                gt_crop = batch_data[1] # x1,y1,x2,y2
                width = batch_data[2].item()
                height = batch_data[3].item()
                image_file = batch_data[4][0]
                image_name = os.path.basename(image_file)

                logits, kcm = cacnet(im, only_classify=True)

                crop = model(im,kcm)
                crop[:,0::2] = crop[:,0::2] / im.shape[-1] * width
                crop[:,1::2] = crop[:,1::2] / im.shape[-2] * height
                pred_crop = crop.detach().cpu()
                gt_crop = gt_crop.reshape(-1, 4)
                pred_crop[:,0::2] = torch.clamp(pred_crop[:,0::2], min=0, max=width)
                pred_crop[:,1::2] = torch.clamp(pred_crop[:,1::2], min=0, max=height)

                iou, disp = compute_iou_and_disp(gt_crop, pred_crop, width, height)
                if iou >= alpha:
                    alpha_cnt += 1
                accum_iou += iou
                accum_disp += disp
                cnt += 1

                if save_results:
                    best_crop = pred_crop[0].numpy().tolist()
                    best_crop = [int(x) for x in best_crop] # x1,y1,x2,y2
                    test_results[image_name] = best_crop

                    # save the best crop
                    source_img = cv2.imread(image_file)
                    croped_img  = source_img[best_crop[1] : best_crop[3], best_crop[0] : best_crop[2]]
                    if croped_img.size == 0:
                        pass
                    else:
                        cv2.imwrite(os.path.join(crop_dir, image_name), croped_img)
    if save_results:
        with open(save_file, 'w') as f:
            json.dump(test_results, f)
    avg_iou  = accum_iou / cnt
    avg_disp = accum_disp / (cnt * 4.0)
    avg_recall = float(alpha_cnt) / cnt
    print('Test on {} images, IoU={:.4f}, Disp={:.4f}, recall={:.4f}(iou>={:.2f})'.format(
        cnt, avg_iou, avg_disp, avg_recall, alpha
    ))
    return avg_iou, avg_disp

if __name__ == '__main__':
    # weight_file = "./pretrained_model/best-FLMS_iou.pth"
    model = ImgCropper(loadweights=False)
    # model.load_state_dict(torch.load(weight_file))
    model = model.to(device).eval()
    evaluate_on_FCDB_and_FLMS(model, dataset='FCDB', save_results=True)
    evaluate_on_FCDB_and_FLMS(model, dataset='FLMS', save_results=True)