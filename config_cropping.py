import os

class Config:
    data_root = './aesthetic_cropping/dataset/'
    predefined_pkl = os.path.join(data_root, 'pdefined_anchors.pkl') # download from https://github.com/luwr1022/listwise-view-ranking/blob/master/pdefined_anchors.pkl
    FCDB_dir = os.path.join(data_root, 'FCDB')
    FLMS_dir = os.path.join(data_root, 'FLMS')
    KUPCP_dir = './aesthetic_cropping/dataset/KU_PCP'

    image_size = (224,224)
    data_augmentation = True
    keep_aspect_ratio = False

    backbone = 'vgg16'
    # training
    gpu_id = 0
    num_workers = 1
    crop_batch_size = 8
    crop_loss_factor = 0.7
    com_loss_factor  = 0.3

    max_epoch = 60
    lr_decay_epoch = [30,60]
    lr = 1e-4
    lr_decay = 0.1
    weight_decay = 1e-4
    eval_freq = 5
    save_freq = 5
    save_image_freq = 200
    display_freq = 50

    prefix = 'cropping_{}croploss_{}classifyloss'.format(crop_loss_factor, com_loss_factor)
    exp_root = os.path.join(os.getcwd(), './experiments/')
    exp_name = prefix
    exp_path = os.path.join(exp_root, prefix)
    while os.path.exists(exp_path):
        index = os.path.basename(exp_path).split(prefix)[-1].split('repeat')[-1]
        try:
            index = int(index) + 1
        except:
            index = 1
        exp_name = prefix + ('_repeat{}'.format(index))
        exp_path = os.path.join(exp_root, exp_name)
    # print('Experiment name {} \n'.format(os.path.basename(exp_path)))
    checkpoint_dir = os.path.join(exp_path, 'checkpoints')
    log_dir = os.path.join(exp_path, 'logs')

    def create_path(self):
        print('Create experiment directory: ', self.exp_path)
        os.makedirs(self.exp_path)
        os.makedirs(self.checkpoint_dir)
        os.makedirs(self.log_dir)

cfg = Config()

if __name__ == '__main__':
    cfg = Config()