import os
from os.path import join
import SimpleITK as sitk
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np


from models.UNet import UNet_reg, UNet_seg, VNet1
from utils.STN import SpatialTransformer, Re_SpatialTransformer
from utils.augmentation import SpatialTransform
from utils.dataloader_brain_train import DatasetFromFolder3D as DatasetFromFolder3D_train
from utils.dataloader_brain_test_reg import DatasetFromFolder3D as DatasetFromFolder3D_test_reg
from utils.dataloader_brain_test_seg import DatasetFromFolder3D as DatasetFromFolder3D_test_seg
from utils.losses import gradient_loss, ncc_loss, MSE, dice_loss, softmax_mse_loss, softmax_kl_loss, l2_regularization
from utils.utils import AverageMeter
from utils import ramps

class JSR(object):
    def __init__(self, k=0,
                 n_channels=1,
                 n_classes=2,
                 reg_lr=1e-4,
                 epoches=100,  #200
                 iters=100,  #200
                 batch_size=1,
                 is_aug=True,
                 shot=30,
                 Sweight=100,
                 Rweight=1,
                 seg_lr=0.01,  # Seger_model
                 consistency=0.1,
                 consistency_type='mse',
                 consistency_rampup=40.0,
                 patches=None,
                 ema_decay=0.99,
                 labeled_dir='E:/Task04_Hippocampus/label_dir/',
                 unlabeled_dir='E:/Task04_Hippocampus/unlabel_dir/',
                 checkpoint_dir='E:/baseTrainJSR/pytorch_lung/Seger/Hippocampus/weights/',
                 result_dir='E:/baseTrainJSR/pytorch_lung/Seger/Hippocampus/results/',
                 model_name='seg+reg1'):
        super(JSR, self).__init__()

        # initialize parameters
        self.k = k
        self.patches = patches
        self.n_classes = n_classes
        self.epoches = epoches
        self.iters = iters
        self.reg_lr = reg_lr
        self.seg_lr = seg_lr
        self.is_aug = is_aug
        self.shot = shot
        self.consistency_type = consistency_type
        self.ema_decay = ema_decay
        self.iter_num = 0

        self.labeled_dir = labeled_dir
        self.unlabeled_dir = unlabeled_dir

        self.results_dir = result_dir
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name

        self.Sweight = Sweight
        self.Rweight = Rweight

        # self.base_lr = base_lr
        self.consistency = consistency
        self.consistency_rampup = consistency_rampup

        # tools
        self.stn = SpatialTransformer() # Spatial Transformer
        self.softmax = nn.Softmax(dim=1)

        # data augmentation
        self.spatial_aug = SpatialTransform(do_rotation=True,
                                            angle_x=(-np.pi / 9, np.pi / 9),
                                            angle_y=(-np.pi / 9, np.pi / 9),
                                            angle_z=(-np.pi / 9, np.pi / 9),
                                            do_scale=True,
                                            scale=(0.75, 1.25))

        # initialize networks
        self.Reger = UNet_reg(n_channels=n_channels)
        self.Seger = VNet1(n_channels=n_channels, n_classes=n_classes, normalization='batchnorm', has_dropout=True)
        self.Seger_ema = VNet1(n_channels=n_channels, n_classes=n_classes, normalization='batchnorm', has_dropout=True)
        for param in self.Seger_ema.parameters():
            param.detach_()

        # self.Seger = self.create_model()                  # student model
        # self.Seger_ema = self.create_model(ema=True)      # teacher model

        if torch.cuda.is_available():
            self.Reger = self.Reger.cuda()
            self.Seger = self.Seger.cuda()
            self.Seger_ema = self.Seger_ema.cuda()

        # initialize optimizer
        self.optR = torch.optim.Adam(self.Reger.parameters(), lr=reg_lr)
        # self.optS = torch.optim.Adam(self.Seger.parameters(), lr=lr)
        self.optS = torch.optim.SGD(self.Seger.parameters(), lr=seg_lr, momentum=0.9, weight_decay=0.0001)

        # initialize dataloader
        train_dataset = DatasetFromFolder3D_train(self.labeled_dir, self.unlabeled_dir, self.n_classes, shot=self.shot)
        self.dataloader_train = DataLoader(train_dataset, batch_size=batch_size)
        test_dataset_seg = DatasetFromFolder3D_test_seg(self.labeled_dir, self.n_classes, shot=self.shot)
        self.dataloader_test_seg = DataLoader(test_dataset_seg, batch_size=batch_size)
        test_dataset_reg = DatasetFromFolder3D_test_reg(self.labeled_dir, self.n_classes, shot=self.shot)
        self.dataloader_test_reg = DataLoader(test_dataset_reg, batch_size=batch_size)

        # define loss log
            # losses in registration
        self.L_smooth_log = AverageMeter(name='L_smooth')
        self.L_sim_log = AverageMeter(name='L_sim')
        self.L_seg_log = AverageMeter(name='L_seg')
        self.L_i_log = AverageMeter(name='L_i')
        self.L_Reg_log = AverageMeter(name='L_Reg')
            # losses in segmentation
        self.L_sup_log = AverageMeter(name='L_sup')
        self.L_cons_log = AverageMeter(name='L_cons')
        self.L_pseu_log = AverageMeter(name='L_pseu')
        self.L_L2_log = AverageMeter(name='L_L2')
        self.L_Seg_log = AverageMeter(name='L_Seg')

        if self.consistency_type == 'mse':
            self.consistency_criterion = softmax_mse_loss
        elif self.consistency_type == 'kl':
            self.consistency_criterion = softmax_kl_loss
        else:
            assert False, self.consistency_type

    def train_iterator(self, labeled_img, labeled_lab, unlabeled_img, epoch):
        # train Reger
        for p in self.Seger.parameters():  # reset requires_grad
            p.requires_grad = False  # they are set to False below in Seger update
        for p in self.Reger.parameters():  # reset requires_grad             -
            p.requires_grad = True  # they are set to True below in Reger update

        # forward deformation
        warpped, flow = self.Reger(labeled_img, unlabeled_img)

        # inverse deformation
        i_warpped, i_flow = self.Reger(unlabeled_img, labeled_img)

        # calculate loss
        loss_smooth = gradient_loss(flow) + gradient_loss(i_flow)                               # smooth loss
        self.L_smooth_log.update(loss_smooth.data, labeled_img.size(0))

        loss_sim = ncc_loss(warpped, unlabeled_img) + ncc_loss(i_warpped, labeled_img)          # similarity loss
        self.L_sim_log.update(loss_sim.data, labeled_img.size(0))

        loss_i = MSE(-1*self.stn(flow, flow), i_flow)                                      # inverse loss
        self.L_i_log.update(loss_i.data, labeled_img.size(0))

        warp_label = self.stn(labeled_lab, flow)
        seg_label = self.softmax(self.Seger(unlabeled_img)).detach()
        i_w_seg_label = self.stn(seg_label, i_flow)
        loss_seg = dice_loss(warp_label, seg_label) + dice_loss(i_w_seg_label, labeled_lab)
        self.L_seg_log.update(loss_seg.data, labeled_img.size(0))                               # segmentation loss

        loss_Reg = loss_smooth + loss_sim + self.Sweight * loss_seg + 0.1*loss_i
        self.L_Reg_log.update(loss_Reg.data, labeled_img.size(0))

        loss_Reg.backward()
        self.optR.step()
        self.Reger.zero_grad()
        self.optR.zero_grad()

        # train Seger
        for p in self.Seger.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to True below in Seger update
        for p in self.Reger.parameters():  # reset requires_grad             -
            p.requires_grad = False  # they are set to False below in Reger update

        # output of reger model
        warpped1, flow1 = self.Reger(labeled_img, unlabeled_img)
        # warpped2, flow2 = self.Reger(warpped1, unlabeled_img)
        # flow = flow1 + flow2                                              #add
        # flow = self.stn(flow1, flow2) + flow2                             #agg
        # loss_pseu = dice_loss(unlabeled_soft, warp_label)
        warp_label1 = self.stn(labeled_lab, flow1)
        # warp_label2 = self.stn(warp_label1, flow2)
        # alpha = np.random.beta(0.3, 0.3)
        # style = alpha * (warpped1 - unlabeled_img)
        # style = alpha * (warpped2 - unlabeled_img)

        # output of student model
        labeled_output = self.Seger(labeled_img)
        unlabeled_output = self.Seger(unlabeled_img)

        # output of teacher model
        noise = torch.clamp(torch.randn_like(unlabeled_img) * 0.1, -0.2, 0.2)
        ema_input = unlabeled_img + noise
        with torch.no_grad():
            ema_output = self.Seger_ema(ema_input)
        T = 8
        unlabeled_batch = unlabeled_img.repeat(2, 1, 1, 1, 1)
        stride = unlabeled_batch.shape[0] // 2
        preds = torch.zeros([stride * T, 2, 32, 64, 32]).cuda()
        for i in range(T // 2):
            ema_inputs = unlabeled_batch + torch.clamp(torch.randn_like(unlabeled_batch) * 0.1, -0.2, 0.2)
            with torch.no_grad():
                preds[2 * stride * i:2 * stride * (i + 1)] = self.Seger_ema(ema_inputs)
        preds = self.softmax(preds)
        preds = preds.reshape(T, stride, 2, 32, 64, 32)
        preds = torch.mean(preds, dim=0)  # (batch, 2, 128, 144, 144)
        uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)/8   # (batch, 1, 128, 144, 144)

        # calculate the loss
            # supervised loss
        labeled_soft = self.softmax(labeled_output)                     # labeled segmentation with label
        loss_sup = dice_loss(labeled_soft, labeled_lab)
        self.L_sup_log.update(loss_sup.data, labeled_img.size(0))
            # consistency_loss
        consistency_weight = self.get_current_consistency_weight(self.iter_num // 150)
        consistency_dist = self.consistency_criterion(unlabeled_output, ema_output)  # (batch, 2, 112,112,80)
        threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(self.iter_num, self.epoches * self.iters)) * np.log(2)
        mask = (uncertainty < threshold).float()
        consistency_dist = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)
        loss_cons = consistency_weight * consistency_dist
        # print(consistency_dist.item(), consistency_weight, loss_cons.item())
        self.L_cons_log.update(loss_cons.data, labeled_img.size(0))
            # pseudo_label_loss
        unlabeled_soft = self.softmax(unlabeled_output)
        loss_pseu = dice_loss(unlabeled_soft, warp_label1)                  #R2
        self.L_pseu_log.update(loss_pseu.data, labeled_img.size(0))
            # l2_regularization_loss
        # loss_l2 = 0.1 * l2_regularization(self.Seger, 0.001)
        # self.L_L2_log.update(loss_l2.data, labeled_img.size(0))

        loss_Seg = loss_sup + loss_cons + self.Rweight * loss_pseu        # Rweight=0.01
        # loss_Seg = loss_sup + loss_cons
        self.L_Seg_log.update(loss_Seg.data, labeled_img.size(0))

        loss_Seg.backward()
        self.optS.step()
        self.Seger.zero_grad()
        self.optS.zero_grad()
        self.update_ema_variables(self.Seger, self.Seger_ema, self.ema_decay, self.iter_num)

        self.iter_num = self.iter_num + 1

        # loss_seg = self.L_seg(self.softmax(s_labeled_img), labeled_lab) + 0.01*self.L_seg(self.softmax(s_unlabeled_img), warp_label2)
        # self.L_seg_log.update(loss_seg.data, labeled_img.size(0))

    def train_epoch(self, epoch):
        self.Reger.train()
        self.Seger.train()
        self.Seger_ema.train()

        for i in range(self.iters):
            labeled_img, labeled_lab, unlabeled_img = next(self.dataloader_train.__iter__())

            if torch.cuda.is_available():
                labeled_img = labeled_img.cuda()
                labeled_lab = labeled_lab.cuda()
                unlabeled_img = unlabeled_img.cuda()

            if self.is_aug:
                code_spa = self.spatial_aug.rand_coords(labeled_img.shape[2:])
                labeled_img = self.spatial_aug.augment_spatial(labeled_img, code_spa)
                labeled_lab = self.spatial_aug.augment_spatial(labeled_lab, code_spa, mode='nearest')
                unlabeled_img = self.spatial_aug.augment_spatial(unlabeled_img, code_spa)

            self.train_iterator(labeled_img, labeled_lab, unlabeled_img, epoch)
            res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, self.epoches),
                             'Iter: [%d/%d]' % (i + 1, self.iters),
                             self.L_smooth_log.__str__(),
                             self.L_sim_log.__str__(),
                             self.L_seg_log.__str__(),
                             self.L_i_log.__str__(),
                             self.L_Reg_log.__str__(),
                             self.L_sup_log.__str__(),
                             self.L_cons_log.__str__(),
                             self.L_pseu_log.__str__(),
                             self.L_Seg_log.__str__()])
            print(res)

    def test_iterator_seg(self, mi):
        with torch.no_grad():
            # Seg
            s_m = self.Seger(mi)
        return s_m

    def test_iterator_reg(self, mi, fi, ml=None):
        with torch.no_grad():
            # Reg
            warpped, flow = self.Reger(mi, fi)
            warp_label = self.stn(ml, flow, mode='nearest')

        return warpped, warp_label, flow

    def test(self):
        self.load()
        self.Seger.eval()
        self.Reger.eval()
        Seg_Dice = []
        Reg_Dice = []

        for i, (mi, ml, name) in enumerate(self.dataloader_test_seg):
            name = name[0]
            if torch.cuda.is_available():
                mi = mi.cuda()
            seg_mi = self.test_iterator_seg(mi)
            seg_mi = np.argmax(seg_mi.data.cpu().numpy()[0], axis=0)
            seg_mi = seg_mi.astype(np.int8)

            ml = np.argmax(ml.numpy()[0], axis=0)
            ml = ml.astype(np.int8)

            seg_dice = self.compute_dice(seg_mi, ml)
            Seg_Dice.append(seg_dice)

            if not os.path.exists(join(self.results_dir, self.model_name, 'seg')):
                os.makedirs(join(self.results_dir, self.model_name, 'seg'))

            seg_mi = sitk.GetImageFromArray(seg_mi)
            sitk.WriteImage(seg_mi, join(self.results_dir, self.model_name, 'seg', name[:-4]+'.nii'))
            print(name[:-4]+'.nii')
            print(seg_dice)

        for i, (mi, ml, fi, fl, name1, name2) in enumerate(self.dataloader_test_reg):
            name1 = name1[0]
            name2 = name2[0]
            if name1 is not name2:
                if torch.cuda.is_available():
                    mi = mi.cuda()
                    fi = fi.cuda()
                    ml = ml.cuda()

                warpped, warp_label, flow = self.test_iterator_reg(mi, fi, ml)

                flow = flow.data.cpu().numpy()[0]
                warpped = warpped.data.cpu().numpy()[0, 0]
                warp_label = np.argmax(warp_label.data.cpu().numpy()[0], axis=0)

                flow = flow.astype(np.float32)
                warpped = warpped.astype(np.float32)
                warp_label = warp_label.astype(np.int8)

                fl = np.argmax(fl.numpy()[0], axis=0)
                fl = fl.astype(np.int8)

                reg_dice = self.compute_dice(warp_label, fl)
                Reg_Dice.append(reg_dice)

                if not os.path.exists(join(self.results_dir, self.model_name, 'flow')):
                    os.makedirs(join(self.results_dir, self.model_name, 'flow'))
                if not os.path.exists(join(self.results_dir, self.model_name, 'warpped')):
                    os.makedirs(join(self.results_dir, self.model_name, 'warpped'))
                if not os.path.exists(join(self.results_dir, self.model_name, 'warp_label')):
                    os.makedirs(join(self.results_dir, self.model_name, 'warp_label'))

                warpped = sitk.GetImageFromArray(warpped)
                sitk.WriteImage(warpped, join(self.results_dir, self.model_name, 'warpped', name2[:-4]+'_'+name1[:-4]+'.nii'))
                warp_label = sitk.GetImageFromArray(warp_label)
                sitk.WriteImage(warp_label, join(self.results_dir, self.model_name, 'warp_label', name2[:-4]+'_'+name1[:-4]+'.nii'))
                flow = sitk.GetImageFromArray(flow)
                sitk.WriteImage(flow, join(self.results_dir, self.model_name, 'flow', name2[:-4]+'_'+name1[:-4]+'.nii'))
                print(name2[:-4]+'_'+name1[:-4]+'.nii')
                print(reg_dice)

        print("mean_Seg_Dice: ", np.mean(Seg_Dice))
        print("mean_Reg_Dice: ", np.mean(Reg_Dice))

    def checkpoint(self, epoch, k):
        if not os.path.exists(join(self.checkpoint_dir, self.model_name)):
            os.makedirs(join(self.checkpoint_dir, self.model_name))
        if not os.path.exists(join(self.checkpoint_dir, self.model_name)):
            os.makedirs(join(self.checkpoint_dir, self.model_name))
        torch.save(self.Seger.state_dict(),
                   '{0}/{1}/{2}_epoch_{3}.pth'.format(self.checkpoint_dir, self.model_name, 'Seger', epoch+k),
                   _use_new_zipfile_serialization=False)
        torch.save(self.Reger.state_dict(),
                   '{0}/{1}/{2}_epoch_{3}.pth'.format(self.checkpoint_dir, self.model_name, 'Reger', epoch+k),
                   _use_new_zipfile_serialization=False)
        # torch.save(self.Reger.state_dict(),
        #            '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, 'Reger_'+self.model_name, epoch+k),
        #            _use_new_zipfile_serialization=False)

    def load(self):
        epoch = 40
        self.Reger.load_state_dict(
            torch.load('{0}/{1}/{2}_epoch_{3}.pth'.format(self.checkpoint_dir, self.model_name, 'Reger', epoch)))
        self.Seger.load_state_dict(
            torch.load('{0}/{1}/{2}_epoch_{3}.pth'.format(self.checkpoint_dir, self.model_name, 'Seger', epoch)))
        # self.Seger.load_state_dict(
        #     torch.load('{0}/{1}/{2}_epoch_{3}.pth'.format(self.checkpoint_dir, self.model_name, 'Seger', '20')))

    # def create_model(self, ema=False):
    #     # Network definition
    #     net = VNet(n_channels=1, n_classes=self.n_classes, normalization='batchnorm', has_dropout=True)
    #     model = net.cuda()
    #     if ema:
    #         for param in model.parameters():
    #             param.detach_()
    #     return model

    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.consistency * ramps.sigmoid_rampup(epoch, self.consistency_rampup)

    def update_ema_variables(self, model, ema_model, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    # def compute_dice(self, gt, pred):
    #     # 需要计算的标签类别，不包括背景和图像中不存在的区域
    #     cls_lst = [1, 2, 3, 4, 5, 6, 7]
    #     dice_lst = []
    #     for cls in cls_lst:
    #         dice = self.DSC(gt == cls, pred == cls)
    #         dice_lst.append(dice)
    #     return np.mean(dice_lst)

    def compute_dice(self, gt, pred):
        # 需要计算的标签类别，不包括背景和图像中不存在的区域
        dice = self.DSC(gt == 1, pred == 1)
        return dice

    def DSC(self, pred, target):
        smooth = 1e-5
        m1 = pred.flatten()
        m2 = target.flatten()
        intersection = (m1 * m2).sum()
        return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

    def train(self):
        # self.Reger.load_state_dict(torch.load('E:/registration project/JSR/MT/weights/noise_style/Reger_epoch_100.pth'))
        # self.Seger.load_state_dict(torch.load('E:/registration project/JSR/MT/weights/noise_style/Seger_epoch_100.pth'))
        # print('Reger has been pretrained.')
        for epoch in range(self.epoches-self.k):
            self.L_smooth_log.reset()
            self.L_sim_log.reset()
            self.L_seg_log.reset()
            self.L_i_log.reset()
            self.L_Reg_log.reset()

            self.L_sup_log.reset()
            self.L_cons_log.reset()
            self.L_pseu_log.reset()
            self.L_L2_log.reset()
            self.L_Seg_log.reset()
            self.train_epoch(epoch+self.k)
            if epoch % 10 == 0:
                self.checkpoint(epoch, self.k)
                ## change lr
                if epoch % 30 == 0:
                    lr_ = self.seg_lr * 0.1 ** (epoch // 20)
                    for param_group in self.optS.param_groups:
                        param_group['lr'] = lr_
        self.checkpoint(self.epoches-self.k, self.k)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"    #0
    JSRNet = JSR()
    # JSRNet.train()
    JSRNet.test()
