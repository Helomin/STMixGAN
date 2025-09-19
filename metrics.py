import torch
import numpy as np
import properscoring as ps
from scipy.stats import norm
from pysteps.verification.spatialscores import fss
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

def MAE(pred, target):
    return np.mean(np.abs(pred - target), dtype=np.float64)

def MSE(pred, target):
    return np.mean((pred - target)**2, dtype=np.float64)

def cal_image_metrics(target_imgs, pred_images):
    target_imgs = target_imgs.cpu().detach().numpy()
    pred_images = pred_images.cpu().detach().numpy()

    mae_s = MAE(pred_images, target_imgs)
    mse_s = MSE(pred_images, target_imgs)
    
    ssim_s = 0.
    for i in range(target_imgs.shape[0]):
        for j in range(target_imgs.shape[1]):
            ssim_s += SSIM(pred_images[i, j], target_imgs[i, j], data_range=1)
    ssim_s = ssim_s / target_imgs.shape[0] / target_imgs.shape[1]

    psnr_s = 0.
    for i in range(target_imgs.shape[0]):
        for j in range(target_imgs.shape[1]):
            psnr_s += PSNR(pred_images[i, j], target_imgs[i, j], data_range=1)
    psnr_s = psnr_s / target_imgs.shape[0] / target_imgs.shape[1]
    
    return mae_s, mse_s, ssim_s, psnr_s

def save_score(test_images, pred_images, save_file, evaluator):
    test_images = test_images[:, 5:, 0]
    pred_images = pred_images[:, 5:, 0]
    ets, pod, far, csi, hss, bias, pooled_crps, sum_fss = evaluator.cal_score(pred_images, test_images)
    with open(save_file, 'w') as f:
        for (i, threshold) in enumerate(evaluator.rainfall_thresholds):
            f.write("Threshold = %g:\n" % threshold)
            f.write("   ETS=%s\n" % str(list(ets[i, :].tolist())))
            f.write("   POD=%s\n" % str(list(pod[i, :].tolist())))
            f.write("   FAR=%s\n" % str(list(far[i, :].tolist())))
            f.write("   CSI=%s\n" % str(list(csi[i, :].tolist())))
            f.write("   HSS=%s\n" % str(list(hss[i, :].tolist())))
            f.write("   Bias=%s\n" % str(list(bias[i, :].tolist())))
            f.write("   ETS stat: avg %g/final %g\n" %
                    (torch.mean(ets[i, :]), ets[i, -1].item()))
            f.write("   POD stat: avg %g/final %g\n" %
                    (torch.mean(pod[i, :]), pod[i, -1].item()))
            f.write("   FAR stat: avg %g/final %g\n" %
                    (torch.mean(far[i, :]), far[i, -1].item()))
            f.write("   CSI stat: avg %g/final %g\n" %
                    (torch.mean(csi[i, :]), csi[i, -1].item()))
            f.write("   HSS stat: avg %g/final %g\n" %
                    (torch.mean(hss[i, :]), hss[i, -1].item()))
            f.write("   Bias stat: avg %g/final %g\n" %
                    (torch.mean(bias[i, :]), bias[i, -1].item()))
        
        for (i, threshold) in enumerate([1, 2, 4, 8]):
             f.write("Scale = %g:\n" % threshold)
             f.write("   Pooled_CRPS=%s\n" % str(list(pooled_crps[i, :].tolist())))

        for (i, threshold) in enumerate(evaluator.rainfall_thresholds):
            for (j, scale) in enumerate([1, 4, 16, 32]):
                f.write("Threshold = %g:\n" % threshold)
                f.write("   Scale = %g:\n" % scale)
                f.write("       Fss=%s\n" % str(list(sum_fss[i, j].tolist())))
    
class Evaluator:
    def __init__(self, pixel_thresholds: list = None):
        self.pixel_thresholds = pixel_thresholds
        self.scales = [1, 2, 4, 8]

    def cal_TP(self, pred=None, target=None, th=None):
        return (torch.where(torch.logical_and(pred >= th, target >= th), 1, 0).sum(dim=(-1, -2))).sum(dim=0)

    def cal_TN(self, pred=None, target=None, th=None):
        return (torch.where(torch.logical_and(pred < th, target < th), 1, 0).sum(dim=(-1, -2))).sum(dim=0)

    def cal_FP(self, pred=None, target=None, th=None):
        return (torch.where(torch.logical_and(pred >= th, target < th), 1, 0).sum(dim=(-1, -2))).sum(dim=0)

    def cal_FN(self, pred=None, target=None, th=None):
        return (torch.where(torch.logical_and(pred < th, target >= th), 1, 0).sum(dim=(-1, -2))).sum(dim=0)

    def cal_Dr(self, pred=None, target=None, th=None, TP=None, TN=None, FP=None, FN=None):
        """
        Dr = (TP + FP)*(TP + FN) / (TP + TN + FP + FN)
        """
        if TP is None and TN is None and FP is None and FN is None:
            TP = self.cal_TP(pred=pred, target=target, th=th)
            TP = self.cal_TN(pred=pred, target=target, th=th)
            FP = self.cal_FP(pred=pred, target=target, th=th)
            FN = self.cal_FN(pred=pred, target=target, th=th)
        return (TP + FP) * (TP + FN) / (TP + FN + FP + TN)

    def cal_POD(self, pred=None, target=None, th=None, TP=None, FN=None):
        """
        Probability of Detection = TP / (TP + FN)
        """
        if TP is None and FN is None:
            TP = self.cal_TP(pred=pred, target=target, th=th)
            FN = self.cal_FN(pred=pred, target=target, th=th)
        return TP / (TP + FN)

    def cal_FAR(self, pred=None, target=None, th=None, FP=None, TP=None):
        """
        False Alarm Rate = FP / (FP + TP)
        """
        if FP is None and TP is None:
            TP = self.cal_TP(pred=pred, target=target, th=th)
            FP = self.cal_FP(pred=pred, target=target, th=th)
        return FP / (FP + TP)

    def cal_CSI(self, pred=None, target=None, th=None, TP=None, FP=None, FN=None):
        """
        Critical Success Index = TP / (TP + FP + FN)
        """
        if TP is None and FP is None and FN is None:
            TP = self.cal_TP(pred=pred, target=target, th=th)
            FP = self.cal_FP(pred=pred, target=target, th=th)
            FN = self.cal_FN(pred=pred, target=target, th=th)
        return TP / (TP + FP + FN)

    def cal_HSS(self, pred=None, target=None, th=None, TP=None, TN=None, FP=None, FN=None):
        """
        Heidke Skill Score = 2 * (TP*TN-FN*FP) / ((TP+FN)*(FN+TN)+(TP+FP)*(FP+TN))
        """
        if TP is None and TN is None and FP is None and FN is None:
            TP = self.cal_TP(pred=pred, target=target, th=th)
            TN = self.cal_TN(pred=pred, target=target, th=th)
            FP = self.cal_FP(pred=pred, target=target, th=th)
            FN = self.cal_FN(pred=pred, target=target, th=th)
        return 2 * (TP*TN-FN*FP) / ((TP+FN)*(FN+TN)+(TP+FP)*(FP+TN))
    
    def cal_ETS(self, pred=None, target=None, th=None, TP=None, TN=None, FP=None, FN=None, Dr=None):
        """
        ETS = (TP - Dr)/ (TP + TN + FP - Dr)
        """
        if TP is None and TN is None and FP is None and FN is None and Dr is None:
            TP = self.cal_TP(pred=pred, target=target, th=th)
            TN = self.cal_TN(pred=pred, target=target, th=th)
            FP = self.cal_FP(pred=pred, target=target, th=th)
            FN = self.cal_FN(pred=pred, target=target, th=th)
            Dr = self.cal_Dr(pred=pred, target=target, th=th, TP=TP, TN=TN, FP=FP, FN=FN)
        return (TP - Dr) / (TP + FN + FP - Dr)

    def cal_CRPS(self, pred=None, target=None, scale=None):
        target_cdf = norm.cdf(x=target.cpu().detach().numpy(), loc=0, scale=scale)
        pred_cdf = norm.cdf(x=pred.cpu().detach().numpy(), loc=0, scale=scale)
        forecast_score = ps.crps_ensemble(target_cdf, pred_cdf).mean(axis=(0, -1, -2))
        return forecast_score

    def cal_FSS(self, pred=None, target=None, threshold=None, scale=None):
        fss_score = []
        for frame in range(target.shape[0]):
            fra_score = 0.0
            count = 0
            for j in range(target.shape[1]):
                if np.any(target[frame, j] > threshold):
                    count +=1
                    fra_score += fss(pred[frame, j], target[frame, j], threshold, scale)
            if count != 0:
                fra_score = fra_score / count
            fss_score.append(fra_score)
        return np.array(fss_score)

    def cal_score(self, pred, target):
        if torch.is_tensor(pred):
            pred = pred.detach()
        if torch.is_tensor(target):
            target = target.detach()

        def check_shape(x): return x.unsqueeze(0) if len(x.shape) == 3 else x
        pred = check_shape(pred)
        target = check_shape(target)

        if isinstance(pred, torch.Tensor):
            pred = torch.nan_to_num(pred, nan=0)
        if isinstance(target, torch.Tensor):
            target = torch.nan_to_num(target, nan=0)
        
        pooled_crps = []
        for scale in self.scales:
            pooled_crps.append(
                self.cal_CRPS(
                    F.avg_pool2d(pred, kernel_size=scale), F.avg_pool2d(target, kernel_size=scale), scale=scale))
        
        ets = []
        pod = []
        far = []
        csi = []
        hss = []
        bias = []

        for th in self.pixel_thresholds:
            TP = self.cal_TP(pred=pred, target=target, th=th)
            TN = self.cal_TN(pred=pred, target=target, th=th)
            FP = self.cal_FP(pred=pred, target=target, th=th)
            FN = self.cal_FN(pred=pred, target=target, th=th)
            Dr = self.cal_Dr(pred=pred, target=target, th=th, TP=TP, TN=TN, FP=FP, FN=FN)
            ets.append(self.cal_ETS(TP=TP, TN=TN, FP=FP, FN=FN, Dr=Dr))
            pod.append(self.cal_POD(TP=TP, FN=FN))
            far.append(self.cal_FAR(TP=TP, FP=FP))
            csi.append(self.cal_CSI(TP=TP, FP=FP, FN=FN))
            hss.append(self.cal_HSS(TP=TP, TN=TN, FP=FP, FN=FN))
            bias.append(self.cal_Bias(TP=TP, TN=TN, FP=FP, FN=FN))

        ets = torch.stack(ets)
        pod = torch.stack(pod)
        far = torch.stack(far)
        csi = torch.stack(csi)
        hss = torch.stack(hss)
        bias = torch.stack(bias)
        pooled_crps = np.array(pooled_crps)
        
        target = target.permute(1, 0, 2, 3)
        pred = pred.permute(1, 0, 2, 3)
        target = target.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        sum_fss = []
        for th in self.pixel_thresholds:
            scale_fss = [self.cal_FSS(pred, target, th, scale) for scale in [1, 4, 16, 32]]
            sum_fss.append(np.array(scale_fss))
        sum_fss = np.array(sum_fss)

        return ets, pod, far, csi, hss, bias, pooled_crps, sum_fss

