import os
import torch
import torch.nn as nn
from time import time
from metrics import Evaluator
from argsparser import create_parser
from make_data import create_rainloader
from metrics import save_score, ed_cal_mse_ssim
from model.STMixGAN import STMixNet


class Predictor:
    def __init__(self, args):
        super(Predictor, self).__init__()
        self.args = args
        self._acquire_device()
        self._preparation()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        else:
            print('Use CPU')

    def _preparation(self):
        self._get_data()
        self._get_evaluator()
        self._build_model()

    def _get_data(self):
        self.test_loader = create_rainloader(
            self.args.test_file, self.args.batch_size, self.args.num_workers)

    def _get_evaluator(self):
        self.evaluator = Evaluator(
            rainfall_thresholds=self.args.rainfall_thresholds)

    def _build_model(self):
        self.gen = STMixNet(input_channel=5, output_channel=1)

        if torch.cuda.device_count() > 1:
            self.gen = nn.DataParallel(self.gen, device_ids=[0, 1])
        self.gen.cuda()

    def predict(self):
        self.gen.eval()
        start = time()
        weight = torch.load(
            f"{self.args.G_weight_dir}stmixnet_{self.args.epoch_weight}.pth")
        self.gen.load_state_dict(weight)

        with torch.no_grad():
            context = []
            output = []
            groundtruth = []

            for test_imgs in self.test_loader:
                test_imgs = test_imgs.type(torch.cuda.FloatTensor)
                input = test_imgs[:, :5]
                target = test_imgs[:, 5:]

                gen_input = input
                pred_imgs = []
                for _ in range(target.shape(1)):
                    pred_img = self.gen(gen_input)
                    pred_imgs.append(pred_img)
                    gen_input = torch.cat(
                        [gen_input, pred_img.unsqueeze(-3)], dim=1)
                    gen_input = gen_input[:, -5:]

                context.append(input)
                groundtruth.append(target)
                output.append(pred_img)

            context = torch.cat(context, dim=0)
            groundtruth = torch.cat(groundtruth, dim=0)
            pred_imgs = torch.cat(output, dim=0)

        end = time()
        print("Forecasting Time Consumption：{:.0f}min {:.1f}s".format(
            (end - start) // 60, (end - start) % 60))

        # Calculation of the radar prediction judgement indicator
        mae_s, mse_s, mssim_s, psnr_s = ed_cal_mse_ssim(groundtruth, pred_imgs)
        print(f"STMixNet_{self.args.epoch_weight}:")
        print(groundtruth.min(), groundtruth.max())
        print(pred_imgs.min(), pred_imgs.max())
        print(f"MAE: {mae_s}")
        print(f"MSE: {mse_s}")
        print(f"MSSIM: {mssim_s}")
        print(f"PSNR: {psnr_s}")

        # Preservation indicators
        score_save_path = (
            f"{self.args.metric_dir}/STMixNet_{self.args.epoch_weight}.txt")
        save_score(groundtruth, pred_imgs, score_save_path, self.evaluator)


if __name__ == '__main__':
    args = create_parser().parse_args()

    # predict
    predictor = Predictor(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>Start of forecasting<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    predictor.predict()
