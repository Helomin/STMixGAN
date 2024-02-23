import argparse


def create_parser():
    parser = argparse.ArgumentParser()

    # Set-up parameters
    parser.add_argument(
        '--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument(
        '--G_weight_dir', default='./save/weight/g_try_d/', type=str)
    parser.add_argument(
        '--D_weight_dir', default='./save/weight/d_try_g/', type=str)
    parser.add_argument(
        '--use_gpu', default=True, type=bool)
    parser.add_argument(
        '--gpu', default=0, type=int)

    # Train parameters
    parser.add_argument(
        '--batch_size', default=10, type=int)
    parser.add_argument(
        '--train_file', default='./Data/train15.npy')
    parser.add_argument(
        '--test_file', default='./Data/test15.npy')
    parser.add_argument(
        '--nt', default=6, type=int)
    parser.add_argument(
        '--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument(
        '--lr_beta1', default=0.5, type=float)
    parser.add_argument(
        '--lr_beta2', default=0.9, type=float)

    # Predict parameters
    parser.add_argument(
        '--epoch_weight', default=90, type=int)
    parser.add_argument(
        '--rainfall_thresholds', default=[2, 5, 10, 30], type=float, nargs='*')

    # Model parameters
    parser.add_argument(
        '--stack_sizes', default=[1, 128, 128, 256], type=int, nargs='*')
    parser.add_argument(
        '--frame', default=10, type=int)
    parser.add_argument(
        '--input_img_num', default=5, type=int)
    parser.add_argument(
        '--channel', default=1, type=int)
    parser.add_argument(
        '--img_rows', default=128, type=int)
    parser.add_argument(
        '--img_cols', default=128, type=int)

    return parser
