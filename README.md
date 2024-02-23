# STMixGAN
This is a radar echo extrapolation model, which can improve the accuracy of rainfall prediction.
Both the generator STMixNet and the discriminator are stored in the file model/STMixGAN.py.

The network is trained to be a predictive model that can infer the next ten frames from the first five frames, i.e., the first half hour is used as input to predict the distribution of the next hour.
STMixGAN consists of two components: the generator is a spatiotemporal mixed coding network (STMixNet), and the dual-channel network (DCNet) serves as a discriminator. The STMixNet is based on the whole echo evolution process and extracts global to local multi-scale features by capturing the spatiotemporal correlation of the radar echo context, which is useful to help avoid regression to the mean and blurring effects. DCNet guides the training of STMixNet by recognizing that the predictions are false, and the ground truth is true, to make its predictions more realistic.

# South China Radar Dataset
The experimental data is the radar mosaic of the South China area provided by the Guangdong Meteorological Bureau. It does not support open sharing. For data access, please contact Kun Zheng (ZhengK@cug.edu.cn) and Long He (Helomin@cug.edu.cn).

# STMixNet
You can adjust the parameters of your data.

# Loss
loss.py contains the loss function needed for our model.

# Note
You can cite the STMixGAN model repository as follows: https://github.com/Helomin/STMixGAN

# Reference
@article{LukaDoncic0/GAN-argcPredNet,
title={GAN–argcPredNet v1.0: a generative adversarial model for radar echo extrapolation based on convolutional recurrent units
author={Kun Zheng, Yan Liu, Jinbiao Zhang, Cong Luo, Siyu Tang, Huihua Ruan, Qiya Tan, Yunlei Yi, and Xiutao Ran},
journal={GMD, 15, 1467–1475, 2022},
year={2022}

AND

@article{QiyaTan/GAN-argcPredNet-v2.0,
title={GAN-argcPredNet v2.0: a radar echo extrapolation model based on spatiotemporal process enhancement
author={Kun Zheng, Qiya Tan, Huihua Ruan, Jinbiao Zhang, Cong Luo, Siyu Tang, Yunlei Yi, Yugang Tian, and Jianmei Cheng},
journal={GMD, 17, 399–413, 2024},
year={2024}
