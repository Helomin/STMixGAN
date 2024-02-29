# STMixGAN
This is a radar echo extrapolation model, which can improve the accuracy of rainfall prediction.
Both the generator STMixNet and the discriminator DCNet are stored in the file ```model/STMixGAN.py```.

The network is trained to be a predictive model that can infer the next ten frames from the first five frames, i.e., the first half hour is used as input to predict the distribution of the next hour.
STMixGAN consists of two components: the generator is a spatiotemporal mixed coding network (STMixNet), and the dual-channel network (DCNet) serves as a discriminator. The STMixNet is based on the whole echo evolution process and extracts global to local multi-scale features by capturing the spatiotemporal correlation of the radar echo context, which is useful to help avoid regression to the mean and blurring effects. DCNet guides the training of STMixNet by recognizing that the predictions are false, and the ground truth is true, to make its predictions more realistic.

# Radar Data
The experimental data is the radar mosaic of the South China area provided by the Guangdong Meteorological Bureau. It does not support open sharing. For data access, please contact Kun Zheng (ZhengK@cug.edu.cn) and Long He (Helomin@cug.edu.cn).

# STMixNet
You can adjust the parameters of your data.

# Train
The files for training the model are stored in the ```Trainer.py``` file. Input your data as a training file and you are ready to start training.
```
self.train_loader = create_rainloader(self.args.train_file, self.args.batch_size, self.args.num_workers, shuffle=True)
```
Save the weight files of the generator and the discriminator respectively:
```
torch.save(self.gen.state_dict(), f"{self.args.G_weight_dir}stmixnet_{epoch+1}.pth")
torch.save(self.critic.state_dict(), f"{self.args.D_weight_dir}dcnet_{epoch+1}.pth")
```

# Prediction
The prediction code is stored in the ```Predictor.py``` file. ```self.test_loader = create_rainloader(self.args.test_file, self.args.batch_size, self.args.num_workers)``` loads the test set file, ```torch.load(f{self.args.G_weight_dir}stmixnet_{self.args.epoch_weight}.pth")``` loads the trained weight file. Then through predict() functions respectively generate prediction data. Finally save the prediction data and metrics can be calculated using ```metrics.py```.

# Loss
loss.py contains the loss function needed for our model.

# Note
You can cite the STMixGAN model repository as follows: https://github.com/Helomin/STMixGAN

# Reference
@article{gmd-15-1467-2022,

title={GAN–argcPredNet v1.0: a generative adversarial model for radar echo extrapolation based on convolutional recurrent units},

author={Kun Zheng, Yan Liu, Jinbiao Zhang, Cong Luo, Siyu Tang, Huihua Ruan, Qiya Tan, Yunlei Yi, and Xiutao Ran},

journal={Geoscientific Model Development},

year={2022}

AND

@article{gmd-17-399-2024,

title={GAN-argcPredNet v2.0: a radar echo extrapolation model based on spatiotemporal process enhancement},

author={Kun Zheng, Qiya Tan, Huihua Ruan, Jinbiao Zhang, Cong Luo, Siyu Tang, Yunlei Yi, Yugang Tian, and Jianmei Cheng},

journal={Geoscientific Model Development},

year={2024}
