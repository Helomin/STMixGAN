import os
import torch
import torch.nn as nn
from tqdm import tqdm
from loss import LogCoshLoss, gradient_penalty_div
from argsparser import create_parser
from make_data import create_rainloader
from model.STMixGAN import STMixNet, DCNet


class Trainer:
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self._acquire_device()
        self._preparation()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        else:
            print('Use CPU')

    def _preparation(self):
        self._get_data()
        self._get_epoch()
        self._get_traintools()
        self._build_model()
        self._select_optimizer()
        self._select_loss()

    def _build_model(self):
        self.gen = STMixNet(in_channels=5, out_channels=1)

        self.critic = DCNet()

        if torch.cuda.device_count() > 1:
            self.gen = nn.DataParallel(self.gen, device_ids=[0, 1])
            self.critic = nn.DataParallel(self.critic, device_ids=[0, 1])

        self.gen.cuda()
        self.critic.cuda()

    def _get_data(self):
        self.train_loader = create_rainloader(
            self.args.train_file, self.args.batch_size, self.args.num_workers, shuffle=True)

    def _get_epoch(self):
        self.total_epochs = 100
        self.train_len = len(self.train_loader)

    def _select_optimizer(self):
        self.optimizer_gan = torch.optim.Adam(
            self.gen.parameters(), lr=self.args.lr, betas=self.args.lr_betas, eps=1e-7)
        self.optimizer_critic = torch.optim.Adam(
            self.critic.parameters(), lr=self.args.lr, betas=self.args.lr_betas, eps=1e-7)

    def _select_loss(self):
        self.regloss = LogCoshLoss()

    def train(self):
        self.gen.train()
        self.critic.train()

        gen_loss = torch.zeros(1)
        critic_loss = torch.zeros(1)
        for epoch in range(self.total_epochs):
            pbar = tqdm(total=len(self.train_loader), desc=f'Epoch {epoch + 1}/{self.total_epochs}', postfix=dict,
                        mininterval=0.3)  # Setting the current epoch display progress

            for (itr, train_imgs) in enumerate(self.train_loader):
                train_imgs = train_imgs.type(torch.cuda.FloatTensor)
                train_input = train_imgs[:, :5]
                train_target = train_imgs[:, 5:]

                # train D
                if (itr + 1) % 2 == 0:
                    train_fake = self.gen(train_input).detach()

                    self.optimizer_critic.zero_grad()

                    input_real = torch.cat([train_input, train_target], dim=1)
                    input_fake = torch.cat([train_input, train_fake], dim=1)
                    critic_real = self.critic(input_real)
                    critic_fake = self.critic(input_fake)

                    gp = gradient_penalty_div(
                        self.critic, input_real, input_fake)
                    critic_loss = - \
                        torch.mean(critic_real) + torch.mean(critic_fake) + gp
                    critic_loss.backward()
                    self.optimizer_critic.step()

                # train G
                self.optimizer_gan.zero_grad()
                train_gen = self.gen(train_input)
                input_seq = torch.cat([train_input, train_gen], dim=1)
                gen_fake = self.critic(input_seq)
                gen_loss = -torch.mean(gen_fake) + \
                    self.regloss(train_gen, train_target)
                gen_loss.backward()
                self.optimizer_gan.step()

                pbar.set_postfix(
                    **{'gen_loss': gen_loss.item(), 'critic_loss': critic_loss.item()})
                pbar.update(1)

            pbar.close()  # Turn off the current epoch display progress

            if (epoch+1) % 5 == 0:
                torch.save(self.gen.state_dict(),
                           f"{self.args.G_weight_dir}stmixnet_{epoch+1}.pth")
                torch.save(self.critic.state_dict(),
                           f"{self.args.D_weight_dir}dcnet_{epoch+1}.pth")


if __name__ == '__main__':
    args = create_parser().parse_args()

    # train
    trainer = Trainer(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>Start of training<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    trainer.train()
