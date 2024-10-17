import os
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
from models.WGAN import Critic, Generator
# from models.WGAN_GPT import Generator, Discriminator

from tqdm import tqdm
import math
import random

import argparse

import logging
import wandb

logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    datefmt='%Y-%m-%d %H:%M:%S',  # 日期时间格式
    handlers=[
        logging.StreamHandler(),  # 将日志输出到控制台
        logging.FileHandler('app.log')  # 将日志输出到文件
    ]
)

logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-r', '--run_name', type=str, default="run0",
                        help="run name for tensorboard")
    parser.add_argument('-e', '--epoch', type=int, default=10,
                        help="total training epochs")
    parser.add_argument('-k', type=int, default=5,
                        help="critic update frequency")
    parser.add_argument('-s', '--train_steps', type=int, default=1000,
                        help="total training steps")
    parser.add_argument('-t', '--test_steps', type=int, default=40,
                        help="steps to test")
    parser.add_argument('-v', '--save_steps', type=int, default=100,
                        help="steps to save")
    parser.add_argument('-l', '--log_steps', type=int, default=20,
                        help="steps to log")
    parser.add_argument('--log_image_steps', type=int, default=200,
                        help="steps to log image")
    parser.add_argument('--save_dir', type=str, default='./save')
    parser.add_argument('--ckpt', type=str, default='./save')
    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gp', action='store_true', default=False)
    args = parser.parse_args()
    return args

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

import torch.autograd as autograd

def gradient_penalty(critic, real_samples, fake_samples, device):
    alpha = torch.randn(real_samples.size(0), 1, 1, 1, device=device).expand_as(real_samples)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    d_interpolates = critic(interpolates)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size(), device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


def train_gan(critic, generator, noise_sample, **train_args):
    train_loader = train_args['train_loader']
    optimizerD = train_args['optimizerD']
    optimizerG = train_args['optimizerG']
    k = train_args['k']
    epoch = train_args['epoch']
    train_steps = train_args['train_steps']
    log_step = train_args['log_steps']
    log_image_steps = train_args['log_image_steps']
    test_step = train_args['test_steps']
    save_step = train_args['save_steps']
    save_path = train_args['save_path']
    device = train_args['device']
    gp = train_args['gp']
    lambda_gp = train_args.get('lambda_gp', 10)

    os.makedirs(save_path, exist_ok=True)

    critic.train()
    generator.train()
    global_step = 0
    train_loss_D_list = []
    train_loss_G_list = []
    D_x_list = []
    D_G_z_list = []
    for epoch_idx in tqdm(range(epoch), desc='train_epoch'):
        for img, _ in tqdm(train_loader, desc='train_batch'):
            for _ in range(k):
                # print(img)
                global_step += 1
                if global_step > train_steps:
                    return

                bs = img.shape[0]

                # train critic, using Wasserstein distance
                optimizerD.zero_grad()
                real = img.to(device)
                # noise = torch.randn(bs, noise_sample.shape[1], 1, 1, device=device)
                noise = torch.randn(bs, noise_sample.shape[1], device=device)
                fake = generator(noise).detach()

                output_real = critic(real)
                output_fake = critic(fake)
                loss_D = torch.mean(output_fake) - torch.mean(output_real)
                if gp:
                    gp_value = gradient_penalty(critic, real, fake, device)
                    loss_D += lambda_gp * gp_value 
                loss_D.backward()
                train_loss_D_list.append(loss_D.item())
                optimizerD.step()

                D_x_list.append(output_real.mean().item())
                D_G_z_list.append(output_fake.mean().item())


                if not gp:
                    # weight clipping
                    for p in critic.parameters():
                        p.data.clamp_(-0.01, 0.01)

            optimizerG.zero_grad()
            # noise = torch.randn(bs, noise_sample.shape[1], 1, 1, device=device)
            noise = torch.randn(bs, noise_sample.shape[1], device=device)
            fake = generator(noise)
            output_fake = critic(fake)
            loss_G = -torch.mean(output_fake)
            loss_G.backward()
            train_loss_G_list.append(loss_G.item())
            optimizerG.step()

            if global_step % log_step == 0:
                train_loss_D = np.mean(np.array(train_loss_D_list))
                train_loss_G = np.mean(np.array(train_loss_G_list))
                mean_D_x = np.mean(np.array(D_x_list))
                mean_D_G_z = np.mean(np.array(D_G_z_list))
                train_loss_D_list.clear()
                train_loss_G_list.clear()
                D_x_list.clear()
                D_G_z_list.clear()
 
                wandb.log({"train_loss_D": train_loss_D}, step=global_step)
                wandb.log({"train_loss_G": train_loss_G}, step=global_step)
                wandb.log({"D(x)": mean_D_x}, step=global_step)
                wandb.log({"D(G(z))": mean_D_G_z}, step=global_step)
                logger.info(f"train_loss_D={loss_D.item()}, train_loss_G={loss_G.item()}, D(x)={mean_D_x}, D(G(z))={mean_D_G_z}")

            if global_step % test_step == 0:
                pass

            if global_step % log_image_steps == 0:
                generator.eval()
                with torch.no_grad():
                    num_images = 8
                    noise = noise_sample[:num_images]   # fix the noise to get a more start-forward comparison
                    generated_images = generator(noise)

                    generated_images = torch.clamp((generated_images*0.5 + 0.5), 0, 1)
                
                    wandb.log({"Generation": wandb.Image(generated_images, caption="Generated Images")}, step=global_step)
                generator.train()

            if global_step % save_step == 0:
                torch.save(generator.state_dict(), os.path.join(save_path, f'generator_{global_step}.pth'))
                torch.save(critic.state_dict(), os.path.join(save_path, f'discriminator_{global_step}.pth'))


def main():
    args = parse_arg()
    set_seed(args.seed)

    bs = args.batch_size

    # load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    fashion_mnist = datasets.FashionMNIST('./data', download=True, train=True, transform=transform)
    val_ratio = 0.01
    val_size = int(val_ratio * len(fashion_mnist))
    train_size = len(fashion_mnist) - val_size

    train_set, val_set = random_split(fashion_mnist, [train_size, val_size])
    

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    img_shape = (1, 28, 28)
    z_dim = 100  
    fearure_dim = 64
    dim_factor = [1, 2, 4]
    critic = Critic(img_shape, fearure_dim, dim_factor).to(device)
    generator = Generator(z_dim, img_shape, fearure_dim, dim_factor + [8]).to(device)
    # critic = Discriminator().to(device)
    # generator = Generator(z_dim).to(device)
    critic.apply(weights_init_normal)
    generator.apply(weights_init_normal)


    wandb.login(relogin=True)
    wandb.init(project="dl-hw3-part2", name=args.run_name, config=args)

    os.makedirs(args.save_dir, exist_ok=True)

    # fixed_noise = torch.randn(bs, z_dim, 1, 1, device=device)
    fixed_noise = torch.randn(bs, z_dim, device=device)

    lr = args.lr
    if args.gp:
        optimizerD = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))
        optimizerG = torch.optim.Adam(generator.parameters(), lr=lr*1.5, betas=(0.5, 0.9))
    else:
        optimizerD = optim.RMSprop(critic.parameters(), lr=lr)
        optimizerG = optim.RMSprop(generator.parameters(), lr=lr)

    train_args = {
        'train_loader': train_loader,
        'test_loader': val_loader,
        'optimizerD': optimizerD,
        'optimizerG': optimizerG,
        'epoch': args.epoch,
        'train_steps': args.train_steps,
        'k': args.k,
        'log_steps': args.log_steps,
        'log_image_steps': args.log_image_steps,
        'test_steps': args.test_steps,
        'save_steps': args.save_steps,
        'save_path': os.path.join(args.save_dir, args.run_name),
        'device': device,
        'gp': args.gp
    }
    
    train_gan(critic, generator, fixed_noise, **train_args)


    wandb.finish()


if __name__ == '__main__':
    main()