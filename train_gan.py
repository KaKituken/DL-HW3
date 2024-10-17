import os
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
from models.DCGAN import Discriminator, Generator

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
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    return args

def train_gan(discriminator, generator, noise_sample, **train_args):
    train_loader = train_args['train_loader']
    optimizerD = train_args['optimizerD']
    optimizerG = train_args['optimizerG']
    criterion = train_args['criterion']
    epoch = train_args['epoch']
    train_steps = train_args['train_steps']
    log_step = train_args['log_steps']
    log_image_steps = train_args['log_image_steps']
    test_step = train_args['test_steps']
    save_step = train_args['save_steps']
    save_path = train_args['save_path']
    device = train_args['device']

    os.makedirs(save_path, exist_ok=True)

    discriminator.train()
    generator.train()
    global_step = 0
    train_loss_D_list = []
    train_loss_G_list = []
    D_x_list = []
    D_G_z_list = []
    for epoch_idx in tqdm(range(epoch), desc='train_epoch'):
        for img, _ in tqdm(train_loader, desc='train_batch'):
            k = 1
            for _ in range(k):
                # print(img)
                global_step += 1
                if global_step > train_steps:
                    return

                optimizerD.zero_grad()
                real = img.to(device)
                bs = real.shape[0]
                label_real = torch.full((bs,), 1, dtype=torch.float, device=device)
                output = discriminator(real)[:, 0, 0, 0]
                loss_D_real = criterion(output, label_real)
                D_x = output.mean().item()
                D_x_list.append(D_x)

                noise = torch.randn(bs, noise_sample.shape[1], 1, 1, device=device)
                fake = generator(noise)
                label_fake = torch.full((bs,), 0, dtype=torch.float, device=device)
                output = discriminator(fake.detach())[:, 0, 0, 0]
                loss_D_fake = criterion(output, label_fake)
                D_G_z1 = output.mean().item()
                loss_D = loss_D_real + loss_D_fake
                loss_D.backward()
                train_loss_D_list.append(loss_D.item())
                optimizerD.step()

            optimizerG.zero_grad()
            noise = torch.randn(bs, noise_sample.shape[1], 1, 1, device=device)
            fake = generator(noise)
            label_gen = torch.full((bs,), 1, dtype=torch.float, device=device)
            output = discriminator(fake)[:, 0, 0, 0]
            loss_G = criterion(output, label_gen)
            loss_G.backward()
            train_loss_G_list.append(loss_G.item())
            D_G_z2 = output.mean().item()
            optimizerG.step()
            D_G_z_list.append(D_G_z1/D_G_z2)

            if global_step % log_step == 0:
                train_loss_D = np.mean(np.array(train_loss_D_list))
                train_loss_G = np.mean(np.array(train_loss_G_list))
                mean_D_x = np.mean(np.array(D_x_list))
                mean_D_G_z = np.mean(np.array(D_G_z_list))
                train_loss_D_list.clear()
                train_loss_G_list.clear()
                D_x_list.clear()
                D_G_z_list.clear()
                # 使用 wandb 记录 loss
                wandb.log({"train_loss_D": train_loss_D}, step=global_step)
                wandb.log({"train_loss_G": train_loss_G}, step=global_step)
                wandb.log({"D(x)": mean_D_x}, step=global_step)
                wandb.log({"D(G(z))": mean_D_G_z}, step=global_step)
                logger.info(f"train_loss_D={loss_D.item()}, train_loss_G={loss_G.item()}, D(x)={D_x}, D(G(z))={D_G_z1 / D_G_z2}")

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
                torch.save(discriminator.state_dict(), os.path.join(save_path, f'discriminator_{global_step}.pth'))


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


    # first stage
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    img_shape = (1, 28, 28)
    z_dim = 100  
    fearure_dim = 64
    dim_factor = [1, 2, 4]
    discriminator = Discriminator(img_shape, fearure_dim, dim_factor).to(device)
    generator = Generator(z_dim, img_shape, fearure_dim, dim_factor).to(device)


    wandb.login(relogin=True)
    wandb.init(project="dl-hw3-part2", name=args.run_name, config=args)

    os.makedirs(args.save_dir, exist_ok=True)


    criterion = nn.BCELoss()

    fixed_noise = torch.randn(bs, z_dim, 1, 1, device=device)

    lr = args.lr
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=lr * 1.5, betas=(0.5, 0.999))

    train_args = {
        'train_loader': train_loader,
        'test_loader': val_loader,
        'optimizerD': optimizerD,
        'optimizerG': optimizerG,
        'criterion': criterion,
        'epoch': args.epoch,
        'train_steps': args.train_steps,
        'log_steps': args.log_steps,
        'log_image_steps': args.log_image_steps,
        'test_steps': args.test_steps,
        'save_steps': args.save_steps,
        'save_path': os.path.join(args.save_dir, args.run_name),
        'device': device
    }
    
    train_gan(discriminator, generator, fixed_noise, **train_args)

    wandb.finish()


if __name__ == '__main__':
    main()