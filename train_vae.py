import os
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
from data import init_semi_dataset, Labeled_MNIST, Unlabeled_MNIST
from model import VAE, build_svm, SVMPipeline

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
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_batch_norm', action="store_true")
    parser.add_argument('--use_drop_out', action="store_true")
    parser.add_argument('--drop_out_prob', type=float, default=0.5)
    parser.add_argument('--test_curve', action="store_true")
    parser.add_argument('--stage', type=int, default=1)
    args = parser.parse_args()
    return args

def train_vae(model, **train_args):
    train_loader = train_args['train_loader']
    test_loader = train_args['test_loader']
    optimizer = train_args['optimizer']
    epoch = train_args['epoch']
    train_steps = train_args['train_steps']
    log_step = train_args['log_steps']
    log_image_steps = train_args['log_image_steps']
    test_step = train_args['test_steps']
    save_step = train_args['save_steps']
    save_path = train_args['save_path']
    device = train_args['device']

    os.makedirs(save_path, exist_ok=True)

    model.train()
    global_step = 0
    training_loss_list = []
    for epoch_idx in tqdm(range(epoch), desc='train_epoch'):
        for img in tqdm(train_loader, desc='train_batch'):
            global_step += 1
            if global_step > train_steps:
                return
            
            img = img.to(device)

            optimizer.zero_grad()
            loss = model(img)
            loss.backward()
            optimizer.step()
            training_loss_list.append(loss.item())

            if global_step % log_step == 0:
                training_loss = np.mean(np.array(training_loss_list))
                training_loss_list.clear()
                # 使用 wandb 记录 loss
                wandb.log({"train_loss": training_loss}, step=global_step)
                logger.info(f"{training_loss=}")

            if global_step % test_step == 0:
                model.eval()
                with torch.no_grad():
                    test_loss = 0
                    for img in tqdm(test_loader, desc='test_batch'):
                        img = img.to(device)
                        loss = model(img)
                        test_loss += loss.item()
                    test_loss /= len(test_loader)
                    wandb.log({"test_loss": test_loss}, step=global_step)
                    logger.info(f"{test_loss=}")
                    model.train()

            if global_step % log_image_steps == 0:
                model.eval()
                with torch.no_grad():
                    num_images = min(8, img.shape[0])
                    input_images = img[:num_images].unsqueeze(1)
                    z, _, _ = model.encode(input_images.view(num_images, -1))
                    recon_images = model.decode(z)
                    recon_images = recon_images.view(num_images, 1, 28, 28)

                    input_images = torch.clamp((input_images*0.5 + 0.5), 0, 1)
                    recon_images = torch.clamp((recon_images*0.5 + 0.5), 0, 1)
                
                    wandb.log({"Reconstruction": [wandb.Image(input_images, caption="input"),
                                                  wandb.Image(recon_images, caption="reconstructed")],
                                }, step=global_step)

            if global_step % save_step == 0:
                torch.save(model.state_dict(), os.path.join(save_path, f'model_{global_step}.pth'))


def main():
    args = parse_arg()
    set_seed(args.seed)

    # load dataset
    bs = args.batch_size
    labeled_samples = [100, 600, 1000, 3000]
    unlabeled_train, labeled_train_list, unlabeled_val, labeled_val = init_semi_dataset('./data', labeled_samples)
    

    vae_train_loader = DataLoader(unlabeled_train, batch_size=bs, shuffle=True)
    vae_val_loader = DataLoader(unlabeled_val, batch_size=bs, shuffle=False)


    # first stage
    print(unlabeled_train[0].shape)
    input_shape = unlabeled_train[0].shape
    input_size = input_shape[0] * input_shape[1]
    z_dim = 50          # 50
    hidden_dim = 600    # 600
    vae = VAE(input_size, z_dim, hidden_dim)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    if args.stage == 1:
        wandb.login(relogin=True)
        wandb.init(project="dl-hw3", name=args.run_name, config=args)

        os.makedirs(args.save_dir, exist_ok=True)

        vae = vae.to(device)

        optimizer = optim.Adam(
            vae.parameters(), 
            lr=args.lr,
            # alpha=0.001,
            # momentum=0.1
        )
        train_args = {
            'train_loader': vae_train_loader,
            'test_loader': vae_val_loader,
            'optimizer': optimizer,
            'epoch': args.epoch,
            'train_steps': args.train_steps,
            'log_steps': args.log_steps,
            'log_image_steps': args.log_image_steps,
            'test_steps': args.test_steps,
            'save_steps': args.save_steps,
            'save_path': os.path.join(args.save_dir, args.run_name),
            'device': device
        }
        
        train_vae(vae, **train_args)
    
    else:
        # second stage
        # load ckpt
        ckpt_path = args.ckpt
        vae.load_state_dict(torch.load(ckpt_path))
        vae = vae.to(device)
        svm = build_svm('rbf', C=1.0)
        svm_pipeline = SVMPipeline(vae, svm, device)
        svm_pipeline.train_and_eval_list(labeled_train_list, labeled_val)

    # 训练完成后关闭 wandb
    wandb.finish()


if __name__ == '__main__':
    main()