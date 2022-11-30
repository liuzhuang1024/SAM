import os
import time
import argparse
import random
import torch
import numpy as np
from tensorboardX import SummaryWriter

from utils import load_config, save_checkpoint, load_checkpoint
from dataset import get_crohme_dataset
from models.backbone import Model
from training import train, eval

parser = argparse.ArgumentParser(description='model training')
parser.add_argument('--dataset', default='CROHME', type=str, help='数据集名称')
parser.add_argument('--config', default='config.yaml', type=str, help='数据集名称')
parser.add_argument('--check', action='store_true', help='测试代码选项')
parser.add_argument('--val', action='store_true', help='测试代码选项')
args = parser.parse_args()

if not args.dataset:
    print('请提供数据集名称')
    exit(-1)

"""加载config文件"""
params = load_config(args.config)

"""设置随机种子"""
random.seed(params['seed'])
np.random.seed(params['seed'])
torch.manual_seed(params['seed'])
torch.cuda.manual_seed(params['seed'])

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params['device'] = device

train_loader, eval_loader_14, eval_loader_16, eval_loader_19 = get_crohme_dataset(params)

model = Model(params)
now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
model.name = f'{params["experiment"]}_{now}_decoder-{params["decoder"]["net"]}'

print(model.name)
model = model.to(device)

if args.check:
    writer = None
else:
    writer = SummaryWriter(f'{params["log_dir"]}/{model.name}')

optimizer = getattr(torch.optim, params['optimizer'])(model.parameters(), lr=float(params['lr']),
                                                      eps=float(params['eps']), weight_decay=float(params['weight_decay']))

if params['finetune']:
    print('加载预训练模型权重')
    print(f'预训练权重路径: {params["checkpoint"]}')
    load_checkpoint(model, optimizer, params['checkpoint'])

if not args.check:
    if not os.path.exists(os.path.join(params['checkpoint_dir'], model.name)):
        os.makedirs(os.path.join(params['checkpoint_dir'], model.name), exist_ok=True)
    os.system(f'cp {args.config} {os.path.join(params["checkpoint_dir"], model.name, model.name)}.yaml')
    os.system(f'cp {params["word_path"]} {os.path.join(params["checkpoint_dir"], model.name)}')


min_score = 0
min_step = 0
rate_2014, rate_2016, rate_2019 = 0.55, 0.54, 0.55
rate_2014, rate_2016, rate_2019 = 0.50, 0.50, 0.50
# init_epoch = 0 if not params['finetune'] else int(params['checkpoint'].split('_')[-1].split('.')[0])

if args.val:
    epoch = 1
    model.load_state_dict(torch.load("checkpoints/v1_l1-loss_2022-11-28-23-38_decoder-Decoder_v1/2016_v1_l1-loss_2022-11-28-23-38_decoder-Decoder_v1_WordRate-0.9057_ExpRate-0.5405_182.pth", map_location="cpu")['model'])
    print()
    eval_loss, eval_word_score, eval_expRate = eval(params, model, epoch, eval_loader_14)
    print(f'2014 Epoch: {epoch + 1}  loss: {eval_loss:.4f}  word score: {eval_word_score:.4f}  ExpRate: {eval_expRate:.4f}')
    if eval_expRate >= rate_2014 and not args.check:
        rate_2014 = eval_expRate
        save_checkpoint(model, optimizer, eval_word_score, eval_expRate, epoch + 1,
                        optimizer_save=params['optimizer_save'], path=params['checkpoint_dir'], tag='2014_')
    """CROHME2016 eval"""
    eval_loss, eval_word_score, eval_expRate = eval(params, model, epoch, eval_loader_16)
    print(f'2016 Epoch: {epoch + 1}  loss: {eval_loss:.4f}  word score: {eval_word_score:.4f}  ExpRate: {eval_expRate:.4f}')
    if eval_expRate >= rate_2016 and not args.check:
        rate_2016 = eval_expRate
        save_checkpoint(model, optimizer, eval_word_score, eval_expRate, epoch + 1, optimizer_save=params['optimizer_save'], path=params['checkpoint_dir'], tag='2016_')


    """CROHME2019 eval"""
    eval_loss, eval_word_score, eval_expRate = eval(params, model, epoch, eval_loader_19)
    print(f'2019 Epoch: {epoch + 1}  loss: {eval_loss:.4f}  word score: {eval_word_score:.4f}  ExpRate: {eval_expRate:.4f}')
    if eval_expRate >= rate_2019 and not args.check:
        rate_2019 = eval_expRate
        save_checkpoint(model, optimizer, eval_word_score, eval_expRate, epoch + 1, optimizer_save=params['optimizer_save'], path=params['checkpoint_dir'], tag='2019_')
    exit()


init_epoch = 0
for epoch in range(init_epoch, params['epochs']):

    train_loss, train_word_score, train_exprate = train(params, model, optimizer, epoch, train_loader, writer)

    if epoch >= params['valid_start']:
        for best_rate, time_tag, loader in zip([rate_2014, rate_2016, rate_2019], [2014, 2016, 2019], [eval_loader_14, eval_loader_16, eval_loader_19]):
            eval_loss, eval_word_score, eval_expRate = eval(params, model, epoch, loader)
            if writer:
                writer.add_scalars("eval_ratio", {
                    "eval_loss": eval_loss,
                    "eval_word_score": eval_word_score,
                    "eval_expRate": eval_expRate
                    }, global_step=epoch)
            print(f'{time_tag} Epoch: {epoch + 1}  loss: {eval_loss:.4f}  word score: {eval_word_score:.4f}  ExpRate: {eval_expRate:.4f}')
            if eval_expRate >= best_rate and not args.check:
                best_rate = eval_expRate
                save_checkpoint(model, optimizer, eval_word_score, eval_expRate, epoch + 1,
                                optimizer_save=params['optimizer_save'], path=params['checkpoint_dir'], tag=f'{time_tag}_')
