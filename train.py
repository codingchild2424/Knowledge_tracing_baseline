import argparse

import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

#다른 .py파일 가져오기
from models.dkt import Dkt
from trainer import Trainer
from utils import Utils
from data_loaders.assist2015_loader import Assist2015_loader

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.6)
    p.add_argument('--valid_ratio', type=float, default=.2)
    p.add_argument('--test_ratio', type=float, default=.2)

    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--dataset', type=str, default = 'assist2015')

    config = p.parse_args()

    return config

def main(config):
    #device 선언
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    #data 호출, default는 assist2015
    if config.dataset == 'assist2015':
        dataset = Assist2015_loader()
    #--> 다른 데이터셋이 추가되면 여기에 설정

    #utils 경로 선언, 경로는 dataset.data_path를 활용하여 받아오기
    utils = Utils(dataset.data_path, device)

    #trainset과 testset 나누기
    train_size = int( len(dataset) * config.train_ratio )
    test_size = len(dataset) - train_size

    train_data, test_data = random_split(
        dataset, [train_size, test_size]
    )

    #hyperparameters
    input_size = len(dataset[0][0])
    hidden_size = 50

    #model 선언
    model = Dkt(input_size = input_size, hidden_size = hidden_size)
    #model의 구조 보여주기
    print(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    #crit을 통해 utils.py에 있는 loss_function()을 받아옴
    crit = utils.loss_function

    #device를 하나 더 받도록 만듬
    trainer = Trainer(model, optimizer, crit, device, dataset.data_path)

    trainer.train(train_data, test_data, config)

    #Save best model weights.
    torch.save({
        'model': trainer.model.state_dict(),
        'config': config
    }, config.model_fn)

#실행
if __name__ == '__main__':
    config = define_argparser()
    main(config)