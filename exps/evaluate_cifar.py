import argparse
import logging
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torch
import torchvision.transforms as transforms

from exps.preact_resnet import PreActResNet18

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_fgsm(model, X, y, epsilon):
    delta = torch.zeros_like(X, requires_grad=True)
    output = model(X + delta)
    loss = F.cross_entropy(output, y)
    loss.backward()
    grad = delta.grad.detach()
    delta.data = epsilon * torch.sign(grad)
    return delta.detach()


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        delta[:, 0, :, :].uniform_(-epsilon[0][0][0].item(), epsilon[0][0][0].item())
        delta[:, 1, :, :].uniform_(-epsilon[1][0][0].item(), epsilon[1][0][0].item())
        delta[:, 2, :, :].uniform_(-epsilon[2][0][0].item(), epsilon[2][0][0].item())
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            x = X[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--fname', type=str)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'ddn', 'none'])
    parser.add_argument('--epsilon', default=8, type=float)
    parser.add_argument('--attack-iters', default=50, type=int)
    parser.add_argument('--alpha', default=2, type=int)
    parser.add_argument('--restarts', default=10, type=int)
    parser.add_argument('--seed', default=0, type=int)
    return parser.parse_args()


def main():
    args = get_args()
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # dataset = cifar10(args.data_dir)
    # test_set = list(zip(transpose(normalise(dataset['test']['data'])), dataset['test']['labels']))
    # batches = Batches(test_set, args.batch_size, shuffle=False, num_workers=2)

    transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = torchvision.datasets.CIFAR10(root='./../../cifar-data/', train=True,
                                            download=True, transform=transform)
    batches = Batches(trainset, args.batch_size, shuffle=True, set_random_choices=False, num_workers=2)

    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    model = PreActResNet18().cuda()
    checkpoint = torch.load(args.fname)
    model.load_state_dict(checkpoint)
    model.eval()
    model.float()

    total_loss = 0
    total_acc = 0
    n = 0

    if args.attack == 'none':
        with torch.no_grad():
            for batch in batches:
                X, y = batch['input'], batch['target']
                output = model(X)
                loss = F.cross_entropy(output, y)
                total_loss += loss.item() * y.size(0)
                total_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
    else:
        for batch in batches:
            X, y = batch['input'], batch['target']
            if args.attack == 'pgd':
                delta = attack_pgd(model, X, y, epsilon, alpha, args.attack_iters, args.restarts)
            elif args.attack == 'fgsm':
                delta = attack_fgsm(model, X, y, epsilon)
            with torch.no_grad():
                output = model(X + delta)
                loss = F.cross_entropy(output, y)
                total_loss += loss.item() * y.size(0)
                total_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)

    logger.info('Test Loss: %.4f, Acc: %.4f', total_loss/n, total_acc/n)


if __name__ == "__main__":
    main()