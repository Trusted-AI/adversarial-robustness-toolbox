import argparse
import copy
import logging
import math
import random
import sys
import time

sys.path.append('/home/ambrish/github/adversarial-robustness-toolbox')
from art.classifiers import PyTorchClassifier
from art.defences.trainer import AdversarialTrainerFBF
from art.utils import load_cifar10
import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
print(upper_limit, lower_limit)
print(upper_limit.size(), lower_limit.size())

import torch





def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        module.weight.data.normal_(0, math.sqrt(2. / n))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', choices=['cyclic', 'piecewise'])
    parser.add_argument('--lr-max', default=0.21, type=float)
    parser.add_argument('--attack', default='fgsm', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=5, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=int)
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
    parser.add_argument('--fname', default='cifar_robust_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--overfit-check', action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    start_start_time = time.time()

    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
    x_train[:, :, :, 0] -= cifar10_mean[0]
    x_train[:, :, :, 1] -= cifar10_mean[1]
    x_train[:, :, :, 2] -= cifar10_mean[2]

    x_train[:, :, :, 0] /= cifar10_std[0]
    x_train[:, :, :, 1] /= cifar10_std[1]
    x_train[:, :, :, 2] /= cifar10_std[2]

    x_test[:, :, :, 0] -= cifar10_mean[0]
    x_test[:, :, :, 1] -= cifar10_mean[1]
    x_test[:, :, :, 2] -= cifar10_mean[2]

    x_test[:, :, :, 0] /= cifar10_std[0]
    x_test[:, :, :, 1] /= cifar10_std[1]
    x_test[:, :, :, 2] /= cifar10_std[2]

    x_train = x_train.transpose(0, 3, 1, 2)
    x_test = x_test.transpose(0, 3, 1, 2)

    model = PreActResNet18().cuda()
    model.apply(initialize_weights)
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=5e-4)

    model, opt = amp.initialize(model, opt, opt_level="O2", loss_scale=1.0, master_weights=False)

    criterion = nn.CrossEntropyLoss()

    # Step 3: Create the ART classifier

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(-1.9887, 2.1158),
        loss=criterion,
        optimizer=opt,
        input_shape=(32,32,3),
        nb_classes=10,
    )
    epsilon = (args.epsilon / 255.) / 0.24
    #pgd_alpha = (args.pgd_alpha / 255.) / std

    trainer = AdversarialTrainerFBF(classifier, eps=epsilon)
    trainer.fit(x_train, y_train)

    best_state_dict = copy.deepcopy(model.state_dict())

    train_time = time.time()
    torch.save(best_state_dict, args.fname + '.pth')
    logger.info('Total time: %.4f', train_time - start_start_time)


if __name__ == "__main__":
    main()
