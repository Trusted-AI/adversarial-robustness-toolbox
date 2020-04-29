import argparse
import copy
import logging
import math
import random
import sys
import time
sys.path.append('/home/ambrish/github/adversarial-robustness-toolbox')

from art.classifiers import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent
from art.utils import load_cifar10
import numpy as np
import torch
import torch.nn as nn

from exps.preact_resnet import PreActResNet18

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
    parser.add_argument('--accfname', default='acc', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--overfit-check', action='store_true')
    return parser.parse_args()


def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    (_, _), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
    # cifar_mu, cifar_std = 0.4733630004850874, 0.25156892506322026
    # cifar_mu = (0.4914, 0.4822, 0.4465)
    # cifar_std = (0.2471, 0.2435, 0.2616)

    cifar_mu = np.ones((3,32,32))
    cifar_mu[0, :, :] = 0.4914
    cifar_mu[1, :, :] = 0.4822
    cifar_mu[2, :, :] = 0.4465

    cifar_std = np.ones((3, 32, 32))
    cifar_std[0, :, :] = 0.2471
    cifar_std[1, :, :] = 0.2435
    cifar_std[2, :, :] = 0.2616

    x_test = x_test.transpose(0, 3, 1, 2).astype('float32')

    model = PreActResNet18().cuda()

    checkpoint = torch.load(args.fname)
    model.load_state_dict(checkpoint)
    model.eval()
    model.float()

    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Step 3: Create the ART classifier
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0.0,1.0),
        preprocessing=(cifar_mu,cifar_std),
        loss=criterion,
        optimizer=opt,
        input_shape=(3,32,32),
        nb_classes=10,
    )


    #compute accuracy
    output = np.argmax(classifier.predict(x_test), axis=1)
    nb_correct_pred = np.sum(output == np.argmax(y_test, axis=1))
    print("accuracy: {}".format(nb_correct_pred / x_test.shape[0]), flush=True)

    acc = [nb_correct_pred / x_test.shape[0]]
    eps_range = [8/255., 16/255.]

    for eps in eps_range:
        eps_step = (1.5 * eps) / 40
        attack_test = ProjectedGradientDescent(classifier=classifier, norm=np.inf, eps=eps,
                                               eps_step=eps_step, max_iter=40, targeted=False,
                                               num_random_init=10, batch_size=32)
        x_test_attack = attack_test.generate(x_test)
        x_test_attack_pred = np.argmax(classifier.predict(x_test_attack), axis=1)
        nb_correct_attack_pred = np.sum(x_test_attack_pred == np.argmax(y_test, axis=1))
        acc.append(nb_correct_attack_pred / x_test.shape[0])
        print(acc, flush=True)

    np.save('./exps/{}.npy'.format(args.accfname), acc)



if __name__ == "__main__":
    main()
