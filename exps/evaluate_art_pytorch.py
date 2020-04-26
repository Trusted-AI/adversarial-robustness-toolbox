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
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--overfit-check', action='store_true')
    return parser.parse_args()


def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    start_start_time = time.time()

    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
    cifar_mu, cifar_std  = 0.4733630004850874, 0.25156892506322026

    # upper_limit = ((1.0 - mu) / std)
    # lower_limit = ((0.0 - mu) / std)

    x_train = x_train.transpose(0, 3, 1, 2).astype('float32')
    x_test = x_test.transpose(0, 3, 1, 2).astype('float32')

    model = PreActResNet18().cuda()
    # model.apply(initialize_weights)
    # model.train()

    checkpoint = torch.load(args.fname)
    model.load_state_dict(checkpoint)
    model.eval()
    model.float()

    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    # model, opt = amp.initialize(model, opt, opt_level="O2", loss_scale=1.0, master_weights=False)

    criterion = nn.CrossEntropyLoss()

    # Step 3: Create the ART classifier

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0.0,1.0),
        preprocessing=(cifar_mu,cifar_std),
        loss=criterion,
        optimizer=opt,
        input_shape=(32,32,3),
        nb_classes=10,
    )

    epsilon = (args.epsilon / 255.)

    # trainer = AdversarialTrainerFBF(classifier, eps=epsilon)
    # classifier.fit(x_train, y_train, nb_epochs=3)

    # best_state_dict = copy.deepcopy(model.state_dict())

    #accuracy
    output = np.argmax(classifier.predict(x_test), axis=1)
    print(output)
    nb_correct_pred = np.sum(output == np.argmax(y_test, axis=1))
    print("accuracy: {}".format(nb_correct_pred / x_test.shape[0]))

    # train_time = time.time()
    # torch.save(best_state_dict, args.fname + '.pth')



if __name__ == "__main__":
    main()
