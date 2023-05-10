import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import checkpoint_filter_fn
from functools import partial
from art.estimators.certification.smoothed_vision_transformers import PyTorchSmoothedViT, ArtViT
import copy
import numpy as np
from torchvision import datasets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cifar_data():
    """
    Get CIFAR-10 data.
    :return: cifar train/test data.
    """
    train_set = datasets.CIFAR10('./data', train=True, download=True)
    test_set = datasets.CIFAR10('./data', train=False, download=True)

    x_train = train_set.data.astype(np.float32)
    y_train = np.asarray(train_set.targets)

    x_test = test_set.data.astype(np.float32)
    y_test = np.asarray(test_set.targets)

    x_train = np.moveaxis(x_train, [3], [1])
    x_test = np.moveaxis(x_test, [3], [1])

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = get_cifar_data()
x_test = torch.from_numpy(x_test)

art_model = PyTorchSmoothedViT(model='vit_small_patch16_224',
                               loss=torch.nn.CrossEntropyLoss(),
                               input_shape=(3, 224, 224),
                               nb_classes=10,
                               ablation_type='column',
                               ablation_size=4,
                               threshold=0.01,
                               logits=True)

scheduler = torch.optim.lr_scheduler.MultiStepLR(art_model.optimizer, milestones=[10, 20], gamma=0.1)
art_model.fit(x_train, y_train, update_batchnorm=True, scheduler=scheduler)
art_model.certify(x_train, y_train)
