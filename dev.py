
# https://github.com/huggingface/pytorch-image-models
import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer, vit_small_patch16_224
from art.estimators.certification.smoothed_vision_transformers import PyTorchSmoothedViT
import copy
import numpy as np
from torchvision import datasets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = vit_small_patch16_224(pretrained=True)
print(type(model))


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


def update_batchnorm(model, x):
    import random
    from tqdm import tqdm

    art_model.model.train()
    batch_size = 32

    ind = np.arange(len(x))
    num_batch = int(len(x) / float(batch_size))

    print('updating batchnorm')
    with torch.no_grad():
        for _ in tqdm(range(200)):
            for m in tqdm(range(num_batch)):
                i_batch = torch.from_numpy(np.copy(x[ind[m * batch_size: (m + 1) * batch_size]])).to(device)
                i_batch = art_model.ablator.forward(i_batch, column_pos=random.randint(0, x.shape[3]))
                art_model.model(i_batch.cuda())
    return model

(x_train, y_train), (x_test, y_test) = get_cifar_data()
x_test = torch.from_numpy(x_test)
print('params: ', model.parameters())
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

# Use same initial point as Madry
checkpoint = torch.hub.load_state_dict_from_url(
    url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
    map_location="cpu", check_hash=True
)
model.load_state_dict(checkpoint["model"])

art_model = PyTorchSmoothedViT(model=model,
                               loss=torch.nn.CrossEntropyLoss(),
                               input_shape=(3, 224, 224),
                               optimizer=optimizer,
                               nb_classes=10,
                               ablation_type='column',
                               ablation_size=4,
                               threshold=0.01,
                               logits=True)

ablated_x = art_model.ablator.ablate(x=copy.deepcopy(x_test[:32]),
                                     column_pos=1)

print('test position 31')

ablated_x = art_model.ablator.ablate(x=copy.deepcopy(x_test[:32]),
                                     column_pos=31)

art_model = update_batchnorm(art_model, x_train)
art_model.fit(x_train, y_train)
