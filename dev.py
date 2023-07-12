import torch
import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
from art.estimators.certification.derandomized_smoothing import PyTorchDeRandomizedSmoothing
import numpy as np
from torchvision import datasets
from torchvision import transforms

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

def get_mnist_data():
    """
    Get the MNIST data.
    """
    train_set = datasets.MNIST('./data', train=True, download=True)
    test_set = datasets.MNIST('./data', train=False, download=True)

    x_train = train_set.data.numpy().astype(np.float32)
    y_train = train_set.targets.numpy()

    x_test = test_set.data.numpy().astype(np.float32)
    y_test = test_set.targets.numpy()

    x_train = np.expand_dims(x_train, axis=1)
    x_test = np.expand_dims(x_test, axis=1)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return (x_train, y_train), (x_test, y_test)

def vit_dev():
    (x_train, y_train), (x_test, y_test) = get_cifar_data()

    art_model = PyTorchDeRandomizedSmoothing(model='vit_small_patch16_224',
                                             loss=torch.nn.CrossEntropyLoss(),
                                             optimizer=torch.optim.SGD,
                                             optimizer_params={"lr": 0.01},
                                             input_shape=(3, 32, 32),
                                             nb_classes=10,
                                             ablation_size=4,
                                             replace_last_layer=True,
                                             load_pretrained=True)
    # art_model.predict(x_train[0:10])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(art_model.optimizer, milestones=[10, 20], gamma=0.1)
    art_model.fit(x_train, y_train,
                  nb_epochs=30,
                  update_batchnorm=False,
                  scheduler=scheduler,
                  transform=transforms.Compose([transforms.RandomHorizontalFlip()]))

    torch.save(art_model.model.state_dict(), 'trained_refactor.pt')
    art_model.model.load_state_dict(torch.load('trained_refactor.pt'))
    art_model.eval_and_certify(x_test, y_test, size_to_certify=4)


def cnn_dev(algo='salman2021'):

    assert algo in ['levine2020', 'salman2021']

    if algo == 'salman2021':
        class MNISTModel(torch.nn.Module):

            def __init__(self):
                super(MNISTModel, self).__init__()

                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                self.conv_1 = torch.nn.Conv2d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=4,
                                              stride=2,
                                              padding=1)

                self.conv_2 = torch.nn.Conv2d(in_channels=64,
                                              out_channels=128,
                                              kernel_size=4,
                                              stride=2, padding=1)

                self.fc1 = torch.nn.Linear(in_features=128*7*7, out_features=500)
                self.fc2 = torch.nn.Linear(in_features=500, out_features=100)
                self.fc3 = torch.nn.Linear(in_features=100, out_features=10)

                self.relu = torch.nn.ReLU()

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                """
                Computes the forward pass though the neural network
                :param x: input data of shape (batch size, N features)
                :return: model prediction
                """
                x = self.relu(self.conv_1(x))
                x = self.relu(self.conv_2(x))
                x = torch.flatten(x, 1)
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.fc3(x)
                return x
    else:
        class MNISTModel(torch.nn.Module):

            def __init__(self):
                super(MNISTModel, self).__init__()

                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                self.conv_1 = torch.nn.Conv2d(in_channels=2,
                                              out_channels=64,
                                              kernel_size=4,
                                              stride=2,
                                              padding=1)

                self.conv_2 = torch.nn.Conv2d(in_channels=64,
                                              out_channels=128,
                                              kernel_size=4,
                                              stride=2, padding=1)

                self.fc1 = torch.nn.Linear(in_features=128*7*7, out_features=500)
                self.fc2 = torch.nn.Linear(in_features=500, out_features=100)
                self.fc3 = torch.nn.Linear(in_features=100, out_features=10)

                self.relu = torch.nn.ReLU()

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                x = self.relu(self.conv_1(x))
                x = self.relu(self.conv_2(x))
                x = torch.flatten(x, 1)
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.fc3(x)
                return x

    model = MNISTModel()
    # (x_train, y_train), (x_test, y_test) = get_cifar_data()
    (x_train, y_train), (x_test, y_test) = get_mnist_data()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    if algo == 'salman2021':
        art_model = PyTorchDeRandomizedSmoothing(model=model,
                                                 loss=torch.nn.CrossEntropyLoss(),
                                                 optimizer=optimizer,
                                                 input_shape=(1, 28, 28),
                                                 nb_classes=10,
                                                 ablation_type='column',
                                                 ablation_size=2,
                                                 algorithm=algo,
                                                 logits=True)
    else:
        art_model = PyTorchDeRandomizedSmoothing(model=model,
                                                 loss=torch.nn.CrossEntropyLoss(),
                                                 optimizer=optimizer,
                                                 input_shape=(1, 28, 28),
                                                 nb_classes=10,
                                                 ablation_type='column',
                                                 ablation_size=2,
                                                 algorithm=algo,
                                                 threshold=0.3,
                                                 logits=True)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(art_model.optimizer, milestones=[200], gamma=0.1)

    art_model.fit(x_train, y_train,
                  nb_epochs=400,
                  scheduler=scheduler)

vit_dev()
# cnn_dev()