import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
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


def cnn_dev():
    class CIFARModel(torch.nn.Module):

        def __init__(self, number_of_classes: int):
            super(CIFARModel, self).__init__()

            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            self.conv_1 = torch.nn.Conv2d(in_channels=6,
                                          out_channels=32,
                                          kernel_size=4,
                                          stride=2)

            self.conv_2 = torch.nn.Conv2d(in_channels=32,
                                          out_channels=32,
                                          kernel_size=4,
                                          stride=1)

            self.fc1 = torch.nn.Linear(in_features=4608, out_features=number_of_classes)

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
            return self.fc1(x)

    model = CIFARModel(number_of_classes=10)
    (x_train, y_train), (x_test, y_test) = get_cifar_data()

    art_model = PyTorchDeRandomizedSmoothing(model=model,
                                             loss=torch.nn.CrossEntropyLoss(),
                                             optimizer=torch.optim.SGD(model.parameters(), lr=0.001),
                                             input_shape=(3, 32, 32),
                                             nb_classes=10,
                                             ablation_type='column',
                                             ablation_size=4,
                                             threshold=0.1,
                                             logits=False)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(art_model.optimizer, milestones=[10, 20], gamma=0.1)
    art_model.predict(x_train[0:10])
    print(art_model)

    art_model.fit(x_train, y_train,
                  nb_epochs=30,
                  update_batchnorm=True,
                  scheduler=scheduler)

vit_dev()
# cnn_dev()