import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import art.estimators.classification
import art.attacks.evasion
import sklearn.datasets
from art.attacks.evasion import ProjectedGradientDescent


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.layer_1 = nn.Linear(20, 32)
        self.layer_2 = nn.Linear(32,1)
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x= torch.sigmoid(self.layer_2(x))

        return x


if __name__ == "__main__":
    device = "cpu"
    x, y = sklearn.datasets.make_classification(n_samples=10000, n_features=20, n_informative=5, n_redundant=2,
                                                n_repeated=0, n_classes=2)

    train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(x,y, test_size=0.2)
    # train_x, test_x, train_y, test_y =  torch.tensor(train_x).type(torch.float32), torch.tensor(test_x).type(torch.float32), torch.tensor(train_y).type(torch.float32), torch.tensor(test_y).type(torch.float32)
    rand_inp = torch.randn((1, 20))
    check_d_type = type(train_x)
    model = BasicModel()
    outs = model(rand_inp)
    loss_func = nn.BCELoss()
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)

    classifier = art.estimators.classification.PyTorchClassifier(
        model=model,
        loss=loss_func,
        optimizer=opt,
        input_shape=(1, 28, 28),
        nb_classes=2,
        # adv_criterion=lambda x, y: x>y
    )
    
    type_of = type(x)
    print(check_d_type)

    classifier.fit(train_x, train_y, batch_size=64, nb_epochs=3)
    test_x_batch = test_x[0:16]
    test_y_batch = test_y[0:16] 
    # test_loss = classifier.compute_loss(test_x_batch.to(device), test_y_batch.to(device))
    test_loss = classifier.compute_loss(test_x_batch, test_y_batch)
    print(test_loss)

    preds = classifier.predict(test_x_batch)
    #acts = art_classifier.get_activations(test_x_batch)
    grads = classifier.loss_gradient(test_x_batch, test_y_batch)

    attacker = art.attacks.evasion.ProjectedGradientDescent(classifier)

    generated = attacker.generate(test_x_batch)
    print(generated.shape)

