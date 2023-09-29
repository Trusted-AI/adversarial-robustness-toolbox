import numpy as np
from art.estimators.hf_mm import HFMMPyTorch
from art.estimators.hf_mm import MultiModalHuggingFaceInput
import matplotlib.pyplot as plt

from art.attacks.evasion import ProjectedGradientDescent

import torch
from torchvision import datasets

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073])
STD = np.asarray([0.26862954, 0.26130258, 0.27577711])


def get_and_process_input(to_one_hot=False, return_batch=False):

    from PIL import Image
    import requests
    import torch
    from transformers import CLIPProcessor

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    text = ["a photo of a cat", "a photo of a dog", "a photo of a bear"]

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    if return_batch:
        input_list = []
        for _ in range(10):
            input_list.append(image)
        inputs = processor(text=text, images=input_list, return_tensors="pt", padding=True)
        original_image = inputs["pixel_values"][0].clone().cpu().numpy()
        if to_one_hot:
            labels = np.zeros((10, 3))
            labels = labels[0:10] + 1
        else:
            labels = np.zeros((10,))

        labels = torch.tensor(labels).type(torch.LongTensor)

    else:

        inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
        original_image = inputs.pixel_values.clone().cpu().numpy()
        labels = torch.tensor(np.asarray([0]))

    return inputs, original_image, labels, len(text)

def norm_bound_eps(eps_bound=None):
    if eps_bound is None:
        eps_bound = np.asarray([8 / 255, 8 / 255, 8 / 255])
    eps_bound = np.abs(eps_bound / STD)
    return eps_bound


def get_cifar_data():
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

    return (x_train[0:100], y_train[0:100]), (x_test[0:100], y_test[0:100])


def attack_clip_pgd():
    from PIL import Image
    import requests

    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    loss_fn = torch.nn.CrossEntropyLoss()

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    text = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    # make a batch
    input_list = []
    input_text = []
    for _ in range(10):
        input_list.append(image)
        input_text.append(text)

    inputs = processor(text=text, images=input_list, return_tensors="pt", padding=True)

    original_image = inputs["pixel_values"][0].clone().cpu().detach().numpy()

    art_classifier = HFMMPyTorch(
        model, loss=loss_fn, clip_values=(np.min(original_image), np.max(original_image)), input_shape=(3, 224, 224)
    )

    my_input = MultiModalHuggingFaceInput(**inputs)

    labels = torch.tensor(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    # loss = art_classifier._get_losses(my_input, labels)
    # grad = art_classifier.loss_gradient(my_input, labels)
    clean_preds = art_classifier.predict(my_input)
    print("The max perturbation is", np.max(np.ones((3, 224, 224)) * np.reshape(norm_bound_eps(), (3, 1, 1))))

    attack = ProjectedGradientDescent(
        art_classifier,
        max_iter=10,
        eps=np.ones((3, 224, 224)) * np.reshape(norm_bound_eps(), (3, 1, 1)),
        eps_step=np.ones((3, 224, 224)) * 0.1,
    )
    x_adv = attack.generate(my_input, labels)
    adv_preds = art_classifier.predict(x_adv)

    eps = norm_bound_eps()

    np.save("eps_mins.npy", original_image - eps.reshape((1, 3, 1, 1)))
    np.save("eps_maxs.npy", original_image + eps.reshape((1, 3, 1, 1)))
    np.save("original_image.npy", original_image)

    print(clean_preds)
    print(adv_preds)


def cifar_clip_pgd():
    from PIL import Image
    import requests

    from transformers import CLIPProcessor, CLIPModel
    image_list = ['000000039769.jpg',
                  '000000000285.jpg',
                  '000000002006.jpg',
                  '000000002149.jpg',
                  '000000005992.jpg',
                  '000000011615.jpg',
                  '000000013597.jpg']
    text = ["a photo of a cat", "a photo of a bear", "a photo of a car", "a photo of a bus", "apples"]

    labels = torch.tensor(np.asarray([0, 1, 3]))

    input_list = []
    for fname in ['000000039769.jpg', '000000000285.jpg', '000000002006.jpg', '000000002149.jpg']:
        url = 'http://images.cocodataset.org/val2017/' + fname
        input_list.append(Image.open(requests.get(url, stream=True).raw))

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    loss_fn = torch.nn.CrossEntropyLoss()
    inputs = processor(text=text, images=input_list, return_tensors="pt", padding=True)
    original_images = []
    for i in range(3):
        original_images.append(inputs["pixel_values"][i].clone().cpu().detach().numpy())

    original_images = np.concatenate(original_images)


    art_classifier = HFMMPyTorch(
        model, loss=loss_fn, clip_values=(np.min(original_images), np.max(original_images)), input_shape=(3, 224, 224)
    )

    my_input = MultiModalHuggingFaceInput(**inputs)
    clean_preds = art_classifier.predict(my_input)
    print(clean_preds)

    attack = ProjectedGradientDescent(
        art_classifier,
        max_iter=10,
        eps=np.ones((3, 224, 224)) * np.reshape(norm_bound_eps(), (3, 1, 1)),
        eps_step=np.ones((3, 224, 224)) * 0.1,
    )
    x_adv = attack.generate(my_input, labels)
    adv_preds = art_classifier.predict(x_adv)

    print(clean_preds)
    print(adv_preds)

def test_fit():
    from transformers import CLIPProcessor, CLIPModel

    (x_train, y_train), (x_test, y_test) = get_cifar_data()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    text = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    inputs = processor(text=text, images=x_train, return_tensors="pt", padding=True)
    original_image = inputs["pixel_values"][0].clone().cpu().detach().numpy()

    inputs = MultiModalHuggingFaceInput(**inputs)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    art_classifier = HFMMPyTorch(
        model,
        optimizer=optimizer,
        nb_classes=10,
        loss=torch.nn.CrossEntropyLoss(),
        clip_values=(np.min(original_image),
                     np.max(original_image)),
        input_shape=(3, 224, 224)
    )

    num_of_samples = len(inputs)
    print(num_of_samples)
    art_classifier.fit(inputs, y_train)


def test_predict():
    import torch
    from transformers import CLIPModel
    from art.estimators.hf_mm import HFMMPyTorch, MultiModalHuggingFaceInput

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    inputs, original_image, labels, num_classes = get_and_process_input()

    art_classifier = HFMMPyTorch(
        model,
        nb_classes=num_classes,
        loss=torch.nn.CrossEntropyLoss(), clip_values=(np.min(original_image), np.max(original_image)), input_shape=(3, 224, 224)
    )
    inputs = MultiModalHuggingFaceInput(**inputs)

    preds = art_classifier.predict(inputs)
    print('Pred shape is ', preds.shape)

test_predict()
# test_fit()
# attack_clip_pgd()
# cifar_clip_pgd()
