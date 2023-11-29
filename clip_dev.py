import numpy as np
from art.experimental.estimators.huggingface_multimodal import HuggingFaceMulitModalPyTorch, HuggingFaceMultiModalInput
from art.experimental.attacks.evasion import CLIPProjectedGradientDescentNumpy

import torch
from torchvision import datasets

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073])
STD = np.asarray([0.26862954, 0.26130258, 0.27577711])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    train_set = datasets.CIFAR10("./data", train=True, download=True)
    test_set = datasets.CIFAR10("./data", train=False, download=True)

    x_train = train_set.data.astype(np.float32)
    y_train = np.asarray(train_set.targets)

    x_test = test_set.data.astype(np.float32)
    y_test = np.asarray(test_set.targets)

    x_train = np.moveaxis(x_train, [3], [1])
    x_test = np.moveaxis(x_test, [3], [1])

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return (x_train[0:250], y_train[0:250]), (x_test[0:250], y_test[0:250])


def attack_clip_plant_pgd():
    from PIL import Image

    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    loss_fn = torch.nn.CrossEntropyLoss()

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    text = ["a photo of some plants", "a photo of a dog", "a photo of a car"]

    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open("ART_Test_Image.jpg")
    image = np.array(image)
    np.save("ART_Test_Image.npy", image)
    # make a batch
    input_list = []
    input_text = []
    for _ in range(1):
        input_list.append(image)
        input_text.append(text)

    inputs = processor(text=text, images=input_list, return_tensors="pt", padding=True)

    original_image = inputs["pixel_values"][0].clone().cpu().detach().numpy()

    art_classifier = HuggingFaceMulitModalPyTorch(
        model, loss=loss_fn, clip_values=(np.min(original_image), np.max(original_image)), input_shape=(3, 224, 224)
    )

    my_input = HuggingFaceMultiModalInput(**inputs)

    labels = torch.tensor(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    # loss = art_classifier._get_losses(my_input, labels)
    # grad = art_classifier.loss_gradient(my_input, labels)
    clean_preds = art_classifier.predict(my_input)
    print(clean_preds)

    print("The max perturbation is", np.max(np.ones((3, 224, 224)) * np.reshape(norm_bound_eps(), (3, 1, 1))))

    attack = CLIPProjectedGradientDescentNumpy(
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

    print(adv_preds)


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

    art_classifier = HuggingFaceMulitModalPyTorch(
        model, loss=loss_fn, clip_values=(np.min(original_image), np.max(original_image)), input_shape=(3, 224, 224)
    )

    my_input = HuggingFaceMultiModalInput(**inputs)

    labels = torch.tensor(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    # loss = art_classifier._get_losses(my_input, labels)
    # grad = art_classifier.loss_gradient(my_input, labels)
    clean_preds = art_classifier.predict(my_input)
    print("The max perturbation is", np.max(np.ones((3, 224, 224)) * np.reshape(norm_bound_eps(), (3, 1, 1))))

    attack = CLIPProjectedGradientDescentNumpy(
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
    from art.experimental.attacks.evasion import CLIPProjectedGradientDescentNumpy

    """
    text = ["a photo of a cat", "a photo of a bear", "a photo of a car", "a photo of a bus", "apples"]

    labels = torch.tensor(np.asarray([0, 1, 3, 4]))

    input_list = []
    for fname in ["000000039769.jpg", "000000000285.jpg", "000000002006.jpg", "000000002149.jpg"]:
        url = "http://images.cocodataset.org/val2017/" + fname
        input_list.append(Image.open(requests.get(url, stream=True).raw))
    """
    text = [
        "a photo of pink flowers",
        "a photo of birds by the sea",
        "a photo of a forest",
        "a photo of a fern",
        "a photo of a bus",
    ]

    input_list = []
    for fname in ["flowers", "birds", "forest", "ferns"]:
        image = Image.open(fname + ".jpg")
        image = np.array(image)
        np.save(fname + ".npy", image)
        print("image shape is ", image.shape)
        input_list.append(image)

    labels = torch.tensor(np.asarray([0, 1, 2, 3]))

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    loss_fn = torch.nn.CrossEntropyLoss()
    inputs = processor(text=text, images=input_list, return_tensors="pt", padding=True)
    original_images = []
    for i in range(3):
        original_images.append(inputs["pixel_values"][i].clone().cpu().detach().numpy())

    original_images = np.stack(original_images)
    print("input shape is ", original_images.shape)

    art_classifier = HuggingFaceMulitModalPyTorch(
        model,
        loss=loss_fn,
        clip_values=(np.min(original_images), np.max(original_images)),
        input_shape=(3, 224, 224),
    )

    my_input = HuggingFaceMultiModalInput(**inputs)
    clean_preds = art_classifier.predict(my_input)
    print(clean_preds)
    clean_acc = np.sum(np.argmax(clean_preds, axis=1) == labels.cpu().detach().numpy()) / len(labels)
    print("clean acc ", clean_acc)
    attack = CLIPProjectedGradientDescentNumpy(
        art_classifier,
        max_iter=10,
        eps=np.ones((3, 224, 224)) * np.reshape(norm_bound_eps(), (3, 1, 1)),
        eps_step=np.ones((3, 224, 224)) * 0.1,
    )
    x_adv = attack.generate(my_input, labels)
    adv_preds = art_classifier.predict(x_adv)
    adv_acc = np.sum(np.argmax(adv_preds, axis=1) == labels.cpu().detach().numpy()) / len(labels)
    print("adv_acc ", adv_acc)

    print(clean_preds)
    print(adv_preds)


def test_fit():
    from transformers import CLIPProcessor, CLIPModel

    (x_train, y_train), (_, _) = get_cifar_data()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    text = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    inputs = processor(text=text, images=x_train, return_tensors="pt", padding=True)
    original_image = inputs["pixel_values"][0].clone().cpu().detach().numpy()

    inputs = HuggingFaceMultiModalInput(**inputs)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    art_classifier = HuggingFaceMulitModalPyTorch(
        model,
        optimizer=optimizer,
        nb_classes=10,
        loss=torch.nn.CrossEntropyLoss(),
        clip_values=(np.min(original_image), np.max(original_image)),
        input_shape=(3, 224, 224),
    )

    num_of_samples = len(inputs)
    print(num_of_samples)
    art_classifier.fit(inputs, y_train)


def test_predict():
    import torch
    from transformers import CLIPModel

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    inputs, original_image, labels, num_classes = get_and_process_input()

    art_classifier = HuggingFaceMulitModalPyTorch(
        model,
        nb_classes=num_classes,
        loss=torch.nn.CrossEntropyLoss(),
        clip_values=(np.min(original_image), np.max(original_image)),
        input_shape=(3, 224, 224),
    )
    inputs = HuggingFaceMultiModalInput(**inputs)

    preds = art_classifier.predict(inputs)
    print("Pred shape is ", preds.shape)


def test_adv_train():
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from art.defences.trainer import AdversarialTrainer
    from art.experimental.attacks.evasion import CLIPProjectedGradientDescentNumpy

    (x_train, y_train), (_, _) = get_cifar_data()

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    inputs, original_image, _, num_classes = get_and_process_input()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    text = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    inputs = processor(text=text, images=x_train, return_tensors="pt", padding=True)
    original_image = inputs["pixel_values"][0].clone().cpu().detach().numpy()

    art_classifier = HuggingFaceMulitModalPyTorch(
        model.to(device),
        nb_classes=num_classes,
        optimizer=optimizer,
        loss=torch.nn.CrossEntropyLoss(),
        clip_values=(np.min(original_image), np.max(original_image)),
        input_shape=(3, 224, 224),
    )

    attack = CLIPProjectedGradientDescentNumpy(
        art_classifier,
        max_iter=10,
        eps=np.ones((3, 224, 224)) * np.reshape(norm_bound_eps(), (3, 1, 1)),
        eps_step=np.ones((3, 224, 224)) * 0.1,
    )

    trainer = AdversarialTrainer(
        art_classifier,
        attacks=attack,
        ratio=1.0,
    )
    inputs = HuggingFaceMultiModalInput(**inputs)

    trainer.fit(inputs, y_train)


# test_adv_train()
# test_predict()
# test_fit()
# attack_clip_pgd()
cifar_clip_pgd()
# attack_clip_plant_pgd()
