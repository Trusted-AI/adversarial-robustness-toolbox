import os
import numpy as np
import pytest

from art.utils import load_dataset


@pytest.fixture()
def fix_get_cifar10_data():
    """
    Get the first 128 samples of the cifar10 test set

    :return: First 128 sample/label pairs of the cifar10 test dataset.
    """
    nb_test = 128

    (_, _), (x_test, y_test), _, _ = load_dataset("cifar10")
    y_test = np.argmax(y_test, axis=1)
    x_test, y_test = x_test[:nb_test], y_test[:nb_test]
    x_test = np.transpose(x_test, (0, 3, 1, 2))  # return in channels first format
    return x_test.astype(np.float32), y_test


def get_and_process_input(to_one_hot=False, return_batch=False):

    import torch
    from transformers import CLIPProcessor

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    text = ["a photo of pink flowers", "a photo of a dog", "a photo of a bear"]

    fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../utils/data/images/flowers.npy")

    image = np.load(fpath)

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


@pytest.mark.only_with_platform("huggingface")
def test_predict():
    import torch
    from transformers import CLIPModel
    from art.experimental.estimators.huggingface_multimodal import (
        HuggingFaceMultiModalPyTorch,
        HuggingFaceMultiModalInput,
    )

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    inputs, original_image, labels, num_classes = get_and_process_input(return_batch=True)

    art_classifier = HuggingFaceMultiModalPyTorch(
        model,
        loss=torch.nn.CrossEntropyLoss(),
        clip_values=(np.min(original_image), np.max(original_image)),
        input_shape=(3, 224, 224),
    )
    inputs = HuggingFaceMultiModalInput(**inputs)
    _ = art_classifier.predict(inputs)


@pytest.mark.only_with_platform("huggingface")
def test_fit(fix_get_cifar10_data):
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from art.experimental.estimators.huggingface_multimodal import (
        HuggingFaceMultiModalPyTorch,
        HuggingFaceMultiModalInput,
    )

    x_train = fix_get_cifar10_data[0]
    y_train = fix_get_cifar10_data[1]

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    text = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    inputs = processor(text=text, images=x_train, return_tensors="pt", padding=True)
    original_image = inputs["pixel_values"][0].clone().cpu().detach().numpy()

    inputs = HuggingFaceMultiModalInput(**inputs)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    art_classifier = HuggingFaceMultiModalPyTorch(
        model,
        optimizer=optimizer,
        loss=torch.nn.CrossEntropyLoss(),
        clip_values=(np.min(original_image), np.max(original_image)),
        input_shape=(3, 224, 224),
    )

    art_classifier.fit(inputs, y_train, nb_epochs=1)
