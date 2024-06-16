import os
import numpy as np
import pytest

from art.utils import load_dataset
from tests.utils import ARTTestException


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


@pytest.mark.only_with_platform("huggingface")
def test_predict(art_warning):
    """
    Assert predictions function as expected.
    """
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        from art.experimental.estimators import (
            HuggingFaceMultiModalPyTorch,
            HuggingFaceMultiModalInput,
        )

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../utils/data/images/")

        text = [
            "a photo of pink flowers",
            "a photo of birds by the sea",
            "a photo of a forest",
            "a photo of a fern",
            "a photo of a bus",
        ]

        input_list = []
        for fname in ["flowers", "birds", "forest", "ferns"]:
            image = np.load(os.path.join(fpath, fname + ".npy"))
            input_list.append(image)

        labels = np.asarray([0, 1, 2, 3])
        inputs = processor(text=text, images=input_list, return_tensors="pt", padding=True)
        original_images = []
        for i in range(len(labels)):
            original_images.append(inputs["pixel_values"][i].clone().cpu().detach().numpy())

        original_images = np.stack(original_images)

        art_classifier = HuggingFaceMultiModalPyTorch(
            model,
            loss=torch.nn.CrossEntropyLoss(),
            clip_values=(np.min(original_images), np.max(original_images)),
            input_shape=(3, 224, 224),
        )
        inputs = HuggingFaceMultiModalInput(**inputs)
        predictions = art_classifier.predict(inputs)
        assert (np.sum(np.argmax(predictions, axis=1) == labels) / len(labels)) == 1.0
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("huggingface")
def test_fit(art_warning, fix_get_cifar10_data):
    """
    Assert training loop executes.
    """
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        from art.experimental.estimators import (
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
    except ARTTestException as e:
        art_warning(e)
