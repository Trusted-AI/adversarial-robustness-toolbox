import numpy as np
import pytest


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


@pytest.mark.only_with_platform("huggingface")
@pytest.mark.parametrize("max_iter", [1, 5])
def test_grad_equivalence(max_iter):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import CLIPModel
    from art.experimental.estimators.huggingface_multimodal import (
        HuggingFaceMulitModalPyTorch,
        HuggingFaceMultiModalInput,
    )

    def grad_art():
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        inputs, original_image, labels, num_classes = get_and_process_input(return_batch=False)

        my_input = HuggingFaceMultiModalInput(**inputs)
        for _ in range(max_iter):
            art_classifier = HuggingFaceMulitModalPyTorch(
                model,
                loss=torch.nn.CrossEntropyLoss(),
                input_shape=(3, 224, 224),
                device_type="gpu",
            )
            loss_grad = art_classifier.loss_gradient(my_input, labels)
        return loss_grad

    def manual_grad():
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model = model.to(device)
        inputs, original_image, labels, num_classes = get_and_process_input(return_batch=False)
        inputs = inputs.to(device)

        inputs.pixel_values.requires_grad_(True)
        lossfn = torch.nn.CrossEntropyLoss()
        for _ in range(max_iter):
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # image-text similarity score

            loss = lossfn(logits_per_image, labels.to(device))
            loss.backward()

        return inputs.pixel_values.grad

    art = grad_art()
    manual = manual_grad()
    assert np.allclose(art, manual.cpu().detach().numpy())


@pytest.mark.only_with_platform("huggingface")
@pytest.mark.parametrize("to_batch", [False, True])
def test_perturbation_equivalence(to_batch):
    """
    Test that the perturbation from using ART tools matches that obtained by manual calculation.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import CLIPModel

    from art.experimental.estimators.huggingface_multimodal import (
        HuggingFaceMulitModalPyTorch,
        HuggingFaceMultiModalInput,
    )
    from art.experimental.attacks.evasion import CLIPProjectedGradientDescentNumpy

    def attack_clip():

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        loss_fn = torch.nn.CrossEntropyLoss()
        inputs, original_image, labels, num_classes = get_and_process_input(return_batch=to_batch)
        original_image = inputs.pixel_values.clone().cpu().numpy()

        my_input = HuggingFaceMultiModalInput(**inputs)
        art_classifier = HuggingFaceMulitModalPyTorch(
            model,
            loss=loss_fn,
            clip_values=(np.min(original_image), np.max(original_image)),
            input_shape=(3, 224, 224),
        )

        attack = CLIPProjectedGradientDescentNumpy(
            art_classifier,
            max_iter=2,
            eps=np.ones((3, 224, 224)) * 0.3,  # np.reshape(norm_bound_eps(), (3, 1, 1)),
            eps_step=np.ones((3, 224, 224)) * 0.1,
        )

        perturbation = attack._compute_perturbation(my_input, labels, mask=None)

        adv_art_x = attack._apply_perturbation(my_input[0:], perturbation, attack.eps_step)

        return perturbation, adv_art_x["pixel_values"].cpu().detach().numpy()

    def manual_attack():

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model = model.to(device)
        inputs, original_image, labels, num_classes = get_and_process_input(return_batch=to_batch)
        lossfn = torch.nn.CrossEntropyLoss()
        inputs = inputs.to(device)

        inputs["pixel_values"] = inputs["pixel_values"].requires_grad_(True)

        outputs = model(**inputs)
        loss = lossfn(outputs.logits_per_image, labels.to(device))
        loss.backward()
        sign = torch.sign(inputs["pixel_values"].grad)

        init_max = torch.max(inputs["pixel_values"])
        init_min = torch.min(inputs["pixel_values"])

        mins = torch.tensor(original_image - 0.3).float().to(device)
        maxs = torch.tensor(original_image + 0.3).float().to(device)

        inputs["pixel_values"] = torch.clamp(inputs["pixel_values"] + sign * 0.1, min=init_min, max=init_max)
        pixel_values = torch.clamp(inputs["pixel_values"], min=mins, max=maxs)

        return sign.cpu().detach().numpy(), pixel_values.cpu().detach().numpy()

    manual_pert, manual_sample = manual_attack()
    perturbation, current_x = attack_clip()

    assert np.allclose(perturbation, manual_pert)
    assert np.allclose(manual_sample, current_x)


def test_attack_functionality():

    import torch
    import requests
    from PIL import Image

    from transformers import CLIPProcessor, CLIPModel

    from art.experimental.estimators.huggingface_multimodal import (
        HuggingFaceMulitModalPyTorch,
        HuggingFaceMultiModalInput,
    )
    from art.experimental.attacks.evasion import CLIPProjectedGradientDescentNumpy

    std = np.asarray([0.26862954, 0.26130258, 0.27577711])

    def norm_bound_eps(eps_bound=None):
        if eps_bound is None:
            eps_bound = np.asarray([8 / 255, 8 / 255, 8 / 255])
        eps_bound = np.abs(eps_bound / std)
        return eps_bound

    text = ["a photo of a cat", "a photo of a bear", "a photo of a car", "a photo of a bus", "apples"]

    labels = torch.tensor(np.asarray([0, 1, 3, 4]))

    input_list = []
    for fname in ["000000039769.jpg", "000000000285.jpg", "000000002006.jpg", "000000002149.jpg"]:
        url = "http://images.cocodataset.org/val2017/" + fname
        input_list.append(Image.open(requests.get(url, stream=True).raw))

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    loss_fn = torch.nn.CrossEntropyLoss()
    inputs = processor(text=text, images=input_list, return_tensors="pt", padding=True)
    original_images = []
    for i in range(4):
        original_images.append(inputs["pixel_values"][i].clone().cpu().detach().numpy())

    original_images = np.stack(original_images)

    art_classifier = HuggingFaceMulitModalPyTorch(
        model,
        loss=loss_fn,
        clip_values=(np.min(original_images), np.max(original_images)),
        input_shape=(3, 224, 224),
    )

    my_input = HuggingFaceMultiModalInput(**inputs)
    clean_preds = art_classifier.predict(my_input)
    clean_acc = np.sum(np.argmax(clean_preds, axis=1) == labels.cpu().detach().numpy()) / len(labels)

    attack = CLIPProjectedGradientDescentNumpy(
        art_classifier,
        max_iter=10,
        eps=np.ones((3, 224, 224)) * np.reshape(norm_bound_eps(), (3, 1, 1)),
        eps_step=np.ones((3, 224, 224)) * 0.1,
    )
    x_adv = attack.generate(my_input, labels)
    adv_preds = art_classifier.predict(x_adv)
    adv_acc = np.sum(np.argmax(adv_preds, axis=1) == labels.cpu().detach().numpy()) / len(labels)

    x_adv = x_adv["pixel_values"]
    x_adv = x_adv.cpu().detach().numpy()

    assert np.all(x_adv >= np.min(original_images))
    assert np.all(x_adv <= np.max(original_images))

    eps_mins = original_images - np.ones((3, 224, 224)) * np.reshape(norm_bound_eps(), (3, 1, 1))
    eps_maxs = original_images + np.ones((3, 224, 224)) * np.reshape(norm_bound_eps(), (3, 1, 1))

    eps_mins = eps_mins.flatten()
    eps_maxs = eps_maxs.flatten()
    x_adv = x_adv.flatten()

    assert np.all(np.logical_or(x_adv >= eps_mins, np.isclose(x_adv, eps_mins)))
    assert np.all(np.logical_or(x_adv <= eps_maxs, np.isclose(x_adv, eps_maxs)))

    assert clean_acc == 1.0
    assert adv_acc == 0.0


@pytest.mark.only_with_platform("huggingface")
def test_predict():
    import torch
    from transformers import CLIPModel
    from art.experimental.estimators.huggingface_multimodal import (
        HuggingFaceMulitModalPyTorch,
        HuggingFaceMultiModalInput,
    )

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    inputs, original_image, labels, num_classes = get_and_process_input(return_batch=True)

    art_classifier = HuggingFaceMulitModalPyTorch(
        model,
        loss=torch.nn.CrossEntropyLoss(),
        clip_values=(np.min(original_image), np.max(original_image)),
        input_shape=(3, 224, 224),
    )
    inputs = HuggingFaceMultiModalInput(**inputs)
    _ = art_classifier.predict(inputs)
