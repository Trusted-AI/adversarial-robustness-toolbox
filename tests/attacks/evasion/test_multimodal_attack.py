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
    from art.experimental.estimators.huggingface_multimodal import HFMMPyTorch, HuggingFaceMultiModalInput

    def grad_art():
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        inputs, original_image, labels, num_classes = get_and_process_input(return_batch=False)

        my_input = HuggingFaceMultiModalInput(**inputs)
        for _ in range(max_iter):
            art_classifier = HFMMPyTorch(
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

    from art.experimental.estimators.huggingface_multimodal import HFMMPyTorch, HuggingFaceMultiModalInput
    from art.experimental.attacks.evasion import CLIPProjectedGradientDescentNumpy

    def attack_clip():

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        loss_fn = torch.nn.CrossEntropyLoss()
        inputs, original_image, labels, num_classes = get_and_process_input(return_batch=to_batch)
        original_image = inputs.pixel_values.clone().cpu().numpy()

        my_input = HuggingFaceMultiModalInput(**inputs)
        art_classifier = HFMMPyTorch(
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


@pytest.mark.only_with_platform("huggingface")
@pytest.mark.parametrize("max_iter", [1, 5])
def test_equivalence(max_iter):
    """
    Test that the result from using ART tools matches that obtained by manual calculation.
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import CLIPModel

    from art.experimental.estimators.huggingface_multimodal import HFMMPyTorch, HuggingFaceMultiModalInput
    from art.experimental.attacks.evasion import CLIPProjectedGradientDescentNumpy

    def attack_clip():

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        loss_fn = torch.nn.CrossEntropyLoss()

        inputs, original_image, labels, num_classes = get_and_process_input()
        original_image = inputs.pixel_values.clone().cpu().numpy()

        my_input = HuggingFaceMultiModalInput(**inputs)

        art_classifier = HFMMPyTorch(
            model,
            loss=loss_fn,
            clip_values=(np.min(original_image), np.max(original_image)),
            input_shape=(3, 224, 224),
        )

        attack = CLIPProjectedGradientDescentNumpy(
            art_classifier,
            max_iter=max_iter,
            eps=np.ones((3, 224, 224)) * 0.3,
            eps_step=np.ones((3, 224, 224)) * 0.1,
            targeted=False,
            num_random_init=0,
        )

        x_adv = attack.generate(my_input, labels)
        x_adv = x_adv[0]
        check_vals = torch.reshape(x_adv["pixel_values"], (-1,))

        assert torch.all(torch.ge(check_vals, np.min(original_image)))
        assert torch.all(torch.le(check_vals, np.max(original_image)))

        eps_mins = torch.tensor(original_image - 0.3).float()
        eps_maxs = torch.tensor(original_image + 0.3).float()
        eps_mins = torch.reshape(eps_mins, (-1,))
        eps_maxs = torch.reshape(eps_maxs, (-1,))

        assert torch.all(torch.ge(check_vals, eps_mins))
        assert torch.all(torch.le(check_vals, eps_maxs))

        return x_adv

    def manual_attack():

        lossfn = torch.nn.CrossEntropyLoss()

        adv_current = None
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model = model.to(device)

        for i in range(max_iter):

            inputs, original_image, labels, _ = get_and_process_input()
            inputs = inputs.to(device)

            eps_mins = torch.tensor(original_image - 0.3).float().to(device)
            eps_maxs = torch.tensor(original_image + 0.3).float().to(device)

            init_max = torch.max(inputs["pixel_values"]).to(device)
            init_min = torch.min(inputs["pixel_values"]).to(device)

            if adv_current is not None:
                inputs["pixel_values"] = torch.tensor(adv_current).to(device)
            inputs["pixel_values"].requires_grad_(True)

            outputs = model(**inputs)

            loss = lossfn(outputs.logits_per_image, labels.to(device))
            loss.backward()

            sign = torch.sign(inputs["pixel_values"].grad)
            pixel_values = torch.clamp(inputs["pixel_values"] + sign * 0.1, min=init_min, max=init_max)
            pixel_values = torch.clamp(pixel_values, min=eps_mins, max=eps_maxs)

            model.zero_grad()

            adv_current = pixel_values.cpu().detach().numpy()

        return adv_current

    inputs, original_image, labels, num_classes = get_and_process_input()
    manual_adv = manual_attack()
    art_adv = attack_clip()

    art_adv = art_adv["pixel_values"]
    art_adv = art_adv.cpu().detach().numpy()

    art_adv = art_adv.flatten()
    original_image = original_image.flatten()
    manual_adv = manual_adv.flatten()

    """
    Assert valid adversarial examples
    """
    assert np.all(art_adv >= np.min(original_image))
    assert np.all(art_adv <= np.max(original_image))
    assert np.all(manual_adv >= np.min(original_image))
    assert np.all(manual_adv <= np.max(original_image))

    eps_mins = original_image - 0.3
    eps_maxs = original_image + 0.3

    assert np.all(art_adv >= eps_mins)
    assert np.all(art_adv <= eps_maxs)
    assert np.all(manual_adv >= eps_mins)
    assert np.all(manual_adv <= eps_maxs)

    art_delta = art_adv - original_image
    target = manual_adv - original_image
    # np.save('art_adv_' + str(max_iter) + '.npy', art_delta)
    # target = np.load('art_adv_' + str(max_iter) + '.npy')

    assert np.allclose(art_delta, target, rtol=1e-04, atol=1e-04)


"""
TODO: move some of the fits to more appropriate testing files
"""


@pytest.mark.only_with_platform("huggingface")
def test_predict():
    import torch
    from transformers import CLIPModel
    from art.experimental.estimators.huggingface_multimodal import HFMMPyTorch, HuggingFaceMultiModalInput

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    inputs, original_image, labels, num_classes = get_and_process_input(return_batch=True)

    art_classifier = HFMMPyTorch(
        model,
        loss=torch.nn.CrossEntropyLoss(),
        clip_values=(np.min(original_image), np.max(original_image)),
        input_shape=(3, 224, 224),
    )
    inputs = HuggingFaceMultiModalInput(**inputs)
    _ = art_classifier.predict(inputs)
