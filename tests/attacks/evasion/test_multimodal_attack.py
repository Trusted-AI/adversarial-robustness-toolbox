import numpy as np
import pytest

MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073])
STD = np.asarray([0.26862954, 0.26130258, 0.27577711])


def norm_bound_eps(eps_bound=None):
    if eps_bound is None:
        eps_bound = np.asarray([8 / 255, 8 / 255, 8 / 255])
    eps_bound = np.abs(eps_bound / STD)
    return eps_bound


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
    from transformers import CLIPModel
    from art.estimators.hf_mm import HFMMPyTorch, MultiModalHuggingFaceInput

    def grad_art():
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        inputs, original_image, labels, num_classes = get_and_process_input(return_batch=False)

        my_input = MultiModalHuggingFaceInput(**inputs)
        for _ in range(max_iter):
            art_classifier = HFMMPyTorch(model,
                                         nb_classes=num_classes,
                                         loss=torch.nn.CrossEntropyLoss(),
                                         input_shape=(3, 224, 224))
            loss_grad = art_classifier.loss_gradient(my_input, labels)
        return loss_grad

    def manual_grad():
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        inputs, original_image, labels, num_classes = get_and_process_input(return_batch=False)

        inputs.pixel_values.requires_grad_(True)
        lossfn = torch.nn.CrossEntropyLoss()
        for _ in range(max_iter):
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # image-text similarity score

            loss = lossfn(logits_per_image, labels)
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
    from transformers import CLIPModel

    from art.estimators.hf_mm import HFMMPyTorch, MultiModalHuggingFaceInput
    from art.attacks.evasion import ProjectedGradientDescentNumpy

    def attack_clip():

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        loss_fn = torch.nn.CrossEntropyLoss()
        inputs, original_image, labels, num_classes = get_and_process_input(return_batch=to_batch)
        original_image = inputs.pixel_values.clone().cpu().numpy()

        my_input = MultiModalHuggingFaceInput(**inputs)
        art_classifier = HFMMPyTorch(
            model,
            nb_classes=num_classes,
            loss=loss_fn, clip_values=(np.min(original_image), np.max(original_image)), input_shape=(3, 224, 224)
        )

        attack = ProjectedGradientDescentNumpy(
            art_classifier,
            max_iter=2,
            eps=np.ones((3, 224, 224)) * np.reshape(norm_bound_eps(), (3, 1, 1)),
            eps_step=np.ones((3, 224, 224)) * 0.1,
        )

        perturbation = attack._compute_perturbation(my_input, labels, mask=None)

        adv_art_x = attack._apply_perturbation(my_input[0:], perturbation, attack.eps_step)

        return perturbation, adv_art_x["pixel_values"].cpu().detach().numpy()

    def manual_attack():

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        inputs, original_image, labels, num_classes = get_and_process_input(return_batch=to_batch)
        lossfn = torch.nn.CrossEntropyLoss()

        inputs["pixel_values"] = inputs["pixel_values"].requires_grad_(True)

        outputs = model(**inputs)
        loss = lossfn(outputs.logits_per_image, labels)
        loss.backward()
        sign = torch.sign(inputs["pixel_values"].grad)

        init_max = torch.max(inputs["pixel_values"])
        init_min = torch.min(inputs["pixel_values"])

        eps = norm_bound_eps()

        mins = torch.tensor(original_image - eps.reshape((1, 3, 1, 1))).float()
        maxs = torch.tensor(original_image + eps.reshape((1, 3, 1, 1))).float()

        inputs["pixel_values"] = torch.clamp(inputs["pixel_values"] + sign * 0.1, min=init_min, max=init_max)
        pixel_values = torch.clamp(inputs["pixel_values"], min=mins, max=maxs)

        return sign.cpu().detach().numpy(), pixel_values.cpu().detach().numpy()

    manual_pert, manual_sample = manual_attack()
    perturbation, current_x = attack_clip()

    assert np.allclose(perturbation, manual_pert)
    assert np.allclose(manual_sample, current_x)


@pytest.mark.only_with_platform("huggingface")
@pytest.mark.parametrize("max_iter", [1, 5])
@pytest.mark.parametrize("to_one_hot", [True, False])
def test_equivalence(max_iter, to_one_hot):
    """
    Test that the result from using ART tools matches that obtained by manual calculation.
    """
    import torch
    from transformers import CLIPModel

    from art.estimators.hf_mm import HFMMPyTorch, MultiModalHuggingFaceInput
    from art.attacks.evasion import ProjectedGradientDescent

    def attack_clip():

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        loss_fn = torch.nn.CrossEntropyLoss()

        inputs, original_image, labels, num_classes = get_and_process_input(to_one_hot=to_one_hot, return_batch=False)
        original_image = inputs.pixel_values.clone().cpu().numpy()

        my_input = MultiModalHuggingFaceInput(**inputs)
        eps = norm_bound_eps()
        art_classifier = HFMMPyTorch(
            model,
            nb_classes=num_classes,
            loss=loss_fn, clip_values=(np.min(original_image), np.max(original_image)), input_shape=(3, 224, 224)
        )

        attack = ProjectedGradientDescent(
            art_classifier,
            max_iter=max_iter,
            eps=np.ones((3, 224, 224)) * np.reshape(norm_bound_eps(), (3, 1, 1)),
            eps_step=np.ones((3, 224, 224)) * 0.1,
            targeted=False,
            num_random_init=0,
        )

        x_adv = attack.generate(my_input, labels)
        x_adv = x_adv[0]
        check_vals = torch.reshape(x_adv["pixel_values"], (-1,))

        assert torch.all(torch.ge(check_vals, np.min(original_image)))
        assert torch.all(torch.le(check_vals, np.max(original_image)))

        eps_mins = torch.tensor(original_image - eps.reshape((1, 3, 1, 1))).float()
        eps_maxs = torch.tensor(original_image + eps.reshape((1, 3, 1, 1))).float()

        eps_mins = torch.reshape(eps_mins, (-1,))
        eps_maxs = torch.reshape(eps_maxs, (-1,))

        assert torch.all(torch.ge(check_vals, eps_mins))
        assert torch.all(torch.le(check_vals, eps_maxs))

        return x_adv

    def manual_attack():

        lossfn = torch.nn.CrossEntropyLoss()
        eps = norm_bound_eps()
        adv_current = None
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        for i in range(max_iter):

            inputs, original_image, labels, num_classes = get_and_process_input()

            eps_mins = torch.tensor(original_image - eps.reshape((1, 3, 1, 1))).float()
            eps_maxs = torch.tensor(original_image + eps.reshape((1, 3, 1, 1))).float()
            init_max = torch.max(inputs["pixel_values"])
            init_min = torch.min(inputs["pixel_values"])

            if adv_current is not None:
                inputs["pixel_values"] = torch.tensor(adv_current, requires_grad=True)
            else:
                inputs["pixel_values"].requires_grad_(True)

            outputs = model(**inputs)

            loss = lossfn(outputs.logits_per_image, labels)
            loss.backward()

            sign = torch.sign(inputs["pixel_values"].grad)
            pixel_values = torch.clamp(inputs["pixel_values"] + sign * 0.1, min=init_min, max=init_max)
            pixel_values = torch.clamp(pixel_values, min=eps_mins, max=eps_maxs)

            model.zero_grad()

            adv_current = pixel_values.cpu().detach().numpy()

        return adv_current

    manual_adv = manual_attack()
    art_adv = attack_clip()

    art_adv = art_adv["pixel_values"]

    assert np.allclose(art_adv, manual_adv[0])


"""
TODO: move some of the fits to more appropriate testing files
"""
@pytest.mark.only_with_platform("huggingface")
def test_predict():
    import torch
    from transformers import CLIPModel
    from art.estimators.hf_mm import HFMMPyTorch, MultiModalHuggingFaceInput

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    inputs, original_image, labels, num_classes = get_and_process_input(return_batch=True)

    art_classifier = HFMMPyTorch(
        model,
        nb_classes=num_classes,
        loss=torch.nn.CrossEntropyLoss(), clip_values=(np.min(original_image), np.max(original_image)), input_shape=(3, 224, 224)
    )
    inputs = MultiModalHuggingFaceInput(**inputs)
    preds = art_classifier.predict(inputs)

