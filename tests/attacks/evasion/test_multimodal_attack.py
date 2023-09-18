import numpy as np
import pytest

MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073])
STD = np.asarray([0.26862954, 0.26130258, 0.27577711])


def norm_bound_eps(eps_bound=None):
    if eps_bound is None:
        eps_bound = np.asarray([8 / 255, 8 / 255, 8 / 255])
    eps_bound = np.abs(eps_bound / STD)
    return eps_bound


def get_and_process_input(return_batch=False):

    from PIL import Image
    import requests
    import torch
    from transformers import CLIPProcessor

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    text = ["a photo of a cat", "a photo of a dog"]

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    if return_batch:
        input_list = []
        for _ in range(10):
            input_list.append(image)
        inputs = processor(text=text, images=input_list, return_tensors="pt",
                           padding=True)
        original_image = inputs['pixel_values'][0].clone().cpu().numpy()
        labels = torch.tensor(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

    else:

        inputs = processor(text=text, images=image, return_tensors="pt",
                           padding=True)
        original_image = inputs.pixel_values.clone().cpu().numpy()
        labels = torch.tensor(np.asarray([0]))

    return inputs, original_image, labels


def test_grad_equivalence():

    import torch
    import numpy as np

    from transformers import CLIPModel

    from art.estimators.hf_mm import HFMMPyTorch, ARTInput

    def grad_art():
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        inputs, original_image, labels = get_and_process_input(return_batch=False)

        my_input = ARTInput(**inputs)
        art_classifier = HFMMPyTorch(model,
                                     loss=torch.nn.CrossEntropyLoss(),
                                     input_shape=(3, 224, 224))

        return art_classifier.loss_gradient(my_input, labels)

    def manual_grad():
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        inputs, original_image, labels = get_and_process_input(return_batch=False)

        inputs.pixel_values.requires_grad_(True)
        lossfn = torch.nn.CrossEntropyLoss()

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # image-text similarity score

        loss = lossfn(logits_per_image, labels)
        loss.backward()

        return inputs.pixel_values.grad

    art = grad_art()
    manual = manual_grad()
    assert np.allclose(art, manual.cpu().detach().numpy())


@pytest.mark.parametrize("to_batch", [False, True])
def test_perturbation_equivalence(to_batch):
    """
    Test that the result from using ART tools matches that obtained by manual calculation.
    """
    import torch
    from transformers import CLIPProcessor, CLIPModel

    import numpy as np
    from art.estimators.hf_mm import HFMMPyTorch, ARTInput
    from art.attacks.evasion import ProjectedGradientDescent, ProjectedGradientDescentNumpy

    def attack_clip():

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        loss_fn = torch.nn.CrossEntropyLoss()
        inputs, original_image, labels = get_and_process_input(return_batch=to_batch)
        original_image = inputs.pixel_values.clone().cpu().numpy()

        my_input = ARTInput(**inputs)
        art_classifier = HFMMPyTorch(model,
                                     loss=loss_fn,
                                     clip_values=(np.min(original_image), np.max(original_image)),
                                     input_shape=(3, 224, 224))

        attack = ProjectedGradientDescentNumpy(art_classifier,
                                               max_iter=2,
                                               eps=np.ones((3, 224, 224)) * np.reshape(norm_bound_eps(), (3, 1, 1)),
                                               eps_step=np.ones((3, 224, 224)) * 0.1,)

        x_pert = attack._compute_perturbation(my_input, labels, mask=None)

        return x_pert

    def manual_attack():

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        inputs, original_image, labels = get_and_process_input(return_batch=to_batch)

        pixel_values = inputs.pixel_values.requires_grad_(True)
        attention_mask = inputs.attention_mask
        input_ids = inputs.input_ids

        lossfn = torch.nn.CrossEntropyLoss()

        inputs = {"pixel_values": pixel_values.requires_grad_(True),
                  "attention_mask": attention_mask,
                  "input_ids": input_ids}

        outputs = model(**inputs)

        loss = lossfn(outputs.logits_per_image, labels)
        loss.backward()

        sign = torch.sign(inputs["pixel_values"].grad)
        model.zero_grad()

        return sign.cpu().detach().numpy()

    manual_pert = manual_attack()
    x_pert = attack_clip()

    assert np.allclose(x_pert, manual_pert)


@pytest.mark.parametrize("to_batch", [False, True])
def test_equivalence(to_batch):
    """
    Test that the result from using ART tools matches that obtained by manual calculation.
    """
    import torch
    from transformers import CLIPProcessor, CLIPModel

    import numpy as np
    from art.estimators.hf_mm import HFMMPyTorch, ARTInput
    from art.attacks.evasion import ProjectedGradientDescent

    from matplotlib import pyplot as plt

    def attack_clip():

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        loss_fn = torch.nn.CrossEntropyLoss()

        inputs, original_image, labels = get_and_process_input(return_batch=to_batch)
        original_image = inputs.pixel_values.clone().cpu().numpy()

        my_input = ARTInput(**inputs)

        art_classifier = HFMMPyTorch(model,
                                     loss=loss_fn,
                                     clip_values=(np.min(original_image), np.max(original_image)),
                                     input_shape=(3, 224, 224))
        clip_min, clip_max = art_classifier.clip_values
        print('Min ', clip_min)
        print('Max ', clip_max)

        attack = ProjectedGradientDescent(art_classifier,
                                          max_iter=2,
                                          eps=np.ones((3, 224, 224)) * np.reshape(norm_bound_eps(), (3, 1, 1)),
                                          eps_step=np.ones((3, 224, 224)) * 0.1)

        x_adv = attack.generate(my_input, labels)
        x_adv = x_adv[0]
        check_vals = torch.reshape(x_adv['pixel_values'], (-1, ))

        for val in check_vals:
            if not torch.ge(val, np.min(original_image)):
                print(f'Val {val} vs Min {np.min(original_image)}')

        # assert torch.all(torch.ge(check_vals, np.min(original_image)))
        # assert torch.all(check_vals <= np.max(original_image))

        '''
        for i, (channel_std, channel_mean) in enumerate(zip(STD, MEAN)):
            x_adv['pixel_values'][i, :, :] = x_adv['pixel_values'][i, :, :] * channel_std
            x_adv['pixel_values'][i, :, :] = x_adv['pixel_values'][i, :, :] + channel_mean

            # original_image[:, i, :, :] = original_image[:, i, :, :] * channel_std
            # original_image[:, i, :, :] = original_image[:, i, :, :] + channel_mean
        pixel_values = x_adv['pixel_values'].cpu().numpy()
        pixel_values = np.squeeze(np.transpose(pixel_values, (1, 2, 0)))
        # original_image = np.squeeze(np.transpose(original_image, (0, 2, 3, 1)))
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(pixel_values)
        plt.show()
        # ax2.imshow(original_image)
        '''
        return x_adv

    def manual_attack():

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        inputs, original_image, labels = get_and_process_input()

        pixel_values = inputs.pixel_values.requires_grad_(True)
        attention_mask = inputs.attention_mask
        input_ids = inputs.input_ids

        init_max = torch.max(pixel_values)
        init_min = torch.min(pixel_values)

        lossfn = torch.nn.CrossEntropyLoss()
        eps = norm_bound_eps()

        mins = torch.tensor(original_image - eps.reshape((1, 3, 1, 1))).float()
        maxs = torch.tensor(original_image + eps.reshape((1, 3, 1, 1))).float()

        for i in range(2):
            print('On step ', i)
            # pixel_values = pixel_values.requires_grad_(True)
            inputs = {"pixel_values": pixel_values.requires_grad_(True),
                      "attention_mask": attention_mask,
                      "input_ids": input_ids}

            outputs = model(**inputs)

            loss = lossfn(outputs.logits_per_image, labels)
            loss.backward()

            with torch.no_grad():
                sign = torch.sign(inputs["pixel_values"].grad)
                inputs["pixel_values"] = torch.clamp(inputs["pixel_values"] + sign * 0.1, min=init_min, max=init_max)
                pixel_values = torch.clamp(inputs["pixel_values"], min=mins, max=maxs)

            model.zero_grad()

        return pixel_values.cpu().detach().numpy()

    manual_adv = manual_attack()
    art_adv = attack_clip()

    art_adv = art_adv['pixel_values']
    print(art_adv.shape)
    print(manual_adv.shape)
    print(art_adv[0:3, 0, 0])
    print(manual_adv[0, 0:3, 0, 0])

    assert np.allclose(art_adv, manual_adv[0])
