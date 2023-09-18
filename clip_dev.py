import numpy as np
from art.estimators.hf_mm import HFMMPyTorch
from art.estimators.hf_mm import ARTInput

from art.attacks.evasion import ProjectedGradientDescentPyTorch, ProjectedGradientDescent

import torch

MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073])
STD = np.asarray([0.26862954, 0.26130258, 0.27577711])


def norm_bound_eps(eps_bound=None):
    if eps_bound is None:
        eps_bound = np.asarray([8/255, 8/255, 8/255])
    eps_bound = np.abs(eps_bound / STD)
    return eps_bound


def attack_clip():
    from PIL import Image
    import requests

    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    loss_fn = torch.nn.CrossEntropyLoss()

    art_classifier = HFMMPyTorch(model,
                                 loss=loss_fn,
                                 input_shape=(3, 224, 224))

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    text = ["a photo of a cat", "a photo of a dog"]

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    # make a batch
    input_list = []
    for _ in range(10):
        input_list.append(image)

    inputs = processor(text=text, images=input_list, return_tensors="pt",
                       padding=True)

    my_input = ARTInput(**inputs)
    check_pixels = my_input['pixel_values']
    print('check_pixels ', check_pixels.shape)
    check_slicing = my_input[0:5]
    print('check_slicing ', check_slicing['pixel_values'].shape)
    check_index = my_input[2]
    print(check_index)

    labels = torch.tensor(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    loss = art_classifier._get_losses(my_input, labels)
    grad = art_classifier.loss_gradient(my_input, labels)

    attack = ProjectedGradientDescent(art_classifier,
                                      max_iter=10,
                                      eps=np.ones((3, 224, 224)) * np.reshape(norm_bound_eps(), (3, 1, 1)),
                                      eps_step=np.ones((3, 224, 224)) * 0.1)
    x_adv = attack.generate(my_input, labels)

attack_clip()
