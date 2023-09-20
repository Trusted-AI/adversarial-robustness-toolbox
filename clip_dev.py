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
    original_image = inputs['pixel_values'][0].clone().cpu().detach().numpy()

    art_classifier = HFMMPyTorch(model,
                                 loss=loss_fn,
                                 clip_values=(np.min(original_image), np.max(original_image)),
                                 input_shape=(3, 224, 224))

    my_input = ARTInput(**inputs)

    labels = torch.tensor(np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    loss = art_classifier._get_losses(my_input, labels)
    grad = art_classifier.loss_gradient(my_input, labels)
    clean_preds = art_classifier.predict(my_input)
    print(clean_preds)
    print('The max perturbation is', np.max(np.ones((3, 224, 224)) * np.reshape(norm_bound_eps(), (3, 1, 1))))

    attack = ProjectedGradientDescent(art_classifier,
                                      max_iter=10,
                                      eps=np.ones((3, 224, 224)) * np.reshape(norm_bound_eps(), (3, 1, 1)),
                                      eps_step=np.ones((3, 224, 224)) * 0.1)
    x_adv = attack.generate(my_input, labels)
    adv_preds = art_classifier.predict(x_adv)

    eps = norm_bound_eps()

    np.save('eps_mins.npy', original_image - eps.reshape((1, 3, 1, 1)))
    np.save('eps_maxs.npy', original_image + eps.reshape((1, 3, 1, 1)))
    np.save('original_image.npy', original_image)

    '''
    eps_mins = torch.tensor(original_image - eps.reshape((1, 3, 1, 1))).float()
    eps_maxs = torch.tensor(original_image + eps.reshape((1, 3, 1, 1))).float()

    eps_mins = torch.reshape(eps_mins, (-1,))
    eps_maxs = torch.reshape(eps_maxs, (-1,))
    check_vals = x_adv['pixel_values']
    check_vals = check_vals[0].clone().cpu().detach()
    check_vals = torch.reshape(check_vals, (-1,))
    '''

    print(clean_preds)
    print(adv_preds)


attack_clip()
