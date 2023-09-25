import numpy as np

from matplotlib import pyplot as plt
import torch

MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073])
STD = np.asarray([0.26862954, 0.26130258, 0.27577711])


def norm_bound_eps(eps_bound=None):
    if eps_bound is None:
        eps_bound = np.asarray([8 / 255, 8 / 255, 8 / 255])
    eps_bound = np.abs(eps_bound / STD)
    return eps_bound


def attack_clip():
    from PIL import Image
    import requests

    from transformers import CLIPProcessor, CLIPModel

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    text = ["a photo of a cat", "a photo of a dog"]

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    original_image = inputs.pixel_values.clone().cpu().numpy()

    pixel_values = inputs.pixel_values.requires_grad_(True)
    attention_mask = inputs.attention_mask
    input_ids = inputs.input_ids

    init_max = torch.max(pixel_values)
    init_min = torch.min(pixel_values)

    labels = torch.tensor(np.asarray([0]))
    lossfn = torch.nn.CrossEntropyLoss()
    eps = norm_bound_eps()

    mins = torch.tensor(original_image - eps.reshape((1, 3, 1, 1))).float()
    maxs = torch.tensor(original_image + eps.reshape((1, 3, 1, 1))).float()

    outputs = model(pixel_values=pixel_values, attention_mask=attention_mask, input_ids=input_ids)
    print("Original class: ", text[torch.argmax(outputs.logits_per_image)])

    for i in range(10):
        pixel_values = pixel_values.requires_grad_(True)

        outputs = model(pixel_values=pixel_values, attention_mask=attention_mask, input_ids=input_ids)
        logits_per_image = outputs.logits_per_image  # image-text similarity score

        loss = lossfn(logits_per_image, labels)
        loss.backward()

        with torch.no_grad():
            sign = torch.sign(pixel_values.grad)
            pixel_values = torch.clamp(pixel_values + sign * 0.1, min=init_min, max=init_max)
            pixel_values = torch.clamp(pixel_values, min=mins, max=maxs)

        model.zero_grad()

    outputs = model(pixel_values=pixel_values, attention_mask=attention_mask, input_ids=input_ids)
    print("Attacked class: ", text[torch.argmax(outputs.logits_per_image)])

    for i, (channel_std, channel_mean) in enumerate(zip(STD, MEAN)):
        pixel_values[:, i, :, :] = pixel_values[:, i, :, :] * channel_std
        pixel_values[:, i, :, :] = pixel_values[:, i, :, :] + channel_mean

        original_image[:, i, :, :] = original_image[:, i, :, :] * channel_std
        original_image[:, i, :, :] = original_image[:, i, :, :] + channel_mean

    pixel_values = pixel_values.cpu().numpy()
    pixel_values = np.squeeze(np.transpose(pixel_values, (0, 2, 3, 1)))
    original_image = np.squeeze(np.transpose(original_image, (0, 2, 3, 1)))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(pixel_values)
    ax2.imshow(original_image)
    plt.show()


attack_clip()
