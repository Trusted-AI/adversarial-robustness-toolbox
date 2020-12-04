# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import argparse
import json
import yaml
import pprint

from art.estimators.object_detection import PyTorchFasterRCNN
from art.attacks.evasion import RobustDPatch


COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def extract_predictions(predictions_):

    # for key, item in predictions[0].items():
    #     print(key, item)

    # Get the predicted class
    predictions_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(predictions_["labels"])]
    print("\npredicted classes:", predictions_class)

    # Get the predicted bounding boxes
    predictions_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions_["boxes"])]

    # Get the predicted prediction score
    predictions_score = list(predictions_["scores"])
    print("predicted score:", predictions_score)

    # Get a list of index with score greater than threshold
    threshold = 0.5
    predictions_t = [predictions_score.index(x) for x in predictions_score if x > threshold][-1]

    predictions_boxes = predictions_boxes[: predictions_t + 1]
    predictions_class = predictions_class[: predictions_t + 1]

    return predictions_class, predictions_boxes, predictions_class


def plot_image_with_boxes(img, boxes, pred_cls):
    text_size = 5
    text_th = 5
    rect_th = 6

    for i in range(len(boxes)):
        # Draw Rectangle with the coordinates
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)

        # Write the prediction class
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)

    plt.axis("off")
    plt.imshow(img.astype(np.uint8), interpolation="nearest")
    plt.show()


def get_loss(frcnn, x, y):
    frcnn._model.train()
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image_tensor_list = list()

    for i in range(x.shape[0]):
        if frcnn.clip_values is not None:
            img = transform(x[i] / frcnn.clip_values[1]).to(frcnn._device)
        else:
            img = transform(x[i]).to(frcnn._device)
        image_tensor_list.append(img)

    loss = frcnn._model(image_tensor_list, y)
    for loss_type in ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]:
        loss[loss_type] = loss[loss_type].cpu().detach().numpy().item()
    return loss


def append_loss_history(loss_history, output):
    for loss in ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]:
        loss_history[loss] += [output[loss]]
    return loss_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, default=None, help="Path of config yaml file")
    cmdline = parser.parse_args()

    if cmdline.config and os.path.exists(cmdline.config):
        with open(cmdline.config, "r") as cf:
            config = yaml.safe_load(cf.read())
    else:
        config = {
            "attack_losses": ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"],
            "cuda_visible_devices": "1",
            "patch_shape": [450, 450, 3],
            "patch_location": [600, 750],
            "crop_range": [0, 0],
            "brightness_range": [1.0, 1.0],
            "rotation_weights": [1, 0, 0, 0],
            "sample_size": 1,
            "learning_rate": 1.0,
            "max_iter": 5000,
            "batch_size": 1,
            "image_file": "banner-diverse-group-of-people-2.jpg",
            "resume": False,
            "path": "xp/",
        }

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)

    if config["cuda_visible_devices"] is None:
        device_type = "cpu"
    else:
        device_type = "gpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]

    frcnn = PyTorchFasterRCNN(
        clip_values=(0, 255), channels_first=False, attack_losses=config["attack_losses"], device_type=device_type
    )

    image_1 = cv2.imread(config["image_file"])
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image_1 = cv2.resize(image_1, dsize=(image_1.shape[1], image_1.shape[0]), interpolation=cv2.INTER_CUBIC)

    image = np.stack([image_1], axis=0).astype(np.float32)

    attack = RobustDPatch(
        frcnn,
        patch_shape=config["patch_shape"],
        patch_location=config["patch_location"],
        crop_range=config["crop_range"],
        brightness_range=config["brightness_range"],
        rotation_weights=config["rotation_weights"],
        sample_size=config["sample_size"],
        learning_rate=config["learning_rate"],
        max_iter=1,
        batch_size=config["batch_size"],
    )

    x = image.copy()

    y = frcnn.predict(x=x)
    for i, y_i in enumerate(y):
        y[i]["boxes"] = torch.from_numpy(y_i["boxes"]).type(torch.float).to(frcnn._device)
        y[i]["labels"] = torch.from_numpy(y_i["labels"]).type(torch.int64).to(frcnn._device)
        y[i]["scores"] = torch.from_numpy(y_i["scores"]).to(frcnn._device)

    if config["resume"]:
        patch = np.load(os.path.join(config["path"], "patch.npy"))
        attack._patch = patch

        with open(os.path.join(config["path"], "loss_history.json"), "r") as file:
            loss_history = json.load(file)
    else:
        loss_history = {"loss_classifier": [], "loss_box_reg": [], "loss_objectness": [], "loss_rpn_box_reg": []}

    for i in range(config["max_iter"]):
        print("Iteration:", i)
        patch = attack.generate(x)
        x_patch = attack.apply_patch(x)

        loss = get_loss(frcnn, x_patch, y)
        print(loss)
        loss_history = append_loss_history(loss_history, loss)

        with open(os.path.join(config["path"], "loss_history.json"), "w") as file:
            file.write(json.dumps(loss_history))

        np.save(os.path.join(config["path"], "patch"), attack._patch)
