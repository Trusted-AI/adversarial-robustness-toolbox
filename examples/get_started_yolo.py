"""
The script demonstrates a simple example of using ART with YOLO (versions 3 and 5).
The example loads a YOLO model pretrained on the COCO dataset
and creates an adversarial example using Projected Gradient Descent method.

- To use Yolov3, run:
        pip install pytorchyolo

- To use Yolov5, run:
        pip install yolov5

Note: If pytorchyolo throws an error in pytorchyolo/utils/loss.py, add before line 174 in that file, the following:
        gain = gain.to(torch.int64)
"""

import requests
import numpy as np
from PIL import Image
from io import BytesIO
import torch

from art.estimators.object_detection.pytorch_yolo import PyTorchYolo
from art.attacks.evasion import ProjectedGradientDescent

import cv2
import matplotlib
import matplotlib.pyplot as plt


"""
#################        Helper functions and labels          #################
"""

COCO_INSTANCE_CATEGORY_NAMES = [
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
    "backpack",
    "umbrella",
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
    "dining table",
    "toilet",
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
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def extract_predictions(predictions_, conf_thresh):
    # Get the predicted class
    predictions_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(predictions_["labels"])]
    #  print("\npredicted classes:", predictions_class)
    if len(predictions_class) < 1:
        return [], [], []
    # Get the predicted bounding boxes
    predictions_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions_["boxes"])]

    # Get the predicted prediction score
    predictions_score = list(predictions_["scores"])
    # print("predicted score:", predictions_score)

    # Get a list of index with score greater than threshold
    threshold = conf_thresh
    predictions_t = [predictions_score.index(x) for x in predictions_score if x > threshold]
    if len(predictions_t) == 0:
        return [], [], []

    # predictions in score order
    predictions_boxes = [predictions_boxes[i] for i in predictions_t]
    predictions_class = [predictions_class[i] for i in predictions_t]
    predictions_scores = [predictions_score[i] for i in predictions_t]
    return predictions_class, predictions_boxes, predictions_scores


def plot_image_with_boxes(img, boxes, pred_cls, title):
    plt.style.use("ggplot")
    text_size = 1
    text_th = 3
    rect_th = 1

    for i in range(len(boxes)):
        cv2.rectangle(
            img,
            (int(boxes[i][0][0]), int(boxes[i][0][1])),
            (int(boxes[i][1][0]), int(boxes[i][1][1])),
            color=(0, 255, 0),
            thickness=rect_th,
        )
        # Write the prediction class
        cv2.putText(
            img,
            pred_cls[i],
            (int(boxes[i][0][0]), int(boxes[i][0][1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            (0, 255, 0),
            thickness=text_th,
        )

    plt.figure()
    plt.axis("off")
    plt.title(title)
    plt.imshow(img.astype(np.uint8), interpolation="nearest")
    plt.show()


"""
#################        Evasion settings        #################
"""
eps = 32
eps_step = 2
max_iter = 10


"""
#################        Model definition        #################
"""
MODEL = "yolov3"  # OR yolov5


if MODEL == "yolov3":

    from pytorchyolo.utils.loss import compute_loss
    from pytorchyolo.models import load_model

    class Yolo(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x, targets=None):
            if self.training:
                outputs = self.model(x)
                loss, loss_components = compute_loss(outputs, targets, self.model)
                loss_components_dict = {"loss_total": loss}
                return loss_components_dict
            else:
                return self.model(x)

    model_path = "./yolov3.cfg"
    weights_path = "./yolov3.weights"
    model = load_model(model_path=model_path, weights_path=weights_path)

    model = Yolo(model)

    detector = PyTorchYolo(
        model=model, device_type="cpu", input_shape=(3, 640, 640), clip_values=(0, 255), attack_losses=("loss_total",)
    )

elif MODEL == "yolov5":

    import yolov5
    from yolov5.utils.loss import ComputeLoss

    matplotlib.use("TkAgg")

    class Yolo(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.model.hyp = {
                "box": 0.05,
                "obj": 1.0,
                "cls": 0.5,
                "anchor_t": 4.0,
                "cls_pw": 1.0,
                "obj_pw": 1.0,
                "fl_gamma": 0.0,
            }
            self.compute_loss = ComputeLoss(self.model.model.model)

        def forward(self, x, targets=None):
            if self.training:
                outputs = self.model.model.model(x)
                loss, loss_items = self.compute_loss(outputs, targets)
                loss_components_dict = {"loss_total": loss}
                return loss_components_dict
            else:
                return self.model(x)

    model = yolov5.load("yolov5s.pt")

    model = Yolo(model)

    detector = PyTorchYolo(
        model=model, device_type="cpu", input_shape=(3, 640, 640), clip_values=(0, 255), attack_losses=("loss_total",)
    )


"""
#################        Example image        #################
"""
response = requests.get("https://ultralytics.com/images/zidane.jpg")
img = np.asarray(Image.open(BytesIO(response.content)).resize((640, 640)))
img_reshape = img.transpose((2, 0, 1))
image = np.stack([img_reshape], axis=0).astype(np.float32)
x = image.copy()

"""
#################        Evasion attack        #################
"""

attack = ProjectedGradientDescent(estimator=detector, eps=eps, eps_step=eps_step, max_iter=max_iter)
image_adv = attack.generate(x=x, y=None)

print("\nThe attack budget eps is {}".format(eps))
print("The resulting maximal difference in pixel values is {}.".format(np.amax(np.abs(x - image_adv))))

plt.axis("off")
plt.title("adversarial image")
plt.imshow(image_adv[0].transpose(1, 2, 0).astype(np.uint8), interpolation="nearest")
plt.show()

threshold = 0.85  # 0.5
dets = detector.predict(x)
preds = extract_predictions(dets[0], threshold)
plot_image_with_boxes(img=img, boxes=preds[1], pred_cls=preds[0], title="Predictions on original image")

dets = detector.predict(image_adv)
preds = extract_predictions(dets[0], threshold)
plot_image_with_boxes(
    img=image_adv[0].transpose(1, 2, 0).copy(),
    boxes=preds[1],
    pred_cls=preds[0],
    title="Predictions on adversarial image",
)
