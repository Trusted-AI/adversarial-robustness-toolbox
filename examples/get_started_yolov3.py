"""
The script demonstrates a simple example of using ART with Yolov3. The example loads a model pretrained on the COCO dataset
and creates an adversarial example using Projected Gradient Descent method.

*** Note ***
Requires modifying pytorchyolo/utils/loss.py, before line 174, add the following:
gain = gain.to(torch.int64)
"""
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import torch
import pandas as pd

from pytorchyolo.utils.loss import compute_loss
from pytorchyolo.models import load_model

from art.estimators.object_detection.pytorch_yolo import PyTorchYolo
from art.attacks.evasion import ProjectedGradientDescent

import cv2
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('ggplot')
matplotlib.use( 'tkagg' )

COCO_CATEGORIES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
        'teddy bear', 'hair drier', 'toothbrush']

def extract_predictions(predictions_):
    # Get the predicted class
    predictions_class = [COCO_CATEGORIES[i] for i in list(predictions_["labels"])]

    # Get the predicted bounding boxes
    predictions_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions_["boxes"])]

    # Get the predicted prediction score
    predictions_score = list(predictions_["scores"])

    predictions_boxes = [[tuple([int (i) for i in t]) for t in r] for r in predictions_boxes]

    return predictions_class, predictions_boxes, predictions_class


def plot_image_with_boxes(img, boxes, pred_cls):
    text_size = 1
    text_th = 3
    rect_th = 3
    
    for i in range(len(boxes)):
        
        color = tuple([int(i) for i in list(np.random.choice(range(256), size=3))])
        
        # Draw Rectangle with the coordinates
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=color, thickness=rect_th)

        # Write the prediction class
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, color, thickness=text_th)

    plt.axis("off")
    plt.imshow(img.astype(np.uint8), interpolation="nearest")
    plt.show()
    
class YoloV3(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, targets=None):
        if self.training:
            outputs = self.model(x)
            # loss is averaged over a batch. Thus, for patch generation use batch_size = 1
            loss, loss_components = compute_loss(outputs, targets, self.model)
            loss_components_dict = {"loss_total": loss}
            return loss_components_dict
        else:
            return self.model(x)
    
    
response = requests.get('https://ultralytics.com/images/zidane.jpg')
image = np.asarray(Image.open(BytesIO(response.content)).resize((640, 640)))
img_reshape = image.transpose(2,0,1)
img = np.stack([img_reshape], axis=0).astype(np.float32) 
x = img.copy()

model_path = "./yolov3.cfg"
weights_path = "./yolov3.weights"
model = load_model(model_path=model_path, weights_path=weights_path)
        
model = YoloV3(model)

detector = PyTorchYolo(model=model,
                       device_type='cpu',
                       input_shape=(3, 640, 640),
                       clip_values=(0, 255), 
                       attack_losses=("loss_total",))

predictions_orig = detector.predict(x=x)[0]

eps = 32
attack = ProjectedGradientDescent(estimator=detector, eps=eps, eps_step=2, max_iter=10)
image_adv = attack.generate(x=x, y=None)

print("\nThe attack budget eps is {}".format(eps))
print("The resulting maximal difference in pixel values is {}.".format(np.amax(np.abs(x - image_adv))))

plt.axis("off")
plt.title("adversarial image")
plt.imshow(image_adv[0].transpose(1,2,0).astype(np.uint8), interpolation="nearest")
plt.show()


print('Detector sent to attack on original image')
predictions_orig = detector.predict(x=x)[0]
threshold = 0.5
scores = pd.DataFrame(predictions_orig['scores'], columns=['scores']) 
labels = pd.DataFrame(predictions_orig['labels'], columns=['labels'])
df = pd.concat([scores, labels], axis=1)
df = df[df.scores>threshold]
df = df.sort_values('scores', ascending=False).reset_index()
print('Yolov3 predictions for original image:')
print(df[['scores', 'labels']].head(10))

n_boxes = []
for i, row in df.iterrows():
    n_boxes.append(predictions_orig['boxes'][int(row['index'])].astype(np.int32))

n_predictions = {'boxes': np.array(n_boxes), 
                 'scores': df.scores.values, 
                 'labels': df.labels.values}

# Process predictions
predictions_orig_class, predictions_orig_boxes, predictions_orig_class = extract_predictions(n_predictions)

plt.title("Yolov3 predictions (original image)")
# Plot predictions
plot_image_with_boxes(img=image.copy(), boxes=predictions_orig_boxes, pred_cls=predictions_orig_class)

print('--------')


model_path = "./yolov3.cfg"
weights_path = "./yolov3.weights"
model = load_model(model_path=model_path, weights_path=weights_path)
        
model = YoloV3(model)

detector = PyTorchYolo(model=model,
                       device_type='cpu',
                       input_shape=(3, 640, 640),
                       clip_values=(0, 255), 
                       attack_losses=("loss_total",))
                       
predictions_adv = detector.predict(x=image_adv)[0]

threshold = 0.99
scores = pd.DataFrame(predictions_orig['scores'], columns=['scores']) 
labels = pd.DataFrame(predictions_orig['labels'], columns=['labels'])
df = pd.concat([scores, labels], axis=1)
df = df[df.scores>threshold]
df = df.sort_values('scores', ascending=False).reset_index()
print('-------')
print('Yolov3 predictions for original image:')
print(df[['scores', 'labels']].head(10))

n_boxes = []
for i, row in df.iterrows():
    n_boxes.append(predictions_orig['boxes'][int(row['index'])].astype(np.int32))

n_predictions = {'boxes': np.array(n_boxes), 
                 'scores': df.scores.values, 
                 'labels': df.labels.values}

# Process predictions
predictions_orig_class, predictions_orig_boxes, predictions_orig_class = extract_predictions(n_predictions)

plt.title("Yolov3 predictions (original image)")
# Plot predictions
plot_image_with_boxes(img=image.copy(), boxes=predictions_orig_boxes, pred_cls=predictions_orig_class)


print('-------')

scores = pd.DataFrame(predictions_adv['scores'], columns=['scores']) 
labels = pd.DataFrame(predictions_adv['labels'], columns=['labels'])
df = pd.concat([scores, labels], axis=1)
df = df[df.scores>threshold]
df = df.sort_values('scores', ascending=False).reset_index()
print('Yolov3 predictions for adversarial image:')
print(df[['scores', 'labels']].head(10))

n_boxes = []
for i, row in df.iterrows():
    n_boxes.append(predictions_adv['boxes'][int(row['index'])].astype(np.int32))

n_predictions = {'boxes': np.array(n_boxes), 
                 'scores': df.scores.values, 
                 'labels': df.labels.values}

# Process predictions
predictions_adv_class, predictions_adv_boxes, predictions_adv_class = extract_predictions(n_predictions)

plt.title("Yolov3 predictions (adversarial image)")
# Plot predictions
plot_image_with_boxes(img=image.copy(), boxes=predictions_adv_boxes, pred_cls=predictions_adv_class)