import cv2
import matplotlib.pyplot as plt

from art.estimators.object_detectors.PyTorchFasterRCNN import PyTorchFasterRCNN


def main():
    frcnn = PyTorchFasterRCNN()

    # For training
    # images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
    # labels = torch.randint(1, 91, (4, 11))
    # images = list(image for image in images)
    # targets = []
    # for i in range(len(images)):
    #     d = {}
    #     d['boxes'] = boxes[i]
    #     d['labels'] = labels[i]
    #     targets.append(d)
    # output = model(images, targets)

    # i_img = 0

    # image = cv2.imread("./banner-diverse-group-of-people-2.jpg")
    image = cv2.imread('./10best-cars-group-cropped-1542126037.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # transform = torchvision.transforms.Compose(
    #     [torchvision.transforms.ToTensor(),
    #      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # image = transform(image)
    # x = [image]
    # predictions = model(x)

    predictions = frcnn.predict(x=image)

    for key, item in predictions[0].items():
        print(key, item)

    COCO_INSTANCE_CATEGORY_NAMES = [
        "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag",
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", ]

    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in
                  list(predictions[0]["labels"].numpy())]  # Get the Prediction Score
    print(pred_class)
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in
                  list(predictions[0]["boxes"].detach().numpy())]  # Bounding boxes
    pred_score = list(predictions[0]["scores"].detach().numpy())

    threshold = 0.5
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][
        -1]  # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[: pred_t + 1]
    pred_class = pred_class[: pred_t + 1]

    # image = cv2.imread("./banner-diverse-group-of-people-2.jpg")
    # image = cv2.imread('./10best-cars-group-cropped-1542126037.jpg')
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = image

    boxes = pred_boxes
    pred_cls = pred_class
    text_size = 5
    text_th = 5
    rect_th = 6

    for i in range(len(boxes)):
        print(i)
        cv2.rectangle(
            img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th
        )  # Draw Rectangle with the coordinates
        cv2.putText(
            img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th
        )  # Write the prediction class

    plt.axis("off")
    plt.imshow(img, interpolation="nearest")
    plt.show()


if __name__ == "__main__":
    main()
