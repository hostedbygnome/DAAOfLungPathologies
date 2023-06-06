import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from pathologies_detection.pathologies_names import PATHOLOGIES_INSTANCE_CATEGORY_NAMES as pathologies_names

np.random.seed(42)
COLORS = np.random.uniform(0, 255, size=(len(pathologies_names), 3))

transform = transforms.Compose([
    transforms.ToTensor(),
])


def predict(image, model, detection_threshold):
    image = transform(image)
    image.unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)

    pred_scores = outputs[0]['scores'].detach().cpu().numpy()

    pred_boxes = outputs[0]['boxes'].detach().cpu().numpy()
    boxes = pred_boxes[pred_scores >= detection_threshold].astype(np.int32)
    labels = outputs[0]['labels'][:len(boxes)]
    pred_classes = [pathologies_names[i] for i in labels.cpu().numpy()]

    return boxes, pred_classes, labels


def draw_boxes(boxes, classes, labels, image):
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    tf = max(lw - 1, 1)
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            img=image,
            pt1=(int(box[0]), int(box[1])),
            pt2=(int(box[2]), int(box[3])),
            color=color[::-1],
            thickness=lw
        )
        cv2.putText(
            img=image,
            text=classes[i],
            org=(int(box[0]), int(box[1] - 5)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=lw / 3,
            color=color[::-1],
            thickness=tf,
            lineType=cv2.LINE_AA
        )
    return image
