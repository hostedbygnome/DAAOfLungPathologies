import cv2
import numpy as np
import torch


def to_binary_np(mask: torch.Tensor):
    mask = (mask > 0.5).type(torch.uint8)
    while len(mask.shape) > 2:
        mask = mask.squeeze()
    return mask.detach().numpy().astype(np.uint8)


def delete_left_lung(lung_mask: np.array):
    lung_mask = lung_mask.astype(np.uint8)

    # You need to choose 4 or 8 for connectivity type
    connectivity = 4
    # Perform the operation
    output = cv2.connectedComponentsWithStats(lung_mask, connectivity, cv2.CV_32S)

    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]

    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]

    labels = np.array(labels)
    row_labels = np.reshape(labels, np.prod(labels.shape))
    row_img = np.reshape(lung_mask, np.prod(lung_mask.shape))

    centroids = np.array(centroids)
    centroids = centroids[1:]

    left_lung_label = 2
    if centroids[0][1] > centroids[1][1]:
        left_lung_label = 1

    left_lung_indexes = np.where(row_labels == left_lung_label)[0]
    row_img[left_lung_indexes] = 0

    lung_mask = np.reshape(row_img, lung_mask.shape)

    return lung_mask


def encoder(input_image: torch.tensor):
    sh = input_image.shape
    return torch.reshape(input_image, [sh[0], sh[1], sh[2] * sh[3]])


def decoder(input_image: torch.tensor):
    sh = input_image.shape
    return torch.reshape(input_image, [sh[0], sh[1], int(np.sqrt(sh[2])), int(np.sqrt(sh[2]))])
