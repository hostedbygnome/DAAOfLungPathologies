import cv2
import numpy as np


def delete_extra_connected_areas_one_pass(img: np.array, changed_value):
    right_lung_point = [img.shape[0] // 2, img.shape[0] // 4]
    left_lung_point = [img.shape[0] // 2, img.shape[0] // 4 * 3]

    img = img.astype(np.uint8)

    # You need to choose 4 or 8 for connectivity type
    connectivity = 4
    # Perform the operation
    output = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)

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
    row_img = np.reshape(img, np.prod(img.shape))

    spaces = np.array(stats[:, 4:])
    spaces = np.reshape(spaces, len(spaces))
    nums = np.arange(num_labels)
    indexed_spaces = np.zeros((2, num_labels), 'uint32')
    indexed_spaces[0] = nums
    indexed_spaces[1] = spaces
    indexed_spaces = np.moveaxis(indexed_spaces, 1, 0)
    indexed_spaces = indexed_spaces[indexed_spaces[:, 1].argsort()]

    lung_labels = indexed_spaces[num_labels - 2][0], indexed_spaces[num_labels - 3][0]
    fone = indexed_spaces[num_labels - 1][0]

    not_lung_indexes = np.where((row_labels != lung_labels[0]) & (row_labels != lung_labels[1]) & (row_labels != fone))[
        0]
    row_img[not_lung_indexes] = changed_value

    img = np.reshape(row_img, img.shape)

    # try with openCV
    return img


def delete_extra_connected_areas(img: np.array):
    img = delete_extra_connected_areas_one_pass(img, 0)
    image, contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    lung = image[0]
    lung = np.reshape(lung, (lung.shape[0], lung.shape[2]))

    cv2.fillPoly(img, pts=[lung], color=(1))

    lung = image[1]
    lung = np.reshape(lung, (lung.shape[0], lung.shape[2]))

    cv2.fillPoly(img, pts=[lung], color=(1))

    return img

