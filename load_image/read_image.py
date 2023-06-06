from copy import copy

import cv2
import numpy as np
from loguru import logger


class ReadImage:
    _path: str
    _image: np.ndarray

    def __init__(self, path: str):
        self._path = path

    def load_image(self):
        logger.info(f'Loading image: {self._path}')
        color_image = cv2.imread(self._path, cv2.IMREAD_COLOR)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        self._image = gray_image
        logger.info(f'Image size: {self._image.shape}')

    def show_image(self):
        cv2.imshow('Image', self._image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_image(self):
        return copy(self._image)
