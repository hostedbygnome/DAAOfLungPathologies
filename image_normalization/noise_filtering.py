import cv2
import numpy as np
from loguru import logger


class NoiseFiltering:
    _image: np.ndarray
    _filtered_image: np.ndarray
    _path: str

    # def __init__(self, path: str):
    #     self._path = path
    # def __init__(self, image):
    #     self._image = image

    def __call__(self, *args, **kwargs):
        parsed_path = self._path.split('.')
        logger.info(f'Image: {self._path} filter')
        self._image = cv2.imread(self._path, cv2.IMREAD_GRAYSCALE)
        logger.info(f'Image size noise filter: {self._image.shape}')
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        kernel2 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel3 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # self._filtered_image = cv2.GaussianBlur(self._image, (5, 5), 0, cv2.BORDER_DEFAULT)

        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
        self._filtered_image = clahe.apply(self._image)
        self._filtered_image = cv2.bilateralFilter(src=self._filtered_image, d=15, sigmaColor=66, sigmaSpace=32,
                                                   borderType=cv2.BORDER_DEFAULT)
        log_gabor_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # Пример фильтра Габора
        self._filtered_image = cv2.filter2D(self._filtered_image, -1, log_gabor_filter)
        # self._filtered_image = cv2.Canny(self._filtered_image, 100, 200)

        # sobelx = cv2.Scharr(src=self._filtered_image, ddepth=cv2.CV_64F, dx=1, dy=0, scale=cv2.FILTER_SCHARR,
        #                     borderType=cv2.BORDER_DEFAULT)
        # sobely = cv2.Scharr(src=self._filtered_image, ddepth=cv2.CV_64F, dx=0, dy=1, scale=cv2.FILTER_SCHARR,
        #                     borderType=cv2.BORDER_DEFAULT)
        # magnitude = cv2.magnitude(sobelx, sobely)
        # self._filtered_image = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        logger.info(f'Save filtered image path: {parsed_path[0]}_filtered.{parsed_path[1]}')
        cv2.imwrite(f'{parsed_path[0]}_filtered.{parsed_path[1]}', self._filtered_image)

    def change(self, image):
        clahe = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(8, 4))
        self._filtered_image = clahe.apply(image)
        self._filtered_image = cv2.bilateralFilter(src=self._filtered_image, d=15, sigmaColor=21, sigmaSpace=41,
                                                   borderType=cv2.BORDER_DEFAULT)
        # self._filtered_image = cv2.Canny(self._filtered_image, 100, 200)

        # log_gabor_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # Пример фильтра Габора
        # self._filtered_image = cv2.filter2D(self._filtered_image, -1, log_gabor_filter)
        return self._filtered_image
