# import logging
#
from urllib.parse import urlparse

import cv2
from bs4 import BeautifulSoup

import numpy as np
import requests
import torch
from torchvision.transforms import transforms

from rib_suppression import rib_suppression
from rib_suppression.train import TrainingNet
from models.u_net import UNet

# from image_normalization.noise_filtering import NoiseFiltering
#
# read = ReadImage('datasets/rib_shadows/train_images.tif')
# read.load_image()
# read.show_image()
# filter = NoiseFiltering('fluorogram.png')
# filter()

# path_to_trained_net = 'trained_models/u_net_rib_shadows.pth'
# training_config = {
#     'dataset_path': 'datasets/rib_shadows',
#     'path_to_trained_net': path_to_trained_net,
#     'epoch_num': 10
# }
# training = TrainingNet(training_config=training_config)
# # training(is_visualize=True)
#
#
# im = 'not_segmented_8.png'
# mask_lung = 'segmented_8.png'
# # noise_filter = NoiseFiltering()
# image = cv2.resize(cv2.imread(im, cv2.IMREAD_GRAYSCALE), (512, 512))
# mask_lung = cv2.resize(cv2.imread(mask_lung, cv2.IMREAD_GRAYSCALE), (512, 512))
#
# image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).type(torch.float32).requires_grad_()
#
# net = UNet()
# net.load_state_dict(torch.load(path_to_trained_net))
# mask = net(torch.clone(image))
# rib_suppression.remove_bones(image, mask, mask_lung)


# image_url = 'https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/CXR_png/index.html'
# left_url = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/ManualMask/leftMask/index.html"
# right_url = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/ManualMask/rightMask/index.html"
# # Загрузка содержимого страницы
# response_image = requests.get(image_url)
# responseLeft = requests.get(left_url)
# responseRight = requests.get(right_url)
# html_content_image = response_image.content
# html_content_left = responseLeft.content
# html_content_right = responseRight.content
#
# # Создание объекта BeautifulSoup для парсинга HTML
# soup_image = BeautifulSoup(html_content_image, 'html.parser')
# soup_left = BeautifulSoup(html_content_left, "html.parser")
# soup_right = BeautifulSoup(html_content_right, "html.parser")
# # Находим все теги <img> на странице
# img_tags_image = soup_image.find_all("a")
# img_tags_left = soup_left.find_all("a")
# img_tags_right = soup_right.find_all("a")
#
# # Создание списка для хранения Tensor-представлений изображений
# tensor_images = []
# image_url = '/'.join(image_url.split('/')[:-1]) + '/'
# image_url_left = '/'.join(left_url.split('/')[:-1]) + '/'
# image_url_right = '/'.join(right_url.split('/')[:-1]) + '/'
# # Проход по всем тегам <img> и загрузка изображений®
# for img_tag in img_tags_image:
#     # Получение URL изображения
#     img_url = image_url + img_tag["href"]
#     img_url_left = image_url_left + img_tag['href']
#     img_url_right = image_url_right + img_tag['href']
#     pathology = img_tag['href']
#     # Полное URL изображения, если оно указано относительно
#     # parsed_url = urlparse(img_url)
#     # if parsed_url.scheme == '':
#     #     img_url = f"{parsed_url.netloc}{img_url}"
#
#     # Загрузка изображения с URL
#     img_response = requests.get(img_url)
#     img_response_left = requests.get(img_url_left)
#     img_response_right = requests.get(img_url_right)
#
#     img_np = np.frombuffer(img_response.content, np.uint8)
#     img_np_left = np.frombuffer(img_response_left.content, np.uint8)
#     img_np_right = np.frombuffer(img_response_right.content, np.uint8)
#
#     img = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
#     img_left = cv2.imdecode(img_np_left, cv2.IMREAD_GRAYSCALE)
#     img_right = cv2.imdecode(img_np_right, cv2.IMREAD_GRAYSCALE)
#     cv2.imwrite(f"datasets/pathologies_detection/images/{pathology}", img)
#     cv2.imwrite(f"datasets/pathologies_detection/masks/{pathology}", img_left + img_right)
#     # Преобразование изображения в формат Tensor
#     transform = transforms.ToTensor()
#     tensor_image = transform(img_left)
#     # Добавление Tensor-представления изображения в список
#     tensor_images.append(tensor_image)
#     print(img_tag['href'])
#     print(f"datasets/pathologies_detection/masks/{pathology}")
#     print(f"datasets/pathologies_detection/images/{pathology}")
#
# # Вывод количества загруженных изображений
# print(f"Загружено {len(tensor_images)} изображений.")

import os
import torch as t
from pathologies_detection.utils.config import opt
from models.faster_rcnn import FasterRCNNVGG16
from pathologies_detection.trainer import FasterRCNNTrainer
from pathologies_detection.data.util import  read_image
from pathologies_detection.utils.vis_tool import vis_bbox
from pathologies_detection.utils import array_tool as at
# %matplotlib inline

img = read_image('datasets/pathologies_detection/images/MCUCXR_0001_0.png', dtype=np.uint8, color=False)
img = t.from_numpy(img)[None]
faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cpu()

trainer.load('trained_models/chainer_best_model_converted_to_pytorch_0.7053.pth')
opt.caffe_pretrain=True # this model was trained from caffe-pretrained model
_bboxes, _labels, _scores = trainer.faster_rcnn.predict(img, visualize=True)
vis_bbox(at.tonumpy(img[0]),
         at.tonumpy(_bboxes[0]),
         at.tonumpy(_labels[0]).reshape(-1),
         at.tonumpy(_scores[0]).reshape(-1))