# import logging
#

import cv2
import numpy as np
import torch

from image_normalization.noise_filtering import NoiseFiltering
from models.u_net_lung import UNetLung
from models.u_net_rib import UNetRib
from rib_suppression import rib_suppression
import lung_segmentation.utils as lg
from utils import transforms as tr

# from image_normalization.noise_filtering import NoiseFiltering
#
# read = ReadImage('datasets/rib_shadows/train_images.tif')
# read.load_image()
# read.show_image()
# filter = NoiseFiltering('fluorogram.png')
# filter()


test = 'datasets/pathologies_detection/images/MCUCXR_0001_0.png'
path_to_trained_lung = 'trained_models/u_net_lung_segmentation.pth'
path_to_trained_rib = 'trained_models/u_net_rib_shadows.pth'
# training_config = {
#     'dataset_path': 'datasets/rib_shadows',
#     'path_to_trained_net': path_to_trained_net,
#     'epoch_num': 10
# }
# training = TrainingNet(training_config=training_config)
# # training(is_visualize=True)
#
#
image = cv2.imread(test, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (512, 512))
noise_filter = NoiseFiltering()
image = noise_filter.change(image)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

net_lung = UNetLung()
net_rib = UNetRib()
net_lung.load_state_dict(torch.load(path_to_trained_lung))
net_rib.load_state_dict(torch.load(path_to_trained_rib))

image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).type(torch.float32).requires_grad_()

lung_mask = net_lung(torch.clone(image_tensor))
rib_mask = net_rib(torch.clone(image_tensor))

lung_mask = tr.to_binary_np(lung_mask)
rib_mask = tr.to_binary_np(rib_mask)
lung_mask = lg.delete_extra_connected_areas(lung_mask)

lung_image = image * lung_mask

# cv2.imshow('lung', lung_mask * 255)
# cv2.imshow('rib', rib_mask * 255)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# net_lung.load_state_dict(torch.
# image = cv2.resize(cv2.imread(im, cv2.IMREAD_GRAYSCALE), (512, 512))
# mask_lung = cv2.resize(cv2.imread(mask_lung, cv2.IMREAD_GRAYSCALE), (512, 512))

rib_suppression.remove_bones(image, rib_mask, lung_mask)

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
#
# import torch as t
# from pathologies_detection.utils.config import opt
# from models.faster_rcnn import FasterRCNNVGG16
# from pathologies_detection.trainer import FasterRCNNTrainer
# from pathologies_detection.data.util import read_image
# from pathologies_detection.utils.vis_tool import vis_bbox
# from pathologies_detection.utils import array_tool as at
#
# # %matplotlib inline
#
# img = read_image('datasets/pathologies_detection/images/MCUCXR_0001_0.png', dtype=np.uint8, color=False)
# img = t.from_numpy(img)[None]
# faster_rcnn = FasterRCNNVGG16()
# trainer = FasterRCNNTrainer(faster_rcnn)
#
# trainer.load('trained_models/fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth')
# opt.caffe_pretrain = False  # this model was trained from torchvision-pretrained model
# _bboxes, _labels, _scores = trainer.faster_rcnn.predict(img, visualize=True)
# vis_bbox(at.tonumpy(img[0]),
#          at.tonumpy(_bboxes[0]),
#          at.tonumpy(_labels[0]).reshape(-1),
#          at.tonumpy(_scores[0]).reshape(-1))
