# import logging
#

import cv2
import numpy as np
import torch
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QMessageBox, QHBoxLayout, QDialog
from PyQt6.QtWidgets import QPushButton, QFileDialog, QLabel

import lung_segmentation.utils.utils as lg
from image_normalization.noise_filtering import NoiseFiltering
from models.seg_net import SegNet
from models.u_net_lung import UNetLung
from models.u_net_rib import UNetRib
from rib_suppression import rib_suppression
from utils import transforms as tr
from utils.transforms import delete_left_lung, encoder, decoder


# from image_normalization.noise_filtering import NoiseFiltering
#
# read = ReadImage('datasets/rib_shadows/train_images.tif')
# read.load_image()
# read.show_image()
# filter = NoiseFiltering('fluorogram.png')
# filter()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.process_result = None
        self.pre_process_result = None
        self.lung_mask = None
        self.resized_image = None
        self.init_UI()

        self.selected_file = None  # Переменная для хранения пути выбранного файла

    def init_UI(self):
        self.setWindowTitle("DAAOfLungPathologies")
        self.setGeometry(300, 300, 1536, 600)

        select_button = QPushButton("Выбрать файл", self)
        select_button.setFixedSize(512, 50)
        select_button.clicked.connect(self.open_file_dialog)

        pre_process_button = QPushButton("Выполнить предобработку", self)
        pre_process_button.setFixedSize(512, 50)
        pre_process_button.clicked.connect(self.pre_process_image)

        process_button = QPushButton("Выполнить обработку", self)
        process_button.setFixedSize(512, 50)
        process_button.clicked.connect(self.process_image)

        analyze_button = QPushButton("Выполнить анализ", self)
        analyze_button.setFixedSize(512, 50)
        analyze_button.clicked.connect(self.analyze_image)

        self.selected_image_label = QLabel(self)
        self.selected_image_label.setFixedSize(512, 512)
        self.selected_image_label.setText("Выбранное изображение")

        self.pre_processed_image_label = QLabel(self)
        self.pre_processed_image_label.setFixedSize(512, 512)
        self.pre_processed_image_label.setText("Предобработанное изображение")

        self.processed_image_label = QLabel(self)
        self.processed_image_label.setFixedSize(512, 512)
        self.processed_image_label.setText("Обработанное изображение")

        # Создаем вертикальный контейнер и добавляем в него кнопки и метки
        text_layout = QHBoxLayout()
        text_layout.addWidget(select_button)
        text_layout.addWidget(pre_process_button)
        text_layout.addWidget(process_button)

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.selected_image_label)
        image_layout.addWidget(self.pre_processed_image_label)
        image_layout.addWidget(self.processed_image_label)

        analyze_layout = QHBoxLayout()
        analyze_layout.addWidget(analyze_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(text_layout)
        main_layout.addLayout(image_layout)
        main_layout.addLayout(analyze_layout)

        # Создаем виджет-контейнер и устанавливаем в него вертикальный контейнер
        widget = QWidget(self)
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

    def open_file_dialog(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Выберите файл", "", "All Files (*)")

        if file_path:
            self.selected_file = file_path
            print("Выбран файл:", self.selected_file)
            self.display_selected_file()

    def display_selected_file(self):
        try:
            image = cv2.imread(self.selected_file, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise Exception("Файл не является изображением")
        except Exception as e:
            self.show_error_message(str(e))
            return

        self.resized_image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        selected_pixmap = convert_image_to_pixmap(self.resized_image)
        self.selected_image_label.setPixmap(selected_pixmap)

    def show_error_message(self, message):
        QMessageBox.critical(self, "Ошибка", message)

    def pre_process_image(self):
        if self.selected_file:
            # Здесь вы можете добавить свой код для обработки выбранного файла
            # Вместо простого вывода результата в консоль, мы устанавливаем обработанное изображение в QLabel
            binary_mask_segments = self.resized_image
            path_to_trained_lung = 'trained_models/u_net_lung_segmentation.pth'
            path_to_trained_rib = 'trained_models/u_net_rib_shadows.pth'

            # image = cv2.imread(test, cv2.IMREAD_GRAYSCALE)
            # image = cv2.resize(image, (512, 512))
            noise_filter = NoiseFiltering()
            image = noise_filter.change(binary_mask_segments)

            net_lung = UNetLung()
            net_rib = UNetRib()
            net_lung.load_state_dict(torch.load(path_to_trained_lung))
            net_rib.load_state_dict(torch.load(path_to_trained_rib))

            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).type(torch.float32).requires_grad_()

            self.lung_mask = net_lung(torch.clone(image_tensor))
            rib_mask = net_rib(torch.clone(image_tensor))

            self.lung_mask = tr.to_binary_np(self.lung_mask)
            rib_mask = tr.to_binary_np(rib_mask)
            self.lung_mask = lg.delete_extra_connected_areas(self.lung_mask)

            lung_image = image * self.lung_mask

            self.pre_process_result = rib_suppression.remove_bones(image, rib_mask, self.lung_mask)
            # Здесь вы можете добавить свой код для обработки выбранного файла
            # Вместо простого вывода результата в консоль, мы обновляем метку QLabel в окне
            # result = f"Результат обработки файла: {self.selected_file}"

            # result_bytes = np.bytes(result)

            pre_processed_pixmap = convert_image_to_pixmap(self.pre_process_result)
            self.pre_processed_image_label.setPixmap(pre_processed_pixmap)

            # Предполагая, что ваш код обработки изображения возвращает объект QPixmap с именем processed_image

    def process_image(self):
        if self.selected_file:
            path_to_trained_segments = 'trained_models/segments.pth'
            net_segments = SegNet()

            net_segments.load_state_dict(torch.load(path_to_trained_segments))

            right_lung = self.lung_mask

            right_lung = delete_left_lung(right_lung)
            right_lung = cv2.resize(right_lung, (128, 128))
            right_lung = right_lung.astype('float32')
            right_lung = torch.from_numpy(right_lung)
            right_lung = torch.reshape(right_lung, (1, 1, 128, 128))
            right_lung = encoder(right_lung)

            test_preds = net_segments(right_lung)
            test_preds = decoder(test_preds)
            binary_mask_segments = test_preds[0][0]
            binary_mask_segments = torch.where(binary_mask_segments > 0.5, 1, 0)
            binary_mask_segments = binary_mask_segments.detach().numpy().astype(np.uint8)

            kernel = np.ones((5, 5), np.uint8)
            binary_mask_segments = cv2.erode(binary_mask_segments, kernel, iterations=1)
            binary_mask_segments = cv2.dilate(binary_mask_segments, kernel, iterations=1)
            binary_mask_segments = cv2.resize(binary_mask_segments, (512, 512))
            # Пороговая обработка
            ret, binary_mask_segments = cv2.threshold(binary_mask_segments.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(binary_mask_segments, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # img = img * binary_mask_segments
            self.process_result = np.copy(self.pre_process_result)
            cv2.drawContours(self.process_result, contours, -1, (255, 255, 0), 2)

            processed_pixmap = convert_image_to_pixmap(self.process_result)
            self.processed_image_label.setPixmap(processed_pixmap)

    def analyze_image(self):
        analyze = AnalysisDialog(self, img=self.pre_process_result)
        analyze.exec()


class AnalysisDialog(QDialog):
    def __init__(self, parent=None, img=None):
        super().__init__(parent)
        self.setWindowTitle("Окно анализа")
        self.setGeometry(200, 200, 600, 600)

        analyze_image_label = QLabel(self)
        analyze_image_label.setFixedSize(512, 512)
        analyze_image_label.setText("Результат анализа")
        cv2.imwrite('test.png', img)
        # analyze_pixmap = QPixmap("path_to_image")  # Замените "path_to_image" на путь к вашему изображению
        analyze_image_label.setPixmap(convert_image_to_pixmap(img))

        layout = QVBoxLayout()
        layout.addWidget(analyze_image_label)
        self.setLayout(layout)


def convert_image_to_pixmap(image):
    height, width = image.shape
    bytes_per_line = width
    qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
    pixmap = QPixmap.fromImage(qimage)
    return pixmap


app = QApplication([])
window = MainWindow()
window.show()
app.exec()
# training_config = {
#     'dataset_path': 'datasets/rib_shadows',
#     'path_to_trained_net': path_to_trained_net,
#     'epoch_num': 10
# }
# training = TrainingNet(training_config=training_config)
# # training(is_visualize=True)
#
#


# cv2.imshow('lung', lung_mask * 255)
# cv2.imshow('rib', rib_mask * 255)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


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
