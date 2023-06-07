import math

import cv2
import numpy as np


def remove_bones(image, mask, lung_mask):
    mask_np = mask * image * lung_mask
    mask_np_inv = 1 - mask_np

    lung_image = image * lung_mask

    ret, binary_mask = cv2.threshold(mask_np.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)

    # Поиск контуров на маске
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_countours = list(filter(lambda c: cv2.contourArea(c) >= 1000, contours))

    out = np.copy(lung_image).astype(np.float64)
    contour_masks = []
    for contour in filtered_countours:
        # Создание пустой маски того же размера, что и исходное изображение
        contour_mask = np.zeros_like(binary_mask)
        # Нарисовать текущий контур на маске
        cv2.drawContours(contour_mask, [contour], 0, 255, cv2.FILLED)
        # Добавить маску контура в список
        contour_masks.append(np.clip(contour_mask, 0, 1))
    for i, contour_mask in enumerate(contour_masks):
        dilate_window_size = 10
        kernel_dilate_mask = np.ones((dilate_window_size, dilate_window_size), np.uint8)
        aroud_rib = (cv2.dilate(src=contour_mask, kernel=kernel_dilate_mask, iterations=1) - contour_mask) * lung_image
        erode_window_size = 8
        kernel_erode_mask = np.ones((erode_window_size, erode_window_size), np.uint8)
        rib = contour_mask * lung_image
        rib_mask = np.where(contour_mask * lung_image > 0, 1, 0)

        rib_mean_val = np.mean(rib[rib > 0])
        rib_around_mean_val = np.mean(aroud_rib[np.logical_and(0 < aroud_rib, aroud_rib < 240)])
        # print(rib_mean_val, rib_around_mean_val)
        nonzero_indexes = np.nonzero(rib_mask)
        rib_intensity = abs(rib_around_mean_val - rib_mean_val) * rib_around_mean_val / rib_mean_val
        y_max_mask = np.max(nonzero_indexes[0])
        y_min_mask = np.min(nonzero_indexes[0])
        x_max_mask = np.max(nonzero_indexes[1])
        x_min_mask = np.min(nonzero_indexes[1])
        points = np.sum(rib_mask)
        # print(points)
        # print(rib_intensity)
        count = 0
        for j in range(y_min_mask, y_max_mask + 1):
            for k in range(x_min_mask, x_max_mask + 1):
                if rib_mask[j, k] == 1:
                    count += 1
                    y_indexes_for_curr_x = nonzero_indexes[0][nonzero_indexes[1] == k]
                    y_max = np.max(y_indexes_for_curr_x)
                    y_min = np.min(y_indexes_for_curr_x)
                    rib_width = y_max - y_min
                    rib_center = y_min + rib_width // 2
                    localiz = abs(rib_center - j)
                    c = rib_width / 5
                    if localiz <= c:
                        out[j, k] = max(out[j, k] - rib_intensity, np.mean(aroud_rib))
                    else:
                        # print(math.sqrt((c - localiz) ** 2))
                        koef = math.sqrt((c - localiz) ** 2)
                        # print(localiz / c)
                        if localiz - c < 1:
                            koef = localiz / c

                        out[j, k] = max(out[j, k] - rib_intensity / koef, np.min(aroud_rib))
                        # print('new val = ', out[j, k])

                # if rib_without_border_mask[j, k] == 1:
                #     if (y_max - j) / (j - y_min)
        # out[rib_without_border_mask > 0] = np.maximum(mask_lung - rib_intensity, np.min(aroud_rib))[
        #     rib_without_border_mask > 0]
        # out[rib_border > 0] = np.maximum(np.mean(aroud_rib))
        # print(points == count)
        # cv2.imshow(f'Contour {i}', (rib_mask * 255).astype(np.uint8))
        # cv2.imshow(f'Rib {i}', rib)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imshow(f'Rf', out.astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    cv2.imshow('test1', lung_image)
    cv2.imshow('test2', out.astype(np.uint8))
    cv2.imshow('test', mask_np.astype(np.uint8) * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # erode_window_size = 4
    # dilate_window_size = 1
    # kernel_erode_mask = np.ones((erode_window_size, erode_window_size), np.uint8)
    # kernel_dilate_mask = np.ones((dilate_window_size, dilate_window_size), np.uint8)
    # erode_mask_np = cv2.erode(src=mask_np, kernel=kernel_erode_mask, iterations=1)
    # # dilate_mask_np = cv2.dilate(src=mask_np, kernel=kernel_dilate_mask, iterations=1)
    # # rib_borders = dilate_mask_np - erode_mask_np
    # # rib_borders_inv = 1 - rib_borders
    # mask_np_without_borders = mask_np * erode_mask_np
    # mask_np_only_borders = mask_np - erode_mask_np
    # mask_np_without_borders_inv = 1 - mask_np_without_borders
    #
    # image_np = image.squeeze().squeeze().detach().numpy().astype(np.uint8)
    # image_only_ribs = image_np * mask_np
    # image_without_ribs = image_np * mask_np_inv
    #
    # image_lung_only_ribs = lung_image * mask_np
    # image_lung_without_ribs = lung_image * mask_np_inv
    #
    # mean_val_lung_only_ribs = np.mean(image_lung_only_ribs[image_lung_only_ribs > 0])
    # mean_val_lung_without_ribs = np.mean(image_lung_without_ribs[image_lung_without_ribs > 0])
    # mean_val_rib = mean_val_lung_only_ribs - mean_val_lung_without_ribs
    # print(mean_val_rib)
    #
    # max_val_lung_only_ribs = np.max(image_lung_only_ribs)
    # max_val_lung_without_ribs = np.max(image_lung_without_ribs)
    # min_val_lung_only_ribs = np.min(image_lung_only_ribs)
    # min_val_lung_without_ribs = np.min(image_lung_without_ribs)
    # mean_val_max_rib = max_val_lung_only_ribs - max_val_lung_without_ribs
    # mean_val_min_rib = min_val_lung_only_ribs - min_val_lung_without_ribs
    #
    # out = np.copy(image_np)
    # window_size = 8
    # window_shape = (window_size, window_size)
    # for i in range(out.shape[0]):
    #     for j in range(out.shape[1]):
    #         if mask_np_without_borders[i, j] == 1:
    #             window = image_np[
    #                      max(i - window_size // 2, 0): min(i + window_size // 2 + 1, image_np.shape[0]),
    #                      max(j - window_size // 2, 0): min(j + window_size // 2 + 1, image_np.shape[1])]
    #             k = np.max(window) / np.min(window) / 4
    #             # k = np.median(window)
    #             # k = mean_val_rib
    #             std = np.std(window)
    #             median = np.median(window)
    #             # k = mean_val_rib + median / std
    #             # print("s %.5f" % std)
    #             # print("m %.5f" % median)
    #             out[i, j] = max((out[i, j] - mean_val_rib * k), min_val_lung_without_ribs)
    #             # out[i, j] *= k
    # window_size = 5
    # out2 = np.copy(out)
    # for i in range(out.shape[0]):
    #     for j in range(out.shape[1]):
    #         if mask_np_only_borders[i, j] == 1:
    #             window = out2[
    #                      max(i - window_size // 2, 0): min(i + window_size // 2 + 1, image_np.shape[0]),
    #                      max(j - window_size // 2, 0): min(j + window_size // 2 + 1, image_np.shape[1])]
    #             # k = out[i, j] / np.max(window)
    #             # k = np.median(window) - mean_val_rib
    #             # k = mean_val_rib
    #             # std = np.std(window)
    #             # print(out[i, j])
    #             median = np.median(window)
    #             # k = mean_val_rib + median / std
    #             # print("s %.5f" % std)
    #             # print("m %.5f" % median)
    #             out[i, j] = median
    #             # out[i, j] *= k
    # cv2.imshow('test1', image_np)
    # cv2.imshow('test', out)
    # cv2.imshow('test', mask_np * 255)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # return out
# def remove_bones(image, mask, mask_lung):
#     mask = torch.where(mask > 0.5, 1, 0).type(torch.uint8)
#     # kernel_size = 5
#     # kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
#     # weak_mask = torch.nn.functional.conv2d(mask, kernel, padding=kernel_size // 2)
#     image_np = image.squeeze().squeeze().detach().numpy().astype(np.uint8)
#     mask_np = mask.squeeze().squeeze().detach().numpy().astype(np.uint8)
#     mask_np_inv = 1 - mask_np
#     # mask_lung = np.where(mask_lung > 0, 1, 0).astype(np.uint8)
#     # ribs border
#     erode_window_size = 1
#     dilate_window_size = 1
#     kernel_erode_mask = np.ones((erode_window_size, erode_window_size), np.uint8)
#     kernel_dilate_mask = np.ones((dilate_window_size, dilate_window_size), np.uint8)
#     erode_mask_np = cv2.erode(src=mask_np, kernel=kernel_erode_mask, iterations=1)
#     dilate_mask_np = cv2.dilate(src=mask_np, kernel=kernel_dilate_mask, iterations=1)
#     rib_borders = dilate_mask_np - erode_mask_np
#     rib_borders_inv = 1 - rib_borders
#     mask_np_without_borders = mask_np - rib_borders
#     mask_np_without_borders_inv = 1 - mask_np_without_borders
#
#     image_only_ribs = image_np * (mask_np)
#     image_without_ribs = image_np * (mask_np_inv)
#
#     image_lung_only_ribs = mask_lung * mask_np
#     image_lung_without_ribs = mask_lung * mask_np_inv
#
#     cv2.imshow('test1', image_np)
#     cv2.imshow('test2', image_only_ribs)
#     cv2.imshow('test3', image_lung_only_ribs)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     mean_val_lung_only_ribs = np.mean(image_lung_only_ribs[image_lung_only_ribs > 0])
#     mean_val_lung_without_ribs = np.mean(image_lung_without_ribs[image_lung_without_ribs > 0])
#     mean_val_rib = mean_val_lung_only_ribs - mean_val_lung_without_ribs
#     print(mean_val_lung_only_ribs, mean_val_lung_without_ribs, mean_val_rib)
#
#     max_val_lung_only_ribs = np.max(image_lung_only_ribs)
#     max_val_lung_without_ribs = np.max(image_lung_without_ribs)
#     min_val_lung_only_ribs = np.min(image_lung_only_ribs)
#     min_val_lung_without_ribs = np.min(image_lung_without_ribs)
#     mean_val_max_rib = max_val_lung_only_ribs - max_val_lung_without_ribs
#     mean_val_min_rib = min_val_lung_only_ribs - min_val_lung_without_ribs
#     window_size = 10
#     # cv2.imshow('test22', (mask_np) * 255)
#     # cv2.imshow('test', (mask_np_without_borders) * 255)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     out = np.copy(image_np)
#     print(image_np.shape)
#     for i in range(image_np.shape[0]):
#         for j in range(image_np.shape[1]):
#             if mask_np[i, j] == 1:
#                 window = image_np[
#                                    max(i - window_size // 2, 0): min(i + window_size // 2 + 1, image_np.shape[0]),
#                                    max(j - window_size // 2, 0): min(j + window_size // 2 + 1, image_np.shape[1])]
#                 k = out[i, j] / np.max(window)
#                 print(k)
#                 out[i, j] = max((out[i, j] - mean_val_rib * k / 2), min_val_lung_without_ribs)

# cv2.imshow('test1', image_np)
# cv2.imshow('test', out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
