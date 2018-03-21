
import os
import pandas as pd
from tqdm import tqdm
from skimage.io import imread, imshow, imsave
from keras.models import load_model
from skimage.transform import resize
from function import mean_iou, prob_to_rles
import numpy as np
import matplotlib.pyplot as plt
import warnings
from keras import applications
from keras.utils.generic_utils import CustomObjectScope
from nms import non_max_suppression_fast
import matplotlib.patches as patches


STRIDES = (4, 4)  # Шаги с которыми идет окошко детектора
COEF_RES = 1.3  # Коэффицент с которым уменьшаеться размер картинки
MIN_RESIZE = 4  #  Коэф минимального размера картинки
MAX_RESIZE = 2

SIZE_IMAGE = (32, 32)


def copy_arr_to_arr(to, from_, mode="mean"):
    if mode == 'mean':
        np.copyto(to, (from_ + to) / 2)
    elif mode == 'min':
        np.copyto(to, np.minimum(from_, to))
    elif mode == 'max':
        np.copyto(to, np.maximum(from_, to))
    elif mode == 'cuttof':
        np.copyto(to, np.maximum(np.array(from_ > 0.5, dtype=np.float64), to))
    elif mode == "mean_except_zero":
        for i in range(to.shape[0]):
            for j in range(to.shape[1]):
                if to[i][j] < 0.05 or from_[i][j] < 0.05:
                    to[i][j] = max(to[i][j], from_[i][j])
                else:
                    to[i][j] = (to[i][j] + from_[i][j]) / 2


def save_result(sdir, source, res_mask, masks, bbs, bbs_nms, res_bbs=None, res_nms=None):
    fig = plt.figure(figsize=(16, 8))
    fig.add_subplot(3, len(masks), 1)
    plt.imshow(source)
    plt.title("Исходное")
    fig.add_subplot(3, len(masks), 2)
    plt.imshow(res_mask)
    plt.title("Выход сети")
    fig.add_subplot(3, len(masks), 3)
    plt.imshow(np.array(res_mask > 0.5, dtype=np.float64))
    plt.title("Маска для сабмита")

    if res_nms is not None and res_bbs is not None:
        ax = fig.add_subplot(3, len(masks), 4)
        plt.imshow(source)
        plt.title("До NMS")
        for b in res_bbs:
            ax.add_patch(
                patches.Rectangle((b[0], b[1]), b[3] - b[1], b[2] - b[0], linewidth=1, edgecolor='r', facecolor='none'))

        ax = fig.add_subplot(3, len(masks), 5)
        plt.imshow(res_mask)
        plt.title("После NMS")
        for b in res_nms:
            ax.add_patch(
                patches.Rectangle((b[0], b[1]), b[3] - b[1], b[2] - b[0], linewidth=1, edgecolor='r', facecolor='none'))

    for i in range(len(masks)):
        ax = fig.add_subplot(3, len(masks), i+1+len(masks))
        plt.imshow(masks[i])
        plt.title("{}".format(masks[i].shape))
        for b in bbs[i]:
            ax.add_patch(
                patches.Rectangle((b[0], b[1]), b[3] - b[1], b[2] - b[0], linewidth=1, edgecolor='r', facecolor='none'))

    for i in range(len(masks)):
        ax = fig.add_subplot(3, len(masks), i+1+2*len(masks))
        plt.imshow(masks[i])
        plt.title("{}".format(masks[i].shape))
        for b in bbs_nms[i]:
            ax.add_patch(
                patches.Rectangle((b[0], b[1]), b[3] - b[1], b[2] - b[0], linewidth=1, edgecolor='r', facecolor='none'))

    plt.savefig(sdir)
    plt.close()


def main_defolt():
    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

    new_test_ids = []
    rles = []

    detector = None
    with CustomObjectScope({'relu6': applications.mobilenet.relu6,
                            'DepthwiseConv2D': applications.mobilenet.DepthwiseConv2D}):
        detector = load_model('detector_MobileNet.h5')
    u_net = load_model("U-net/Unet(32x32).h5", custom_objects={'mean_iou': mean_iou})

    dir_ = "../../data/stage1_test/"
    ids_test = os.walk(dir_)
    ids = next(ids_test)[1]
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        try:
            path = dir_ + id_ + "/images/"
            image = imread(path + id_ + ".png", as_grey=True)

            images = []
            masks = []
            # Уменьшаем размер картинки
            cur_size = (image.shape[0], image.shape[1])
            min_size = (image.shape[0] // MIN_RESIZE, image.shape[1] // MIN_RESIZE)

            images.append(image)
            masks.append(np.zeros(cur_size, dtype=np.float64))
            while SIZE_IMAGE[0] <= cur_size[0] >= min_size[0] and SIZE_IMAGE[1] <= cur_size[1] >= min_size[1]:
                cur_size = (int(cur_size[0] // COEF_RES), int(cur_size[1] // COEF_RES))
                if SIZE_IMAGE[0] > cur_size[0] or SIZE_IMAGE[1] > cur_size[1]:
                    break

                images.append(resize(image, cur_size, mode='constant', preserve_range=True))
                masks.append(np.zeros(cur_size, dtype=np.float64))
                # masks[-1].fill(0.45)

            # Предсказание
            for i in range(len(images)):
                im = images[i]

                # Проход детектора
                for h in range(SIZE_IMAGE[0], im.shape[0] + STRIDES[0], STRIDES[0]):
                    lower_bound = min(h, im.shape[0] - 1)
                    for w in range(SIZE_IMAGE[1], im.shape[1] + STRIDES[1], STRIDES[1]):
                        right_bound = min(w, im.shape[1] - 1)

                        part_image_analis = np.ndarray((1, SIZE_IMAGE[0], SIZE_IMAGE[1], 1), dtype=np.float64)
                        part_image_analis[0] = im[lower_bound - SIZE_IMAGE[0]: lower_bound,
                                               right_bound - SIZE_IMAGE[1]: right_bound] \
                            .reshape((SIZE_IMAGE[0], SIZE_IMAGE[1], 1))

                        is_nucl = detector.predict(part_image_analis)[0][0]

                        if is_nucl > 0.99:
                            pred_mask = u_net.predict(part_image_analis)[0]

                            copy_arr_to_arr(masks[i][lower_bound - SIZE_IMAGE[0]: lower_bound,
                                            right_bound - SIZE_IMAGE[1]: right_bound], pred_mask.reshape(SIZE_IMAGE),
                                            "mean_except_zero")

            masks_resize = np.zeros((len(masks), image.shape[0], image.shape[1]), dtype=np.float64)
            masks_resize[0] = masks[0]
            for j in range(1, len(masks)):
                masks_resize[j] = resize(masks[j], (image.shape[0], image.shape[1]), mode='constant',
                                         preserve_range=True)
                copy_arr_to_arr(masks_resize[0], masks_resize[j], mode="mean_except_zero")

            # Кодирование
            mask_to_encde = masks_resize[0]

            rle = list(prob_to_rles(mask_to_encde, cutoff=0.5))
            rles.extend(rle)
            new_test_ids.extend([id_] * len(rle))
            if len(rle) == 0:
                rles.extend([[1, 1]])
                new_test_ids.extend([id_])

            # Вывод
            asdd = np.array(masks_resize[0] > 0.5, dtype=np.float64)
            if not os.path.isdir("../../data/detect/" + id_ + "/"):
                os.makedirs("../../data/detect/" + id_ + "/")
            imsave("../../data/detect/" + id_ + "/" + "source.png", image)
            for i in range(len(masks)):
                imsave("../../data/detect/" + id_ + "/" + str(i) + ".png", masks[i])

            fig = plt.figure(figsize=(8, 4))
            fig.add_subplot(1, 3, 1)
            plt.imshow(image)
            plt.title("Исходное")
            fig.add_subplot(1, 3, 2)
            plt.imshow(mask_to_encde)
            plt.title("Выход сети")
            fig.add_subplot(1, 3, 3)
            plt.imshow(np.array(mask_to_encde > 0.5, dtype=np.float64))
            plt.title("Маска для сабмита")
            # plt.show(block=True)
            plt.savefig("../../data/detector_unet_pred/{}.png".format(id_))
            plt.close()

        except:
            print("Exception id: " + id_)

    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('detector_Unet.csv', index=False)


def main_ROI_NMS():
    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

    new_test_ids = []
    rles = []

    detector = None
    with CustomObjectScope({'relu6': applications.mobilenet.relu6,
                            'DepthwiseConv2D': applications.mobilenet.DepthwiseConv2D}):
        detector = load_model('detector_MobileNet.h5')
    u_net = load_model("U-net/Unet(32x32).h5", custom_objects={'mean_iou': mean_iou})

    dir_ = "../../data/stage1_test/"
    #dir_ = "../../data/1/"
    ids_test = os.walk(dir_)
    ids = next(ids_test)[1]
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        try:
            path = dir_ + id_ + "/images/"
            image = imread(path + id_ + ".png", as_grey=True)

            images = []
            masks = []

            bounding_boxs = []
            bounding_boxs_nms = []

            # Уменьшаем размер картинки
            cur_size = (image.shape[0], image.shape[1])
            min_size = (image.shape[0] // MIN_RESIZE, image.shape[1] // MIN_RESIZE)

            images.append(image)
            masks.append(np.zeros(cur_size, dtype=np.float64))
            while SIZE_IMAGE[0] <= cur_size[0] >= min_size[0] and SIZE_IMAGE[1] <= cur_size[1] >= min_size[1]:
                cur_size = (int(cur_size[0] // COEF_RES), int(cur_size[1] // COEF_RES))
                if SIZE_IMAGE[0] > cur_size[0] or SIZE_IMAGE[1] > cur_size[1]:
                    break

                images.append(resize(image, cur_size, mode='constant', preserve_range=True))
                masks.append(np.zeros(cur_size, dtype=np.float64))
                # masks[-1].fill(0.45)

            # Предсказание
            for i in range(len(images)):
                im = images[i]

                # Проход детектора
                bbs = []
                for h in range(SIZE_IMAGE[0], im.shape[0] + STRIDES[0], STRIDES[0]):
                    lower_bound = min(h, im.shape[0])
                    for w in range(SIZE_IMAGE[1], im.shape[1] + STRIDES[1], STRIDES[1]):
                        right_bound = min(w, im.shape[1])

                        part_image_analis = np.ndarray((1, SIZE_IMAGE[0], SIZE_IMAGE[1], 1), dtype=np.float64)
                        part_image_analis[0] = im[lower_bound - SIZE_IMAGE[0]: lower_bound,
                                               right_bound - SIZE_IMAGE[1]: right_bound] \
                            .reshape((SIZE_IMAGE[0], SIZE_IMAGE[1], 1))

                        is_nucl = detector.predict(part_image_analis)[0][0]

                        if is_nucl > 0.99:
                            bbs.append((right_bound-SIZE_IMAGE[0],
                                        lower_bound-SIZE_IMAGE[1],
                                        right_bound,
                                        lower_bound))

                bounding_boxs.append(bbs)
                bounding_boxs_nms.append(non_max_suppression_fast(np.array(bbs), 0.6))

                # Предсказание по NMS
                for box in bounding_boxs_nms[-1]:
                    im_to_unet = np.ndarray((1, SIZE_IMAGE[0], SIZE_IMAGE[1], 1), dtype=np.float64)
                    im_to_unet[0] = im[box[1]:box[3], box[0]:box[2]].reshape((SIZE_IMAGE[0], SIZE_IMAGE[1], 1))
                    pred_mask = u_net.predict(im_to_unet)[0]
                    copy_arr_to_arr(masks[i][box[1]:box[3], box[0]:box[2]], pred_mask.reshape(SIZE_IMAGE),
                                    "mean_except_zero")

            masks_resize = np.zeros((len(masks), image.shape[0], image.shape[1]), dtype=np.float64)
            masks_resize[0] = masks[0]
            for j in range(1, len(masks)):
                masks_resize[j] = resize(masks[j], (image.shape[0], image.shape[1]), mode='constant',
                                         preserve_range=True)
                copy_arr_to_arr(masks_resize[0], masks_resize[j], mode="mean_except_zero")

            # Кодирование
            mask_to_encde = masks_resize[0]

            rle = list(prob_to_rles(mask_to_encde, cutoff=0.5))
            rles.extend(rle)
            new_test_ids.extend([id_] * len(rle))
            if len(rle) == 0:
                rles.extend([[1, 1]])
                new_test_ids.extend([id_])

            save_result("../../data/detector_unet_pred/{}.png".format(id_), image, mask_to_encde, masks,
                        bounding_boxs, bounding_boxs_nms)

        except StopIteration:
            print("Exception id: " + id_)

    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

    # Чистка от мусора
    def f(x):
        res = 0
        for i in range(1, len(x), 2):
            res += i
        return res

    sub['CountPix'] = pd.Series(rles).apply(f)
    sub = sub[sub['CountPix'] > 10][['ImageId', 'EncodedPixels']]

    sub.to_csv('detector_Unet.csv', index=False)


def main_merge_with_NMS():
    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

    new_test_ids = []
    rles = []

    detector = None
    detector_second = None
    with CustomObjectScope({'relu6': applications.mobilenet.relu6,
                            'DepthwiseConv2D': applications.mobilenet.DepthwiseConv2D}):
        detector = load_model('detector_MobileNet.h5')
        detector_second = load_model('detector_MobileNet_Second.h5')
    u_net = load_model("U-net/Unet(32x32).h5", custom_objects={'mean_iou': mean_iou})

    dir_ = "../../data/stage1_test/"
    #dir_ = "../../data/1/"
    ids_test = os.walk(dir_)
    ids = next(ids_test)[1]
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        try:
            path = dir_ + id_ + "/images/"
            image = imread(path + id_ + ".png", as_grey=True)

            images = []
            masks = []

            bounding_boxs = []
            bounding_boxs_nms = []

            # Создаем разные размеры исходной картинки
            cur_size = (int(image.shape[0] * MAX_RESIZE), int(image.shape[1] * MAX_RESIZE))
            min_size = (image.shape[0] // MIN_RESIZE, image.shape[1] // MIN_RESIZE)

            images.append(image)
            masks.append(np.zeros(image.shape, dtype=np.float64))
            while True:
                images.append(resize(image, cur_size, mode='constant', preserve_range=True))
                masks.append(np.zeros(cur_size, dtype=np.float64))

                cur_size = (int(cur_size[0] // COEF_RES), int(cur_size[1] // COEF_RES))
                if cur_size[0] <= min_size[0] or cur_size[1] <= min_size[1]:
                    break

            # Предсказание
            for i in range(len(images)):
                im = images[i]

                # Проход детектора
                bbs = []
                for h in range(SIZE_IMAGE[0], im.shape[0] + STRIDES[0], STRIDES[0]):
                    lower_bound = min(h, im.shape[0])

                    for w in range(SIZE_IMAGE[1], im.shape[1] + STRIDES[1], STRIDES[1]):
                        right_bound = min(w, im.shape[1])

                        part_image_analis = np.ndarray((1, SIZE_IMAGE[0], SIZE_IMAGE[1], 1), dtype=np.float64)
                        part_image_analis[0] = im[lower_bound - SIZE_IMAGE[0]: lower_bound,
                                               right_bound - SIZE_IMAGE[1]: right_bound] \
                            .reshape((SIZE_IMAGE[0], SIZE_IMAGE[1], 1))

                        # Говорит что в квадрате есть клетки
                        is_nucl = detector.predict(part_image_analis)[0][0]

                        if is_nucl > 0.99:

                            # Говорит, что в квадрате только одна клетка
                            is_one_nucl = detector_second.predict(part_image_analis)[0][0]

                            if is_one_nucl > 0.95:
                                bbs.append((right_bound - SIZE_IMAGE[0],
                                            lower_bound - SIZE_IMAGE[1],
                                            right_bound,
                                            lower_bound))

                bounding_boxs.append(bbs)
                bounding_boxs_nms.append(non_max_suppression_fast(np.array(bbs), 0.6))

            result_mask = np.zeros(image.shape, dtype=np.float64)
            result_bbs = []

            # Мержим все боксы
            for i in range(len(images)):
                h_coef = images[i].shape[0] / image.shape[0]
                w_coef = images[i].shape[1] / image.shape[1]
                for bb in bounding_boxs_nms[i]:
                    box = (bb[0] // w_coef,
                           bb[1] // h_coef,
                           bb[2] // w_coef,
                           bb[3] // h_coef)
                    result_bbs.append(box)

            result_nms = non_max_suppression_fast(np.array(result_bbs), 0.5)

            # Предсказание по NMS
            for box in result_nms:
                im_to_unet = np.ndarray((1, SIZE_IMAGE[0], SIZE_IMAGE[1], 1), dtype=np.float64)

                # Приводим размеры квадрата к размерам входа Юнет
                im_to_unet[0] = resize(image[box[1]:box[3], box[0]:box[2]], SIZE_IMAGE,
                                       mode='constant', preserve_range=True).reshape((SIZE_IMAGE[0], SIZE_IMAGE[1], 1))
                # Предсказывание
                pred_mask = u_net.predict(im_to_unet)[0]

                # Возвращаем начальный размер и вставляем в итоговую маску
                copy_arr_to_arr(result_mask[box[1]:box[3], box[0]:box[2]],
                                resize(pred_mask, (box[3] - box[1], box[2] - box[0]), mode='constant', preserve_range=True),
                                "mean_except_zero")

            # Кодирование
            mask_to_encde = result_mask

            rle = list(prob_to_rles(mask_to_encde, cutoff=0.5))
            rles.extend(rle)
            new_test_ids.extend([id_] * len(rle))
            if len(rle) == 0:
                rles.extend([[1, 1]])
                new_test_ids.extend([id_])

            save_result("../../data/detector_unet_pred/{}.png".format(id_), image, mask_to_encde, images,
                        bounding_boxs, bounding_boxs_nms, result_bbs, result_nms)

        except StopIteration:
            print("Exception id: " + id_)

    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

    # Чистка от мусора
    def f(x):
        res = 0
        for i in range(1, len(x), 2):
            res += i
        return res

    sub['CountPix'] = pd.Series(rles).apply(f)
    sub = sub[sub['CountPix'] > 10][['ImageId', 'EncodedPixels']]

    sub.to_csv('detector_Unet.csv', index=False)


if __name__ == "__main__":
    main_merge_with_NMS()






