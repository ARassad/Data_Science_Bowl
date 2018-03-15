
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

STRIDES = (8, 8)  # Шаги с которыми идет окошко детектора
COEF_RES = 1.5  # Коэффицент с которым уменьшаеться размер картинки
MIN_RESIZE = 4  #  Коэф минимального размера картинки
DETEC_SIZE = (22, 22)
UNET_SIZE = (20, 20)

SIZE_IMAGE = (24, 24)


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


if __name__ == "__main__":

    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

    new_test_ids = []
    rles = []

    detector = load_model("detector(24x24).h5")
    u_net = load_model("U-net/Unet(ver2).h5",  custom_objects={'mean_iou': mean_iou})

    dir_ = "../../data/stage1_test/"
    ids_test = os.walk(dir_)
    ids = next(ids_test)[1]
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        path = dir_ + id_ + "/images/"
        image = imread(path + id_ + ".png", as_grey=True)

        images = []
        masks = []
        # Уменьшаем размер картинки
        cur_size = (image.shape[0], image.shape[1])
        min_size = (image.shape[0]//MIN_RESIZE, image.shape[1]//MIN_RESIZE)

        images.append(image)
        masks.append(np.zeros(cur_size, dtype=np.float64))
        while SIZE_IMAGE[0] <= cur_size[0] >= min_size[0] and SIZE_IMAGE[1] <= cur_size[1] >= min_size[1]:
            cur_size = (int(cur_size[0] // COEF_RES), int(cur_size[1] // COEF_RES))

            images.append(resize(image, cur_size, mode='constant', preserve_range=True))
            masks.append(np.zeros(cur_size, dtype=np.float64))
            #masks[-1].fill(0.45)

        # Предсказание
        for i in range(len(images)):
            im = images[i]

            # Проход детектора
            for h in range(SIZE_IMAGE[0], im.shape[0] + STRIDES[0], STRIDES[0]):
                lower_bound = min(h, im.shape[0] - 1)
                for w in range(SIZE_IMAGE[1], im.shape[1] + STRIDES[1], STRIDES[1]):
                    right_bound = min(w, im.shape[1] - 1)

                    part_image_analis = np.ndarray((1, SIZE_IMAGE[0], SIZE_IMAGE[1], 1), dtype=np.float64)
                    part_image_analis[0] = im[lower_bound - SIZE_IMAGE[0]: lower_bound, right_bound - SIZE_IMAGE[1]: right_bound]\
                        .reshape((SIZE_IMAGE[0], SIZE_IMAGE[1], 1))

                    is_nucl = detector.predict(part_image_analis)[0][0]

                    if is_nucl > 0.9996:
                        pred_mask = u_net.predict(part_image_analis)[0]

                        copy_arr_to_arr(masks[i][lower_bound - SIZE_IMAGE[0]: lower_bound,
                                                 right_bound - SIZE_IMAGE[1]: right_bound], pred_mask.reshape(SIZE_IMAGE), "mean_except_zero")

        masks_resize = np.zeros((len(masks), image.shape[0], image.shape[1]), dtype=np.float64)
        masks_resize[0] = masks[0]
        for j in range(1, len(masks)):
            masks_resize[j] = resize(masks[j], (image.shape[0], image.shape[1]),  mode='constant', preserve_range=True)
            copy_arr_to_arr(masks_resize[0], masks_resize[j], mode="mean_except_zero")

        # Кодирование
        rle = list(prob_to_rles(masks[0], cutoff=0.5))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))
        if len(rle) == 0:
            rles.extend([[1, 1]])
            new_test_ids.extend([id_])
    '''
        asdd = np.array(masks_resize[0] > 0.5, dtype=np.float64)
        if not os.path.isdir("../../data/detect/" + id_ + "/"):
            os.makedirs("../../data/detect/" + id_ + "/")
        imsave("../../data/detect/" + id_ + "/" + "source.png", image)
        for i in range(len(masks)):
        imsave("../../data/detect/" + id_ + "/" + str(i) + ".png", masks[i])
    '''
    """
        fig = plt.figure(figsize=(8, 4))
        fig.add_subplot(1, 3, 1)
        plt.imshow(image)
        fig.add_subplot(1, 3, 2)
        plt.imshow(masks_resize[0])
        fig.add_subplot(1, 3, 3)
        plt.imshow(np.array(masks_resize[0] > 0.5, dtype=np.float64))
        # plt.show(block=True)
        plt.savefig("../../data/detector_unet_pred/{}.png".format(id_))
        plt.close()
    """

    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('detector_Unet.csv', index=False)






