from skimage.io import imsave, imread, imshow
import os
import tqdm
import warnings

import random
import numpy as np
from skimage.transform import resize

PATH_FROM = "../../data/stage1_train/"
PATH_TO = "../../data/detector/"
PATH_TO_NON_NUCL = "../../data/detector_non/"


def cut_nuclears():
    ids = next(os.walk(PATH_FROM))[1]

    for id_ in tqdm.tqdm(ids, total=len(ids)):
        path = PATH_FROM + id_
        image = imread(path + "/images/" + id_ + ".png")[:, :, :3]

        masks_ids = next(os.walk(path + "/masks/"))[2]
        for m_id in masks_ids:
            if not m_id.endswith('.png'):
                break
            mask = imread(path + "/masks/" + m_id)[:, :]

            #  Тупо, переписать
            upper = -1
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i][j] > 0:
                        upper = i
                        break
                if upper != -1:
                    break

            bottom = mask.shape[0] - 1
            empty_row = True
            for i in range(upper, mask.shape[0]):
                empty_row = True
                for j in range(mask.shape[1]):
                    if mask[i][j] > 0:
                        empty_row = False
                        break
                if empty_row:
                    bottom = i
                    break

            left = -1
            for i in range(mask.shape[1]):
                for j in range(upper, bottom + 1):
                    if mask[j][i] > 0:
                        left = i
                        break
                if left != -1:
                    break

            right = -1
            for i in range(mask.shape[1]-1, -1, -1):
                for j in range(upper, bottom + 1):
                    if mask[j][i] > 0:
                        right = i
                        break
                if right != -1:
                    break

            try:
                if not os.path.isdir(PATH_TO + id_ + "/images/"):
                    os.makedirs(PATH_TO + id_ + "/images/" )
                imsave(PATH_TO + id_ + "/images/" + m_id, image[upper:bottom, left:right])

                if not os.path.isdir(PATH_TO + id_ + "/masks/"):
                    os.makedirs(PATH_TO + id_ + "/masks/")
                imsave(PATH_TO + id_ + "/masks/" + m_id, mask[upper:bottom, left:right])
            except:
                print("EXCEPTION")


def cut_non_nuclears(shape_win=(40, 40), strides=(40, 40), limit=(0.000, 0.02), part_one_color_image=0.05,
                     min_diff_color=5):
    ids = next(os.walk(PATH_FROM))[1]
    id_im = 0
    print("")
    for id_ in tqdm.tqdm(ids, total=len(ids)):
        path = PATH_FROM + id_
        image = imread(path + "/images/" + id_ + ".png")[:, :, :3]

        mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
        for m_id in next(os.walk(path + "/masks/"))[2]:
            if not m_id.endswith('.png'):
                break
            mask = np.maximum(mask, np.expand_dims(imread(path + "/masks/" + m_id)[:, :], axis=-1))

        h_win, h_img, h_strd = shape_win[0], mask.shape[0], strides[0]
        w_win, w_img, w_strd = shape_win[1], mask.shape[1], strides[1]
        for h in range(h_win, h_img + h_strd, h_strd):
            lower_bound = min(h, h_img - 1)
            for w in range(w_win, w_img + w_strd, w_strd):
                right_bound = min(w, w_img - 1)

                part_mask = mask[lower_bound-h_win: lower_bound, right_bound-w_win: right_bound]

                #  Сохранение маски и изображения если белых пикселей в маске немного
                size = part_mask.shape[0] * part_mask.shape[1]
                if size * limit[0] <= num_nuclea_pix(part_mask) <= size * limit[1]:
                    id_im += 1
                    save_image = image[lower_bound-h_win: lower_bound, right_bound-w_win: right_bound]

                    if save_image.max() - save_image.min() > min_diff_color \
                            or random.randint(0, 100) <= 100 * part_one_color_image:

                        if not os.path.isdir(PATH_TO_NON_NUCL + id_ + "/images/"):
                            os.makedirs(PATH_TO_NON_NUCL + id_ + "/images/")
                        imsave(PATH_TO_NON_NUCL + id_ + "/images/" + str(id_im) + ".png", save_image)
                        if not os.path.isdir(PATH_TO_NON_NUCL + id_ + "/masks/"):
                            os.makedirs(PATH_TO_NON_NUCL + id_ + "/masks/")
                        imsave(PATH_TO_NON_NUCL + id_ + "/masks/" + str(id_im) + ".png", part_mask.reshape(shape_win))

    return None


def get_nucleas(max_size=None, dir=PATH_TO, shape=(100, 100), only_image=False):
    ids = next(os.walk(dir))[1]

    images = []
    masks = []

    for n, id_ in tqdm.tqdm(enumerate(ids), total=len(ids)):
        if max_size is not None and n > max_size:
            break

        path = dir + id_
        path_im = path + "/images/"
        path_mk = path + "/masks/"

        for im_id in next(os.walk(path_im))[2]:
            if not im_id.endswith('.png'):
                break

            images.append(resize(imread(path_im + im_id, as_grey=True), shape, mode="constant", preserve_range=True))

            if not only_image:
                masks.append(
                    resize(imread(path_mk + im_id, as_grey=True) / 255, shape, mode="constant", preserve_range=True))

    return np.array(images), np.array(masks)


def num_nuclea_pix(mask):
    non_nucl_pix, num_nucl_pix = 0, 0

    for pix in mask.flatten():
        num_nucl_pix += int(pix != non_nucl_pix)

    return num_nucl_pix


if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

    cut_non_nuclears()
