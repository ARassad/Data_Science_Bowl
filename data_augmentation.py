import os
import warnings
import sys
import random

from tqdm import tqdm

from skimage.transform import rotate
from skimage.io import imread, imsave
import data_preparation as dp
import numpy as np


def data_rotate():

    if not os.path.isdir(dp.TRAIN_SAVE_PATH):
        raise OSError

    ids = next(os.walk(dp.TRAIN_SAVE_PATH))[1]

    print('Rotating images')
    for id_ in tqdm(ids, total=len(ids)):
        path = dp.TRAIN_SAVE_PATH + id_ + '/'
        if not os.path.isdir(path):
            raise OSError

        items = next(os.walk(path))[2]
        for item in items:
            if os.path.isfile(path + item) and item.endswith(dp.IMG_FORMAT):
                    pic = imread(path + item)
                    for angle in [90, 180, 270]:
                        imsave(path + item[:-len(dp.IMG_FORMAT)] + '_' + str(angle) + dp.IMG_FORMAT,
                               rotate(pic, angle, resize=True))


def data_white_black(dir=dp.TRAIN_PATH, save_dir=dp.TRAIN_SAVE_PATH):

    # Get train and test IDs
    train_ids_ = next(os.walk(dir))[1]

    print('Transforming to grey... ')
    for id_ in tqdm(train_ids_, total=len(train_ids_)):
        path = dir + id_ + "/images/" + id_
        img = imread(path + '.png', as_grey=True)

        save_path = save_dir + id_
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        imsave(save_path + '/image.png', img)

        if dir == dp.TRAIN_PATH:
            path = dir + id_ + '/masks/'
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            for mask_file in next(os.walk(path))[2]:
                if not mask_file.endswith('.png'):
                    continue
                mask_ = imread(path + mask_file)
                mask = np.maximum(mask, mask_)

            imsave(save_dir + id_ + '/mask.png', mask)


def cut_image(nparr, w_cut=dp.IMG_WIDTH, h_cut=dp.IMG_HEIGHT):
    h_img = nparr.shape[0]
    w_img = nparr.shape[1]
    for i in range(h_cut, h_img + h_cut//2, h_cut//2):
        lower_bound = min(i, h_img - 1)
        for j in range(w_cut, w_img + w_cut//2, w_cut//2):
            right_bound = min(int(j), w_img - 1)
            yield nparr[lower_bound-h_cut: lower_bound, right_bound-w_cut: right_bound]


def cut_images(dir=dp.TRAIN_SAVE_PATH):

    if not os.path.isdir(dir):
        raise OSError

    ids = next(os.walk(dir))[1]

    print('Cutting images')
    for id_ in tqdm(ids, total=len(ids)):
        path = dir + id_ + '/'
        if not os.path.isdir(path):
            raise OSError

        items = next(os.walk(path))[2]
        for item in items:
            if os.path.isfile(path + item) and item.endswith(dp.IMG_FORMAT):
                pic = imread(path + item)
                os.remove(path + item)
                for n, img in enumerate(cut_image(pic)):
                    imsave(path + item[:-len(dp.IMG_FORMAT)] + '_' + str(n) + dp.IMG_FORMAT, img)


def glue_image(arr_img, h_img, w_img, w_cut=dp.IMG_WIDTH, h_cut=dp.IMG_HEIGHT):
    
    maskres = np.zeros((h_img, w_img, 1), dtype=np.uint8)
    cur_img = 0
    for i in range(h_cut, h_img + h_cut//2, h_cut//2):
        lower_bound = min(i, h_img - 1)
        for j in range(w_cut, w_img + w_cut//2, w_cut//2):
            right_bound = min(int(j), w_img - 1)
            maskres[lower_bound-h_cut: lower_bound, right_bound-w_cut: right_bound] = \
                np.maximum(maskres[lower_bound-h_cut: lower_bound, right_bound-w_cut: right_bound], arr_img)
            cur_img += 1
    return maskres
    
    
def remove_empty_img():
    if not os.path.isdir(dp.TRAIN_SAVE_PATH):
        raise OSError

    ids = next(os.walk(dp.TRAIN_SAVE_PATH))[1]
    print('remove empty image')
    for id_ in tqdm(ids, total=len(ids)):
        path = dp.TRAIN_SAVE_PATH + id_ + '/'
        if not os.path.isdir(path):
            raise OSError

        items = next(os.walk(path))[2]
        for item in items:
            if os.path.isfile(path + item) and item.endswith(dp.IMG_FORMAT) and item.startswith(dp.NAME_SAVED_MASK):
                pic = imread(path + item)
                arr = pic.flatten()
                is_not_black = False
                for i in range(len(arr) - 1):
                    if arr[i] != arr[i+1]:
                        is_not_black = True
                        break
                if not is_not_black:
                    os.remove(path + item)
                    os.remove(path + dp.NAME_SAVED_IMAGE + item[len(dp.NAME_SAVED_MASK):])


def remove_part_data(part_to_del=0.5):
    if not os.path.isdir(dp.TRAIN_SAVE_PATH):
        raise OSError

    ids = next(os.walk(dp.TRAIN_SAVE_PATH))[1]
    print('remove part images')
    for id_ in tqdm(ids, total=len(ids)):
        path = dp.TRAIN_SAVE_PATH + id_ + '/'
        if not os.path.isdir(path):
            raise OSError

        items = next(os.walk(path))[2]
        for item in items:
            if os.path.isfile(path + item) and item.endswith(dp.IMG_FORMAT) and item.startswith(dp.NAME_SAVED_IMAGE):
                if random.randint(0, 100) <= part_to_del * 100:
                    os.remove(path + item)
                    os.remove(path + dp.NAME_SAVED_MASK + item[len(dp.NAME_SAVED_IMAGE):])


if __name__ == "__main__":

    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

    if '-g' not in sys.argv:
        data_white_black()

    if '-c' not in sys.argv:
        cut_images()

    if '-r' not in sys.argv:
        data_rotate()

    if '-d' not in sys.argv:
        remove_empty_img()

    if '-d' not in sys.argv:
        remove_part_data(0.75)
