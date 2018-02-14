import os
import warnings
import sys

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
            if item == dp.NAME_SAVED_IMAGE + dp.IMG_FORMAT:
                if os.path.isfile(path + dp.NAME_SAVED_MASK + dp.IMG_FORMAT):
                    for nameSaved in [dp.NAME_SAVED_IMAGE, dp.NAME_SAVED_MASK]:
                        pic = imread(path + nameSaved + dp.IMG_FORMAT)
                        for angle in [90, 180, 270]:
                            array_cut_pic = cut_image(rotate(pic, angle, resize=True))
                            for numpic, finalpic in enumerate(array_cut_pic):
                                imsave(path + nameSaved + '_' + str(angle) + '_' + str(numpic) + dp.IMG_FORMAT,
                                       finalpic)


def data_white_black():

    # Get train and test IDs
    train_ids_ = next(os.walk("data/stage1_train"))[1]

    print('Getting and resizing images... ')
    for id_ in tqdm(train_ids_, total=len(train_ids_)):
        path = "data/stage1_train/" + id_ + "/images/" + id_
        img = imread(path + '.png', as_grey=True)

        save_path = "data/input_data/train/" + id_
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        imsave(save_path + '/image.png', img)

        path = "data/stage1_train/" + id_ + '/masks/'
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for mask_file in next(os.walk(path))[2]:
            if not mask_file.endswith('.png'):
                continue
            mask_ = imread(path + mask_file)
            mask = np.maximum(mask, mask_)

        imsave("data/input_data/train/" + id_ + '/mask.png', mask)


def cut_image(nparr, w_cut=dp.IMG_WIDTH, h_cut=dp.IMG_HEIGHT):
    h_img = nparr.shape[0]
    w_img = nparr.shape[1]
    for i in range(h_cut, h_img + h_cut//2, h_cut//2):
        lower_bound = min(i, h_img)
        for j in range(w_cut, w_img + w_cut//2, w_cut//2):
            right_bound = min(int(j), w_img)
            yield nparr[lower_bound-h_cut: lower_bound][right_bound-w_cut: right_bound]


if __name__ == "__main__":

    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

    if not sys.argv.__contains__('-grey'):
        data_white_black()

    if not sys.argv.__contains__('-rot'):
        data_rotate()
