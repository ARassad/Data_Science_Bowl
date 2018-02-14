import os
import warnings
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
    for id_ in tqdm.tqdm(ids, total=len(ids)):
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
                            imsave(path + nameSaved + '_' + str(angle) + dp.IMG_FORMAT, rotate(pic, angle))


def data_white_black():

    img_white_black = []
    masks = []

    # Get train and test IDs
    train_ids_ = next(os.walk("data/stage1_train"))[1]

    print('Getting and resizing images... ')
    for id_ in tqdm(train_ids_, total=len(train_ids_)):
        path = "data/stage1_train/" + id_ + "/images/" + id_
        img = imread(path + '.png', as_grey=True)

        save_path = "data/input_data/train/" + id_
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        imsave(save_path + '/image_wh_bl.png', img)

        path = "data/stage1_train/" + id_ + '/masks/'
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for mask_file in next(os.walk(path))[2]:
            if not mask_file.endswith('.png'):
                continue
            mask_ = imread(path + mask_file)
            mask = np.maximum(mask, mask_)

        imsave("data/input_data/train/"+ id_ + '/mask_wh_bl.png', mask)


if __name__ == "__main__":

    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

    data_white_black()
