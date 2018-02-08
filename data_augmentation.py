import os
import warnings
import tqdm

from skimage.transform import rotate
from skimage.io import imread, imsave
import data_preparation as dp


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
                    image = imread(path + dp.NAME_SAVED_IMAGE + dp.IMG_FORMAT)
                    mask = imread(path + dp.NAME_SAVED_MASK + dp.IMG_FORMAT)
                    for angle in [90, 180, 270]:
                        imsave(path + dp.NAME_SAVED_IMAGE + '_' + str(angle) + dp.IMG_FORMAT, rotate(image, angle))
                        imsave(path + dp.NAME_SAVED_MASK + '_' + str(angle) + dp.IMG_FORMAT, rotate(mask, angle))


if __name__ == "__main__":

    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

    data_rotate()

