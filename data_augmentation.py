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

        fileNames = next(os.walk(path))[2]
        for fileName in fileNames:
            pic = imread(path + fileName)

            for angle in range(90,360,90):
                p = fileName.split('.')
                newName = "".join(fileName.split('.')[0:-1]) + '_' + str(angle) + dp.IMG_FORMAT
                imsave( os.path.join(path, newName ), rotate(pic, angle))


if __name__ == "__main__":

    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

    data_rotate()

