import data_augmentation as da
import detector.detector_data_prep as ddp
import os
from tqdm import tqdm

if __name__ == "__main__":

    dir_ = "../../../data/detector/"
    ids = next(os.walk(dir_))[1]
    for id_ in tqdm(ids, total=len(ids)):
        for p in ["images/", "masks/"]:
            path = dir_ + id_ + "/" + p
            da.rotate_images_in_directory(path)
