from skimage.io import imsave, imread, imshow
import os
import tqdm
import warnings

PATH_FROM = "../../data/stage1_train/"


def cut_nuc_from_image():
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
                if not os.path.isdir("../../data/detector/" + id_ + "/images/"):
                    os.makedirs("../../data/detector/" + id_ + "/images/" )
                imsave("../../data/detector/" + id_ + "/images/" + m_id, image[upper:bottom, left:right])

                if not os.path.isdir("../../data/detector/" + id_ + "/masks/"):
                    os.makedirs("../../data/detector/" + id_ + "/masks/")
                imsave("../../data/detector/" + id_ + "/masks/" + m_id, mask[upper:bottom, left:right])
            except:
                print("EXCEPTION")


if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

    cut_nuc_from_image()
