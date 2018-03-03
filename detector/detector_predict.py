
from keras.models import load_model
import os
from skimage.io import imread, imshow
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from detector_data_prep import get_nucleas, PATH_TO, PATH_TO_NON_NUCL

dirr = "../../data/detector_pred/"

if __name__ == "__main__":

    h = 22
    w = 22

    model = load_model("detector.h5")

    X_yes, masks = get_nucleas(None, dir=PATH_TO, only_image=False, shape=(h, w, 1))
    #X_non, _ = get_nucleas(None, dir=PATH_TO_NON_NUCL, only_image=True, shape=(h, w, 1))

    print("Predict image with Nucleas")
    X_yes_pred = model.predict(X_yes)
    #print(X_yes_pred)

    #print("Predict image without Nucleas")
    #print(model.predict(X_non))

    num = 0
    for n, i in enumerate(X_yes_pred):
        if i[0] < 0.5:
            num += 1
            fig = plt.figure(figsize=(8, 4))
            fig.add_subplot(1, 2, 1)
            plt.imshow(X_yes[n].reshape(22, 22))
            fig.add_subplot(1, 2, 2)
            plt.imshow(masks[n].reshape(22, 22))
            plt.title(str(i[0]))
            #plt.show(block=True)
            plt.savefig("../../data/detector_pred/{}.png".format(n))
            plt.close()

    print( num / len(X_yes_pred))
