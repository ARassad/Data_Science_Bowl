
import pandas as pd
import data_preparation
import re

data = pd.read_csv('sub-dsbowl2018-1.csv')
_, ids, sizes = data_preparation.get_test_data()

id_size = {}
for n in range(len(ids)):
    id_size[ids[n]] = sizes[n]

im_id = data["ImageId"]
en_px = data["EncodedPixels"]
patt = r"\d+"
for n in range(len(im_id)):
    id = im_id[n]
    px = en_px[n]
    w, h = id_size[id]
    count = w * h
    pixs = re.findall(patt, px)
    for i in range(0, len(pixs), 2):
        if count <= int(pixs[i]) + int(pixs[i+1]):
            print("WARNING")

# ДРАТУТИ