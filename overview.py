import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import numpy as np

DATASET_PATHNAME = "/Volumes/NSS_dataset/processed"
IMAGES_PATHNAME = DATASET_PATHNAME + "/image"
CURATED_JSON = DATASET_PATHNAME + "/curated_labels.json"

overview_dict = dict()

CURATED_DATA = pd.read_json(CURATED_JSON)
CURATED_IMAGES_FILEPATHS = list(CURATED_DATA.columns)


IMAGES_PATHLIB = pathlib.Path(IMAGES_PATHNAME)


for device in IMAGES_PATHLIB.iterdir():
    overview_dict[device.name] = {'curated':0, "not_curated":0}
    for sample in device.rglob("*"):
        image_path = sample.relative_to(IMAGES_PATHLIB)
        if str(image_path) in CURATED_IMAGES_FILEPATHS:
            overview_dict[device.name]['curated'] += 1
        else:
            overview_dict[device.name]['not_curated'] += 1
pd.DataFrame.from_dict(overview_dict).to_json('overview_stats.json')

overview_dict = pd.read_json('overview_stats.json').to_dict()
overview_dict.pop('.DS_Store')




# Plot Graphing
width = 0.5

fig, ax = plt.subplots()
bottom = np.zeros(len(overview_dict))
print(bottom)

curated_count_list = [value['curated'] for value in overview_dict.values()]
uncurated_count_list = [value['not_curated'] for value in overview_dict.values()]

p = ax.bar(overview_dict.keys(), curated_count_list, width, label='Curated', bottom=bottom)
bottom += curated_count_list
p = ax.bar(overview_dict.keys(), uncurated_count_list, width, label='Not Curated', bottom=bottom)

ax.set_title("Dataset Curation")
ax.legend(loc="upper right")

plt.show()


