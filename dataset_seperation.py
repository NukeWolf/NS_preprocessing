import pandas as pd
import random
random.seed(10)


# VARIABLES
DATASET_PATHNAME = "/Volumes/NSS_dataset/processed"
IMAGES_PATHNAME = DATASET_PATHNAME + "/image"
CURATED_JSON = DATASET_PATHNAME + "/curated_labels.json"

TRAINING_RATIO = .7
VALIDATION_RATIO = .15
TEST_RATIO = .15

#In Frames per second
SAMPLE_RATE = 3
SAMPLE_MODULUS = 30 / SAMPLE_RATE


# Label and Column Separation
CURATED_DATA = pd.read_json(CURATED_JSON,orient="index")

CURATED_DATA['device'] = CURATED_DATA.apply(lambda row: row.name.split('/')[0], axis=1)
CURATED_DATA['subject_id'] = CURATED_DATA.apply(lambda row: row.name.split('/')[1], axis=1)
CURATED_DATA['eye_type'] = CURATED_DATA.apply(lambda row: row.name.split('/')[2], axis=1)
CURATED_DATA['cross_iris_group'] = CURATED_DATA.apply(lambda row: row.name.split('/')[3], axis=1)
CURATED_DATA['frame'] = CURATED_DATA.apply(lambda row: int(row.name.split('/')[4][:-4]), axis=1)


# Separate Pupil Coordinates
CURATED_DATA['pupil_center_x'] = CURATED_DATA['pupil_center'].apply(lambda center:center[0])
CURATED_DATA['pupil_center_y'] = CURATED_DATA['pupil_center'].apply(lambda center:center[1])
CURATED_DATA = CURATED_DATA.drop('pupil_center',axis=1)

# Separate all iris points
for x in range(8):
    CURATED_DATA[f'iris_point_x_{x}'] = CURATED_DATA['iris_lm'].apply(lambda iris_array:iris_array[x][0])
    CURATED_DATA[f'iris_point_y_{x}'] = CURATED_DATA['iris_lm'].apply(lambda iris_array:iris_array[x][1])
CURATED_DATA = CURATED_DATA.drop('iris_lm',axis=1)




# Dataset Slicing and Shuffling

def slice_dataset(arr,start_float,end_float):
    return arr[int((len(arr)+1)*start_float):int((len(arr)+1)*end_float)]

#X Subjects
subjects = CURATED_DATA['subject_id'].unique()
random.shuffle(subjects)

train_subj = slice_dataset(subjects,0,TRAINING_RATIO)
val_subj = slice_dataset(subjects,TRAINING_RATIO,TRAINING_RATIO + VALIDATION_RATIO)
test_subj = slice_dataset(subjects,TRAINING_RATIO + VALIDATION_RATIO, 1)

CURATED_DATA['dataset_type'] = CURATED_DATA['subject_id'].apply(lambda subj: 'train' if subj in train_subj else 'validation' if subj in val_subj else 'test')
CURATED_DATA.query(f'dataset_type == "train" and (frame % {SAMPLE_MODULUS}) == 0').to_csv('x_subjects/train.csv')
CURATED_DATA.query(f'dataset_type == "validation" and (frame % {SAMPLE_MODULUS}) == 0').to_csv('x_subjects/val.csv')
CURATED_DATA.query(f'dataset_type == "test" and (frame % {SAMPLE_MODULUS}) == 0').to_csv('x_subjects/test.csv')



# X iris groups
groups = CURATED_DATA['cross_iris_group'].unique()
random.shuffle(groups)

train_group = slice_dataset(groups,0,TRAINING_RATIO)
val_group = slice_dataset(groups,TRAINING_RATIO,TRAINING_RATIO + VALIDATION_RATIO)
test_group = slice_dataset(groups,TRAINING_RATIO + VALIDATION_RATIO, 1)

CURATED_DATA['dataset_type'] = CURATED_DATA['cross_iris_group'].apply(lambda group: 'train' if group in train_group else 'validation' if group in val_group else 'test')
CURATED_DATA.query(f'dataset_type == "train" and (frame % {SAMPLE_MODULUS}) == 0').to_csv('x_iris/train.csv')
CURATED_DATA.query(f'dataset_type == "validation" and (frame % {SAMPLE_MODULUS}) == 0').to_csv('x_iris/val.csv')
CURATED_DATA.query(f'dataset_type == "test" and (frame % {SAMPLE_MODULUS}) == 0').to_csv('x_iris/test.csv')



