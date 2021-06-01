import random

import SimpleITK as sitk  # For loading the dataset
import numpy as np  # For data manipulation
from model import build_model  # For creating the model
import glob  # For populating the list of files
from scipy.ndimage import zoom  # For resizing
import re  # For parsing the filenames (to know their modality)
import math


def read_img(img_path):
    """
    Reads a .nii.gz image and returns as a numpy array.
    """
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))


def resize(img, shape, mode='constant', orig_shape=(155, 240, 240)):
    """
    Wrapper for scipy.ndimage.zoom suited for MRI images.
    """
    assert len(shape) == 3, "Can not have more than 3 dimensions"
    factors = (
        shape[0] / orig_shape[0],
        shape[1] / orig_shape[1],
        shape[2] / orig_shape[2]
    )

    # Resize to the given shape
    return zoom(img, factors, mode=mode)


def preprocess(img, out_shape=None):
    """
    Preprocess the image.
    Just an example, you can add more preprocessing steps if you wish to.
    """
    if out_shape is not None:
        img = resize(img, out_shape, mode='constant')

    # Normalize the image
    mean = img.mean()
    std = img.std()
    return (img - mean) / std


def preprocess_label(img, out_shape=None, mode='nearest'):
    """
    Separates out the 3 labels from the segmentation provided, namely:
    GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2))
    and the necrotic and non-enhancing tumor core (NCR/NET — label 1)
    """
    ncr = img == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET)
    ed = img == 2  # Peritumoral Edema (ED)
    et = img == 4  # GD-enhancing Tumor (ET)

    if out_shape is not None:
        ncr = resize(ncr, out_shape, mode=mode)
        ed = resize(ed, out_shape, mode=mode)
        et = resize(et, out_shape, mode=mode)

    return np.array([ncr, ed, et], dtype=np.uint8)


BRATS_TRAIN = '../data/brain-mri/brats/MICCAI_BraTS_2018_Data_Training/'
# TODO
BRATS_VAL = '../data/brain-mri/brats/MICCAI_BraTS_2018_Data_Training/'


def get_paths(path_root):
    # Get a list of files for all modalities individually
    t1 = glob.glob(path_root + '*GG/*/*t1.nii.gz')
    t2 = glob.glob(path_root + '*GG/*/*t2.nii.gz')
    flair = glob.glob(path_root + '*GG/*/*flair.nii.gz')
    t1ce = glob.glob(path_root + '*GG/*/*t1ce.nii.gz')
    seg = glob.glob(path_root + '*GG/*/*seg.nii.gz')  # Ground Truth

    pat = re.compile('.*_(\w*)\.nii\.gz')

    return [{
        pat.findall(item)[0]: item
        for item in items
    }
        for items in list(zip(t1, t2, t1ce, flair, seg))]


data_paths_train = get_paths(BRATS_TRAIN)
data_paths_val = get_paths(BRATS_VAL)

# TODO
max_samples = 2
data_paths_train = data_paths_train[:max_samples]
data_paths_val = data_paths_train

# TODO input shape
input_shape = (4, 32, 32, 16)
output_channels = 3


def data_gen(
        data_paths,
        batch_size,
        modalities=('t1', 't2', 't1ce', 'flair'),
):
    xs, ys = [], []

    def yield_batch():
        return np.array(xs), [np.concatenate([y[0] for y in ys]), np.array([y[1] for y in ys])]

    while True:
        shuffled_paths = random.sample(data_paths, len(data_paths))
        for imgs in shuffled_paths:
            try:
                x = np.array(
                    [preprocess(read_img(imgs[m]), input_shape[1:]) for m in modalities],
                    dtype=np.float32,
                )
                y = preprocess_label(read_img(imgs['seg']), input_shape[1:])[None, ...]
            except Exception as e:
                print(f'Something went wrong with {imgs[modalities[0]]}, skipping...\n Exception:\n{str(e)}')
                continue

            xs.append(x)
            # return x as well for the VAE reconstruction loss
            ys.append([y, x])

            if len(xs) == batch_size:
                yield yield_batch()
                xs, ys = [], []

        # in case of the last batch being smaller than batch_size
        if len(xs) > 0:
            yield yield_batch()


# TODO parameterize loss weights etc
model = build_model(input_shape=input_shape, output_channels=3)
# TODO parameterize epochs, batch_size
batch_size = 1
model.fit_generator(
    data_gen(data_paths_train, batch_size=batch_size),
    epochs=8,
    steps_per_epoch=len(data_paths_train) // batch_size,
    validation_data=data_gen(data_paths_val, batch_size=batch_size),
    validation_steps=len(data_paths_val) // batch_size,
)
# TODO path
model.save('model.h5')
