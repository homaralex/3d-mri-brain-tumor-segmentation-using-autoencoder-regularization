import argparse
import random
import glob  # For populating the list of files
import re  # For parsing the filenames (to know their modality)
from datetime import datetime
from pathlib import Path

import gin
import matplotlib.pyplot as plt
import SimpleITK as sitk  # For loading the dataset
import numpy as np  # For data manipulation
import keras.callbacks as k_callbacks
from scipy.ndimage import zoom  # For resizing

from model import build_model  # For creating the model


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


def preprocess(img, out_shape=None, augment=True):
    # normalize the image (based on non-zero voxels)
    mean = img[img != 0].mean()
    std = img[img != 0].std()

    if augment:
        # random intensity and scale shifts
        mean += std * np.random.uniform(-.1, .1)
        std *= np.random.uniform(.9, 1.1)

        # random axis flipping
        for ax_idx in range(3):
            if random.getrandbits(1):
                img = np.flip(img, axis=ax_idx)

    img = (img - mean) / std

    if out_shape is not None:
        img = resize(img, out_shape, mode='constant')

    return img


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

    return np.array([ncr, ed, et], dtype=np.float32)


def save_preds(
        model,
        data,
        model_dir,
        data_format,
):
    data, _ = next(data_gen(data, 1))

    ret = model.predict(data)

    if data_format == 'channels_first':
        plt.imshow(ret[0][0][0][20], cmap='Greys_r')
        plt.savefig(model_dir / 'segmentation.png')
        plt.imshow(ret[1][0][0][20], cmap='Greys_r')
        plt.savefig(model_dir / 'reconstruction.png')
        plt.imshow(data[0][0][20], cmap='Greys_r')
        plt.savefig(model_dir / 'original.png')
    else:
        plt.imshow(ret[0][0][:, :, :, 0][20], cmap='Greys_r')
        plt.savefig(model_dir / 'segmentation.png')
        plt.imshow(ret[1][0][:, :, :, 0][20], cmap='Greys_r')
        plt.savefig(model_dir / 'reconstruction.png')
        plt.imshow(data[0][:, :, :, 0][20], cmap='Greys_r')
        plt.savefig(model_dir / 'original.png')


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


@gin.configurable(denylist=['data_paths'])
def data_gen(
        data_paths,
        batch_size,
        input_shape,
        modalities,
        data_format,
):
    xs, ys = [], []
    out_shape = input_shape[1:] if data_format == 'channels_first' else input_shape[:-1]

    def yield_batch(xs, ys):
        xs, ys = np.array(xs), [np.concatenate([y[0] for y in ys]), np.array([y[1] for y in ys])]

        if data_format == 'channels_last':
            xs, ys = np.moveaxis(xs, 1, -1), [np.moveaxis(ys[0], 1, -1), np.moveaxis(ys[1], 1, -1)]

        return xs, ys

    while True:
        shuffled_paths = random.sample(data_paths, len(data_paths))
        for imgs in shuffled_paths:
            try:
                x = np.array(
                    [preprocess(read_img(imgs[m]), out_shape) for m in modalities],
                    dtype=np.float32,
                )
                y = preprocess_label(read_img(imgs['seg']), out_shape)[None, ...]
            except Exception as e:
                print(f'Something went wrong with {imgs[modalities[0]]}, skipping...\n Exception:\n{str(e)}')
                continue

            xs.append(x)
            # return x as well for the VAE reconstruction loss
            ys.append([y, x])

            if len(xs) == batch_size:
                yield yield_batch(xs, ys)
                xs, ys = [], []

        # in case of the last batch being smaller than batch_size
        if len(xs) > 0:
            yield yield_batch(xs, ys)


@gin.configurable
def train(
        brats_train_dir=gin.REQUIRED,
        brats_val_dir=None,
        val_ratio=.2,
        model_name='ResNet3DVAE_Brats',
        input_shape=(160, 192, 128),
        data_format='channels_last',
        modalities=('t1', 't2', 't1ce', 'flair'),
        batch_size=1,
        epochs=300,
        # for debugging purposes
        max_samples=None,
):
    assert len(input_shape) == 3
    input_shape = (len(modalities),) + input_shape if data_format == 'channels_first' else input_shape + (
        len(modalities),)
    gin.bind_parameter('data_gen.input_shape', input_shape)
    gin.bind_parameter('data_gen.batch_size', batch_size)
    gin.bind_parameter('data_gen.modalities', modalities)
    gin.bind_parameter('data_gen.data_format', data_format)

    data_paths_train = get_paths(brats_train_dir)
    if brats_val_dir is None:
        random.shuffle(data_paths_train)
        stop_idx = int(val_ratio * len(data_paths_train))
        data_paths_val = data_paths_train[:stop_idx][:max_samples]
        data_paths_train = data_paths_train[stop_idx:]
    else:
        data_paths_val = get_paths(brats_val_dir)[:max_samples]
    data_paths_train = data_paths_train[:max_samples]
    print(f'Train samples: {len(data_paths_train)}\nVal samples: {len(data_paths_val)}')

    model_dir = Path('models') / model_name / datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    model_dir.mkdir(exist_ok=True, parents=True)
    print(f'Model dir: {str(model_dir)}')
    # save the gin config to file
    print(gin.config.config_str(), file=(model_dir / 'config.gin').open(mode='w'))

    model = build_model(input_shape=input_shape, output_channels=3, data_format=data_format)
    # model.summary()

    model.fit(
        data_gen(data_paths_train),
        epochs=epochs,
        steps_per_epoch=len(data_paths_train) // batch_size,
        validation_data=data_gen(data_paths_val),
        validation_steps=len(data_paths_val) // batch_size,
        callbacks=[
            k_callbacks.CSVLogger(filename=model_dir / 'log.csv'),
            k_callbacks.ModelCheckpoint(
                filepath=str(model_dir / 'model.hdf5'),
                verbose=1,
            ),
            k_callbacks.ModelCheckpoint(
                filepath=str(model_dir / 'model_best.hdf5'),
                verbose=1,
                save_best_only=True,
            ),
            k_callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: save_preds(
                    model=model,
                    data=data_paths_val,
                    model_dir=model_dir,
                    data_format=data_format,
                ),
            )
        ],
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        type=str,
    )
    args = parser.parse_args()

    gin.parse_config_file(args.config)

    # TODO
    from tensorflow.python.framework.ops import disable_eager_execution

    disable_eager_execution()
    train()
