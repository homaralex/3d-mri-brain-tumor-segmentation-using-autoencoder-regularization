import argparse
import random
import glob  # For populating the list of files
import re  # For parsing the filenames (to know their modality)
from datetime import datetime
from pathlib import Path

import gin
import cv2
import wandb
from wandb.keras import WandbCallback
import matplotlib.pyplot as plt
import SimpleITK as sitk  # For loading the dataset
import numpy as np  # For data manipulation
import tensorflow.keras.callbacks as k_callbacks
from scipy.ndimage import zoom  # For resizing
from tensorflow.python.framework.ops import disable_eager_execution

from model import build_model  # For creating the model


def read_img(img_path):
    """
    Reads a .nii.gz image and returns as a numpy array.
    Added: transpose it to match the paper implementation dimension order.
    """
    img_path = str(img_path)
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path)).T


def resize(img, shape, mode='constant'):
    """
    Wrapper for scipy.ndimage.zoom suited for MRI images.
    """
    assert len(shape) == 3, "Can not have more than 3 dimensions"
    factors = (
        shape[0] / img.shape[0],
        shape[1] / img.shape[1],
        shape[2] / img.shape[2]
    )

    # Resize to the given shape
    return zoom(img, factors, mode=mode)


@gin.configurable(allowlist=['z_score'])
def preprocess(
        img_path,
        resized_path=None,
        out_shape=None,
        augment=True,
        z_score=False,
):
    if resized_path is None or not resized_path.exists():
        img = read_img(img_path)
        # normalize the image (based on non-zero voxels)
        mean = img[img != 0].mean()
        std = img[img != 0].std()

        if out_shape is not None:
            img = resize(img, out_shape, mode='constant')
            if resized_path is not None:
                resized_path.parent.mkdir(exist_ok=True)
                np.savez(
                    resized_path,
                    img=img,
                    mean=mean,
                    std=std,
                )
    else:
        npz = np.load(resized_path)
        img, mean, std = npz['img'], npz['mean'], npz['std']

    if augment:
        # random intensity and scale shifts
        mean += std * np.random.uniform(-.1, .1)
        std *= np.random.uniform(.9, 1.1)

    img = (img - mean) / std
    if z_score:
        img = (img - img.min()) / (img.max() - img.min())

    return img


def preprocess_label(
        img_path,
        resized_path=None,
        out_shape=None,
        mode='nearest',
):
    """
    Separates out the 3 labels from the segmentation provided, namely:
    GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2))
    and the necrotic and non-enhancing tumor core (NCR/NET — label 1)
    """
    if resized_path is None or not resized_path.exists():
        img = read_img(img_path)

        ncr = img == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET)
        ed = img == 2  # Peritumoral Edema (ED)
        et = img == 4  # GD-enhancing Tumor (ET)

        if out_shape is not None:
            ncr = resize(ncr, out_shape, mode=mode).astype('uint8')
            ed = resize(ed, out_shape, mode=mode).astype('uint8')
            et = resize(et, out_shape, mode=mode).astype('uint8')

            kernel = np.ones((3, 3))
            ncr = np.array([cv2.morphologyEx(a, cv2.MORPH_CLOSE, kernel, iterations=2) for a in ncr])
            ed = np.array([cv2.morphologyEx(a, cv2.MORPH_CLOSE, kernel, iterations=2) for a in ed])
            et = np.array([cv2.morphologyEx(a, cv2.MORPH_CLOSE, kernel, iterations=2) for a in et])

        img = np.array([ncr, ed, et], dtype=np.float32)[None, ...]

        if resized_path is not None:
            resized_path.parent.mkdir(exist_ok=True)
            np.savez(resized_path, img=img)
    else:
        img = np.load(resized_path)['img']

    return img


def get_paths(path_root):
    # Get a list of files for all modalities individually
    t1 = glob.glob(path_root + '*GG/*/*t1.nii.gz')
    t2 = glob.glob(path_root + '*GG/*/*t2.nii.gz')
    flair = glob.glob(path_root + '*GG/*/*flair.nii.gz')
    t1ce = glob.glob(path_root + '*GG/*/*t1ce.nii.gz')
    seg = glob.glob(path_root + '*GG/*/*seg.nii.gz')  # Ground Truth

    pat = re.compile('.*_(\w*)\.nii\.gz')

    return [{
        pat.findall(item)[0]: Path(item)
        for item in items
    } for items in list(zip(t1, t2, t1ce, flair, seg))]


@gin.configurable(denylist=['data_paths'])
def data_gen(
        data_paths,
        batch_size,
        input_shape,
        modalities,
        data_format,
        augment=True,
        save_resized=True,
):
    xs, ys = [], []
    out_shape = input_shape[1:] if data_format == 'channels_first' else input_shape[:-1]

    def yield_batch(xs, ys):
        xs, ys = np.array(xs), [np.array([y[0] for y in ys]), np.array([y[1] for y in ys])]

        if data_format == 'channels_last':
            xs, ys = np.moveaxis(xs, 1, -1), [np.moveaxis(ys[0], 1, -1), np.moveaxis(ys[1], 1, -1)]

        # fake output for the kld loss
        ys.append(np.zeros((ys[0].shape[0], 1)))

        return xs, ys

    while True:
        shuffled_paths = random.sample(data_paths, len(data_paths))
        for imgs in shuffled_paths:
            try:
                resized_dir = imgs['seg'].parent / f'resized_{"_".join(str(s) for s in out_shape)}'
                x = np.array([
                    preprocess(
                        img_path=imgs[m],
                        out_shape=out_shape,
                        resized_path=(resized_dir / imgs[m].name).with_suffix('').with_suffix(
                            '.npz') if save_resized else None,
                        augment=augment,
                    ) for m in modalities],
                    dtype=np.float32,
                )
                y = preprocess_label(
                    img_path=imgs['seg'],
                    out_shape=out_shape,
                    resized_path=(resized_dir / imgs['seg'].name).with_suffix('').with_suffix(
                        '.npz') if save_resized else None,
                ).squeeze()
                # we have to do axis flipping here to be consistent with all modalities
                if augment:
                    for ax_id in range(len(x.shape) - 1):
                        if random.getrandbits(1):
                            x = np.flip(x, axis=ax_id + 1)
                            y = np.flip(y, axis=ax_id + 1)

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


def save_preds(
        model,
        data,
        model_dir,
        data_format,
):
    data, y = next(data_gen(data, 1, augment=False))

    ret = model.predict(data)

    # take the middle slice
    slice_idx = ret[1].shape[1] // 2
    if data_format == 'channels_first':
        plt.imshow(ret[0][0][0][slice_idx], cmap='Greys_r')
        plt.savefig(model_dir / 'segmentation.png')
        plt.imshow(ret[1][0][0][slice_idx], cmap='Greys_r')
        plt.savefig(model_dir / 'reconstruction.png')
        plt.imshow(data[0][0][slice_idx], cmap='Greys_r')
        plt.savefig(model_dir / 'original.png')
    else:
        plt.imshow(ret[0][0][:, :, :, 0][slice_idx], cmap='Greys_r')
        plt.savefig(model_dir / 'segmentation.png')
        plt.imshow(ret[1][0][:, :, :, 0][slice_idx], cmap='Greys_r')
        plt.savefig(model_dir / 'reconstruction.png')
        plt.imshow(data[0][:, :, :, 0][slice_idx], cmap='Greys_r')
        plt.savefig(model_dir / 'original.png')
        plt.imshow(y[0][0][:, :, :, 0][slice_idx], cmap='Greys_r')
        plt.savefig(model_dir / 'original_seg.png')


def wandb_callback(
        model,
        data_gen,
        data_format,
):
    segs, recs = [], []
    while len(segs) < 4:
        x, y = next(data_gen)

        # take the middle slice
        slice_idx = x[0].shape[1] // 2
        if data_format == 'channels_first':
            seg_orig = y[0][0][0][slice_idx]
            if seg_orig.sum() == 0:
                continue

            preds = model.predict(x)
            rec = preds[1][0][0][slice_idx]
            orig = x[0][0][slice_idx]
            seg = preds[0][0][0][slice_idx]
        else:
            seg_orig = y[0][0][:, :, :, 0][slice_idx]
            if seg_orig.sum() == 0:
                continue

            preds = model.predict(x)
            rec = preds[1][0][:, :, :, 0][slice_idx]
            orig = x[0][:, :, :, 0][slice_idx]
            seg = preds[0][0][:, :, :, 0][slice_idx]

        recs.append(wandb.Image(rec))
        mask_img = wandb.Image(orig, masks={
            "predictions": {
                "mask_data": seg,
            },
            "ground_truth": {
                "mask_data": seg_orig,
            },
        })
        segs.append(mask_img)

    wandb.log({"reconstructions": recs})
    wandb.log({"segmentations": segs})


@gin.configurable
def train(
        brats_train_dir=gin.REQUIRED,
        brats_val_dir=None,
        val_ratio=.2,
        model_name='ResNet3DVAE_Brats',
        input_shape=(160, 192, 128),
        data_format='channels_last',
        z_score=False,
        modalities=('t1', 't2', 't1ce', 'flair'),
        batch_size=1,
        epochs=300,
        wandb_project=None,
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
    gin.bind_parameter('preprocess.z_score', z_score)

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

    model = build_model(
        input_shape=input_shape,
        output_channels=3,
        data_format=data_format,
        z_score=z_score,
    )

    callbacks = [
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
        ),
    ]

    if wandb_project is not None:
        wandb_config = {}
        for line in gin.config.config_str().split('\n'):
            if len(line.strip()) > 0 and line[0] != '#':
                split_idx = line.find('=')
                key, val = line[:split_idx].strip(), line[split_idx + 1:].strip()
                wandb_config[key] = val
        wandb.init(project=wandb_project, config=wandb_config)
        wandb.run.name = Path(args.config).with_suffix('').name + '-' + wandb.run.id
        # replace last callback with wandb
        callbacks[-1] = WandbCallback(save_model=False)
        callbacks.append(
            k_callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: wandb_callback(
                    model=model,
                    data_gen=data_gen(data_paths_val, augment=False),
                    data_format=data_format,
                ),
            )
        )

    model.fit(
        data_gen(data_paths_train),
        epochs=epochs,
        steps_per_epoch=len(data_paths_train) // batch_size,
        validation_data=data_gen(data_paths_val, augment=False),
        validation_steps=len(data_paths_val) // batch_size,
        callbacks=callbacks,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        type=str,
    )
    args = parser.parse_args()

    gin.parse_config_file(args.config)

    disable_eager_execution()
    train()
