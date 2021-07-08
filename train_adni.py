import argparse
import random
from datetime import datetime
from pathlib import Path

import gin
import pandas as pd
import wandb
from wandb.keras import WandbCallback
import numpy as np
import tensorflow.keras.callbacks as k_callbacks
from tensorflow.python.framework.ops import disable_eager_execution

from model_adni import vae_reg

SCAN_DIR_COL_NAME = 'scan_dir'


def preprocessed_filename(input_shape):
    return f'masked_{"_".join(str(s) for s in input_shape)}.npy'


@gin.configurable(allowlist=['z_score'])
def preprocess(
        img_path,
        augment=False,
        z_score=False,
):
    img = np.load(img_path)
    mean, std = img.mean(), img.std()

    if augment:
        # random intensity and scale shifts
        mean += std * np.random.uniform(-.1, .1)
        std *= np.random.uniform(.9, 1.1)

        for ax_id in range(len(img.shape) - 1):
            if random.getrandbits(1):
                img = np.flip(img, axis=ax_id + 1)

    img = (img - mean) / std
    if z_score:
        img = (img - img.min()) / (img.max() - img.min())

    return np.expand_dims(img, 0)


@gin.configurable(denylist=['df'])
def data_gen(
        df,
        batch_size,
        input_shape,
        data_format,
        augment=False,
):
    xs, ys = [], []

    def yield_batch(xs, ys):
        xs, ys = np.array(xs), [np.array([y[0] for y in ys]), np.array([y[1] for y in ys])]

        if data_format == 'channels_last':
            xs, ys = np.moveaxis(xs, 1, -1), [np.moveaxis(ys[0], 1, -1), ys[1]]

        # fake output for the kld loss
        ys.append(np.zeros((ys[0].shape[0], 1)))

        return xs, ys

    filename = preprocessed_filename(input_shape)
    while True:
        for _, row in df.sample(frac=1).iterrows():
            try:
                x = preprocess(Path(row[SCAN_DIR_COL_NAME]) / filename, augment=augment)
                y = row['CDGLOBAL']
            except Exception as e:
                print(f'Exception while loading: {row[SCAN_DIR_COL_NAME]}, skipping...\n Exception:\n{str(e)}')
                continue

            xs.append(x)
            # return x as well for the VAE reconstruction loss
            ys.append([x, y])

            if len(xs) == batch_size:
                yield yield_batch(xs, ys)
                xs, ys = [], []

        # in case of the last batch being smaller than batch_size
        if len(xs) > 0:
            yield yield_batch(xs, ys)


def wandb_callback(
        model,
        data_gen,
        data_format,
):
    origs, recs = [], []
    while len(recs) < 4:
        x, y = next(data_gen)

        # take the middle slice
        slice_idx = x[0].shape[1] // 2
        if data_format == 'channels_first':
            preds = model.predict(x)
            rec = preds[0][0][0][slice_idx]
            orig = x[0][0][slice_idx]
        else:
            preds = model.predict(x)
            rec = preds[0][0][slice_idx, :, :, 0]
            orig = x[0][slice_idx, :, :, 0]

        origs.append(wandb.Image(orig))
        recs.append(wandb.Image(rec))

    wandb.log({'original': origs})
    wandb.log({'reconstructions': recs})


def get_dfs(data_root, input_shape, val_ratio):
    df = pd.read_pickle(data_root / 'df.pkl')

    scan_dirs = set(p.parent for p in data_root.rglob(preprocessed_filename(input_shape)))

    rows = []
    for scan_dir in scan_dirs:
        for scan_file in scan_dir.iterdir():
            if scan_file.name.endswith('.dcm'):
                image_data_id = scan_file.with_suffix('').name.split('_')[-1]
                row = df.loc[df['Image Data ID'] == image_data_id]
                row[SCAN_DIR_COL_NAME] = str(scan_dir)
                rows.append(row)
                break

    df = pd.concat(rows)

    num_subjects = len(df['Subject'].unique())
    val_subjects = np.random.choice(df['Subject'].unique(), int(val_ratio * num_subjects))
    df_val = df.loc[df['Subject'].isin(val_subjects)]
    df_train = df.loc[~df['Subject'].isin(val_subjects)]

    return df_train, df_val


@gin.configurable
def train(
        data_root=gin.REQUIRED,
        val_ratio=.2,
        model_name='VAE_REG_ADNI',
        input_shape=(96, 96, 96),
        data_format='channels_last',
        z_score=False,
        batch_size=1,
        epochs=300,
        wandb_project=None,
        # for debugging purposes
        max_samples=None,
):
    assert len(input_shape) == 3
    gin.bind_parameter('data_gen.input_shape', input_shape)
    gin.bind_parameter('data_gen.batch_size', batch_size)
    gin.bind_parameter('data_gen.data_format', data_format)
    gin.bind_parameter('preprocess.z_score', z_score)

    data_root = Path(data_root)
    assert data_root.exists()
    df_train, df_val = get_dfs(
        data_root=data_root,
        input_shape=input_shape,
        val_ratio=val_ratio,
    )

    df_train, df_val = df_train.iloc[:max_samples], df_val.iloc[:max_samples]
    print(f'Train samples: {len(df_train)}\nVal samples: {len(df_val)}')

    model_dir = Path('models') / model_name / datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    model_dir.mkdir(exist_ok=True, parents=True)
    print(f'Model dir: {str(model_dir)}')
    # save the gin config to file
    print(gin.config.config_str(), file=(model_dir / 'config.gin').open(mode='w'))

    model = vae_reg(
        input_shape=input_shape,
        data_format=data_format,
        z_score=z_score,
    )
    model.summary()

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
                    data_gen=data_gen(df_val, augment=False),
                    data_format=data_format,
                ),
            )
        )

    model.fit(
        data_gen(df_train),
        epochs=epochs,
        steps_per_epoch=len(df_train) // batch_size,
        validation_data=data_gen(df_val, augment=False),
        validation_steps=len(df_train) // batch_size,
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
