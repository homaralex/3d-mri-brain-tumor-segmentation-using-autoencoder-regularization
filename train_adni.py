import argparse
import random
from datetime import datetime
from pathlib import Path

import gin
import pandas as pd
from skimage import exposure

import wandb
from wandb.keras import WandbCallback
import numpy as np
from scipy.ndimage import rotate
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
        # random rotation
        axes = np.random.choice(
            (0, 1, 2),
            size=2,
            replace=False,
        )
        angle = np.random.randint(0, 6)
        img = rotate(
            img,
            angle=angle,
            axes=axes,
            prefilter=False,
            reshape=False,
        )

        # random gamma correction [.5, 3]
        gamma = np.random.uniform(.5, 3)
        img = exposure.adjust_gamma(img.astype(float), gamma)

        # random translation
        translated = np.zeros_like(img)
        shifts = np.random.randint(-5, 5, size=(2,))
        w, h = img.shape[:2]
        cropped = img[
                  shifts[0] if shifts[0] > 0 else None:w + shifts[0] if shifts[0] < 0 else None,
                  shifts[1] if shifts[1] > 0 else None:h + shifts[1] if shifts[1] < 0 else None,
                  ]
        translated[
        -shifts[0] if shifts[0] < 0 else None:cropped.shape[0] if shifts[0] > 0 else None,
        -shifts[1] if shifts[1] < 0 else None:cropped.shape[1] if shifts[1] > 0 else None,
        ] = cropped
        img = translated

        # axis flipping
        for ax_id in range(len(img.shape) - 1):
            if random.getrandbits(1):
                img = np.flip(img, axis=ax_id + 1)

    img = (img - mean) / std
    if z_score:
        img = (img - img.min()) / (img.max() - img.min())

    return np.expand_dims(img, 0)


@gin.configurable(denylist=['df', 'augment'])
def data_gen(
        df,
        batch_size,
        input_shape,
        data_format,
        augment=False,
        binary=False,
        max_val=False,
        add_covariates=None,
):
    target_col_name = 'CDGLOBAL_MAX' if max_val else 'CDGLOBAL'
    xs, ys = [], []

    def yield_batch(xs, ys):
        xs = np.array(xs) if add_covariates is None else [np.array([x[0] for x in xs]), np.array([x[1] for x in xs])]
        ys = [np.array([y[0] for y in ys]), np.array([y[1] for y in ys])]

        if data_format == 'channels_last':
            ys = [np.moveaxis(ys[0], 1, -1), ys[1]]
            xs = np.moveaxis(xs, 1, -1) if add_covariates is None else [np.moveaxis(xs[0], 1, -1), xs[1]]

        # fake output for the kld loss
        ys.append(np.zeros((ys[0].shape[0], 1)))

        return xs, ys

    filename = preprocessed_filename(input_shape)
    while True:
        for _, row in df.sample(frac=1).iterrows():
            try:
                x = img = preprocess(Path(row[SCAN_DIR_COL_NAME]) / filename, augment=augment)
                if add_covariates is not None:
                    x = [x, row[add_covariates].values]

                y = int(row[target_col_name] >= .5) if binary else row[target_col_name]
            except Exception as e:
                print(f'Exception while loading: {row[SCAN_DIR_COL_NAME]}, skipping...\n Exception:\n{str(e)}')
                continue

            xs.append(x)
            # return img as well for the VAE reconstruction loss
            ys.append([img, y])

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
    for i in range(3):
        for _ in range(4):
            x, y = next(data_gen)
            orig_scans = y[0]
            preds = model.predict(x)
            if data_format == 'channels_first':
                slice_idx = orig_scans[0].shape[i + 1] // 2
                idxs = [0, slice(None), slice(None), slice(None)]
                idxs[i + 1] = slice_idx

                rec = preds[0][0][slice_idx]
                orig = orig_scans[0][slice_idx]
            else:
                slice_idx = orig_scans[0].shape[i] // 2
                idxs = [slice(None), slice(None), slice(None), 0]
                idxs[i] = slice_idx

                rec = preds[0][0][idxs]
                orig = orig_scans[0][idxs]

            origs.append(wandb.Image(orig))
            recs.append(wandb.Image(rec))

    wandb.log({'original': origs})
    wandb.log({'reconstructions': recs})


def get_dfs(
        data_root,
        df_name,
        input_shape,
        val_ratio,
        max_val,
        balance_val,
        balance_ad_cn,
        add_covariates,
):
    df = pd.read_pickle(data_root / df_name)
    df = df.loc[df['Sex'] != 'X']
    df['Sex'] = df['Sex'].astype('category').cat.codes

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

    if balance_ad_cn:
        # keep the same number of CN and AD subjects in the data frame
        df_grouped = df.groupby('Group')
        df_grouped_nunique = df_grouped['Subject'].nunique()
        remove_subjs = np.random.choice(
            df_grouped['Subject'].unique()['CN'],
            df_grouped_nunique['CN'] - df_grouped_nunique['AD'],
        )
        df = df.loc[~df['Subject'].isin(remove_subjs)]

    num_subjects = len(df['Subject'].unique())
    target_col_name = 'CDGLOBAL_MAX' if max_val else 'CDGLOBAL'
    while True:
        # try to keep an equal ratio of AD subjects in both subsets of data
        val_subjects = np.random.choice(df['Subject'].unique(), int(val_ratio * num_subjects))
        df_val = df.loc[df['Subject'].isin(val_subjects)]
        df_train = df.loc[~df['Subject'].isin(val_subjects)]

        mean_ad_val = (df_val[target_col_name].values >= .5).mean()
        mean_ad_train = (df_train[target_col_name].values >= .5).mean()

        if not balance_val or np.abs(mean_ad_val - mean_ad_train) < .01:
            print(f'Mean AD: Train: {mean_ad_train:.2f} Val: {mean_ad_val:.2f}')
            df_train_grouped = df_train.groupby('Group')['Subject'].nunique()
            df_val_grouped = df_val.groupby('Group')['Subject'].nunique()
            print(f'Num AD: Train: {df_train_grouped["AD"]} Val: {df_val_grouped["AD"]}')
            print(f'Num CN: Train: {df_train_grouped["CN"]} Val: {df_val_grouped["CN"]}')
            break

    if add_covariates is not None:
        for cov_name in add_covariates:
            mu, std = df_train[cov_name].mean(), df_train[cov_name].std()
            df_train[cov_name] = (df_train[cov_name] - mu) / std
            df_val[cov_name] = (df_val[cov_name] - mu) / std

    return df_train, df_val


@gin.configurable
def train(
        data_root=gin.REQUIRED,
        df_name='df.pkl',
        val_ratio=.2,
        model_name='VAE_REG_ADNI',
        input_shape=(96, 96, 96),
        data_format='channels_last',
        z_score=False,
        binary=False,
        augment=False,
        max_val=False,
        balance_val=True,
        balance_ad_cn=False,
        add_covariates=None,
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
    gin.bind_parameter('data_gen.binary', binary)
    gin.bind_parameter('data_gen.max_val', max_val)
    gin.bind_parameter('data_gen.add_covariates', add_covariates)
    gin.bind_parameter('preprocess.z_score', z_score)

    data_root = Path(data_root)
    assert data_root.exists()
    df_train, df_val = get_dfs(
        data_root=data_root,
        df_name=df_name,
        input_shape=input_shape,
        val_ratio=val_ratio,
        max_val=max_val,
        balance_val=balance_val,
        balance_ad_cn=balance_ad_cn,
        add_covariates=add_covariates,
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
        binary=binary,
        add_covariates=add_covariates,
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
        data_gen(df_train, augment=augment),
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
