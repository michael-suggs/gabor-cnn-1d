from pathlib import Path
from random import shuffle
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple, Union, overload

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


Dataset = tf.data.Dataset


class TensorTuple(NamedTuple):
    """Named tuple holding information needed to make an audio tensor."""
    filename: str
    audio: tf.Tensor
    category: str
    target: int


def list_wav_files(path: Union[str, Path]) -> List[Path]:
    """Returns list of pathlib Path objects for each .wav in the path."""
    return [f for f in path.iterdir() if f.suffix == '.wav']


def load_wavs_to_tf(paths: List[Path], limit: int) -> Iterator[Tuple[str, tf.Tensor]]:
    audio_binaries = {f.name.split('.')[0]: tf.io.read_file(f) for f in paths}
    # limited_files = audio_binaries if limit is None else audio_binaries[:limit]

    i = 0
    for filename, wav in audio_binaries.items():
        if i == limit:
            break
        audio, _ = tf.audio.decode_wav(wav)
        yield filename, tf.squeeze(audio, axis=-1)
        i += 1


def wav_to_tensor(path: Path, label: Tuple[str, str]) -> TensorTuple:
    path_str = str(path.relative_to('.'))
    return TensorTuple(
        filename=str(path.name),
        audio=tf.audio.decode_wav(tf.io.read_file(path_str))[0],
        category=label[0],
        target=label[1],
    )


def get_audio(path: str) -> tf.Tensor:
    audio, _ = tf.audio.decode_wav(tf.io.read_file(path))
    return tf.reshape(audio, (1,-1))


def librosa_audio(path: str):
    samples, rate = librosa.load(path)
    samples = librosa.resample(samples, rate, 16000)
    return samples


@overload
def get_labels(
    paths: List[Path], datadir: Path, csv: Path, df: bool = True
) -> pd.DataFrame:
    ...


@overload
def get_labels(
    paths: List[Path], datadir: Path, csv: Path, df: bool = False
) -> Dict[str, Tuple[str, str]]:
    ...


def get_labels(
    paths: List[Path], datadir: Path, csv: Path, df: bool = True
) -> Union[pd.DataFrame, Dict[str, Tuple[str, str]]]:
    metadata = pd.read_csv(csv)
    metadata['filename'] = (
        metadata['filename'].map(lambda s: str(datadir.joinpath(s)))
    )
    if df:
        return metadata[['filename', 'category', 'target']]
    else:
        metadata = metadata.set_index('filename', drop=True)
        return {path.name: (metadata.loc[path.name, :].category,
                            metadata.loc[path.name, :].target)
                for path in paths}


def make_arrays(datadir: str, csv: str, test_size: float, fold: int = None):
    datadir = Path(datadir)
    meta = pd.read_csv(csv)
    meta = meta[meta.fold == fold][['filename', 'category']]
    meta['filename'] = meta.filename.map(lambda f: str(datadir.joinpath(f)))
    return train_test_np(meta, test_size)


def train_test_np(label_df: pd.DataFrame, test_size: float):
    X = np.asarray(list(map(
            lambda p: librosa_audio(p), list(label_df.filename)
    )))
    y = np.reshape(label_df['category'].to_numpy(), (-1,1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test


def train_test_datasets(label_df: pd.DataFrame, test_size: float) -> Tuple[Dataset, Dataset]:
    ntrain, ntest = (round(len(label_df) * (-test_size % 1)),
                     round(len(label_df) * test_size))

    audio_train = tf.data.Dataset.from_tensor_slices(
        label_df.iloc[:ntrain,:].filename).map(get_audio)
    audio_test = tf.data.Dataset.from_tensor_slices(
        label_df.iloc[-ntest:,:].filename).map(get_audio)
    label_train = tf.data.Dataset.from_tensor_slices(
        label_df.iloc[:ntrain,:].category)
    label_test = tf.data.Dataset.from_tensor_slices(
        label_df.iloc[-ntest:,:].category)

    train_ds = tf.data.Dataset.zip((audio_train, label_train))
    test_ds = tf.data.Dataset.zip((audio_test, label_test))

    return train_ds, test_ds


def tensor_train_test_split(
    tensors: List[TensorTuple], test_size: float, val_size: Optional[float] = None
) -> Tuple[Dataset, Dataset, Dataset, Dataset, Optional[Dataset], Optional[Dataset]]:
    shuffle(tensors)
    if val_size is None:
        ntrain, ntest = (round(len(tensors) * (1 - test_size)),
                         round(len(tensors) * test_size))

        X_train, y_train = zip(
            *map(lambda t: (t.audio, t.target), tensors[:ntrain]))
        X_test, y_test = zip(
            *map(lambda t: (t.audio, t.target), tensors[-ntest:]))
        X_train = Dataset.from_tensor_slices(X_train)
        y_train = Dataset.from_tensor_slices(
            tf.constant([l for l in y_train]))
            # tf.one_hot(y_train, max(y_train) + 1))
        X_test = Dataset.from_tensor_slices(X_test)
        y_test = Dataset.from_tensor_slices(
            tf.constant([l for l in y_test]))
            # tf.one_hot(y_test, max(y_test) + 1))

        return X_train, X_test, y_train, y_test

    else:
        ntrain = round(len(tensors) * (1 - test_size))
        ntest = round(len(tensors) * test_size)
        nval = round(len(tensors) * val_size)

        X_train, y_train = zip(
            *map(lambda t: (t.audio, t.target), tensors[:ntrain]))
        X_val, y_val = zip(
            *map(lambda t: (t.audio, t.target), tensors[ntrain:ntrain + nval]))
        X_test, y_test = zip(
            *map(lambda t: (t.audio, t.target), tensors[-ntest:]))
        X_train, y_train = Dataset.from_tensor_slices(
            X_train), Dataset.from_tensor_slices(y_train)
        X_val, y_val = Dataset.from_tensor_slices(
            X_val), Dataset.from_tensor_slices(y_val)
        X_test, y_test = Dataset.from_tensor_slices(
            X_test), Dataset.from_tensor_slices(y_test)

        return X_train, X_val, X_test, y_train, y_val, y_test


def make_dataset(
    datadir: Union[Path, str], metadir: Union[Path, str],
    test_size: float, val_size: Optional[float] = None
):
    datadir = datadir if isinstance(datadir, Path) else Path(datadir)
    metadir = metadir if isinstance(metadir, Path) else Path(metadir)

    wav_files = list_wav_files(datadir)
    labels = get_labels(wav_files, datadir, metadir)
    # tensors: List[TensorTuple] = [wav_to_tensor(
    #     path, labels[path.name]) for path in wav_files]
    return train_test_np(labels, test_size)
