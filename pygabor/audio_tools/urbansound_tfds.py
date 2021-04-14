import os
from pathlib import Path
from typing import Dict

import librosa
import pandas as pd
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import tensorflow_io as tfio


class UrbanSound8K(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'audio': tfds.features.Audio,
                'fdID': tfds.features.Tensor,
                'classID': tfds.features.ClassLabel(num_classes=10),
                'class': tfds.features.ClassLabel(
                    names=['air_conditioner', 'car_horn', 'children_playing',
                           'dog_barking', 'drilling', 'engine_idling',
                           'gun_shot', 'jackhammer', 'siren', 'street_mustic']
                )
            })
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager) -> Dict[tfds.splits_lib.Split, tfds.split_builder_lib.SplitGenerator]:
        path = dl_manager.download_and_extract(...)

        return {
            'fold1': self._generate_examples(path / 'audio/fold1'),
            'fold2': self._generate_examples(path / 'audio/fold2'),
            'fold3': self._generate_examples(path / 'audio/fold3'),
            'fold4': self._generate_examples(path / 'audio/fold4'),
            'fold5': self._generate_examples(path / 'audio/fold5'),
            'fold6': self._generate_examples(path / 'audio/fold6'),
            'fold7': self._generate_examples(path / 'audio/fold7'),
            'fold8': self._generate_examples(path / 'audio/fold8'),
            'fold9': self._generate_examples(path / 'audio/fold9'),
            'fold10': self._generate_examples(path / 'audio/fold10'),
        }

    def _generate_examples(self, path: tfds.core.ReadOnlyPath) -> tfds.split_builder_lib.SplitGenerator:
        ...

    # def __init__(self, csv_path: str, audio_path: str) -> None:
    #     self.csv_data = pd.read_csv(csv_path)
    #     self.audio_path = Path(audio_path)
