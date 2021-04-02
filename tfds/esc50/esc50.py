"""esc50 dataset."""

import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """
The ESC-50 dataset is a labeled collection of 2000 environmental audio
recordings suitable for benchmarking methods of environmental sound
classification.

The dataset consists of 5-second-long recordings organized into 50 semantical
classes (with 40 examples per class) loosely arranged into 5 major categories.
"""

_CITATION = """
@data{DVN/YDEPUT_2015,
    author = {Karol J. Piczak},
    publisher = {Harvard Dataverse},
    title = {{ESC: Dataset for Environmental Sound Classification}},
    year = {2015},
    version = {V2},
    doi = {10.7910/DVN/YDEPUT},
    url = {https://doi.org/10.7910/DVN/YDEPUT}
}
"""


class Esc50(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for esc50 dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(esc50): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'audio': tfds.features.Audio(file_format='wav'),
                'category': tfds.features.ClassLabel(
                    names=['no', 'yes']
                ),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('audio', 'category'),
            homepage='https://github.com/karolpiczak/ESC-50',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(esc50): Downloads the data and defines the splits
        path = dl_manager.download_and_extract('https://github.com/karoldvl/ESC-50/archive/master.zip')

        # TODO(esc50): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train': self._generate_examples(path / 'train_imgs'),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(esc50): Yields (key, example) tuples from the dataset
        for f in path.glob('*.jpeg'):
            yield 'key', {
                'image': f,
                'label': 'yes',
            }
