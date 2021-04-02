"""urbansound8k dataset."""

import tensorflow_datasets as tfds

# TODO(urbansound8k): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from
10 classes:

    - air_conditioner
    - car_horn
    - children_playing
    - dog_bark
    - drilling
    - enginge_idling
    - gun_shot
    - jackhammer
    - siren
    - street_music.

The classes are drawn from the [urban sound taxonomy](https://urbansounddataset.weebly.com/taxonomy.html).
For a detailed description of the dataset and how it was compiled please refer
to our [paper](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_urbansound_acmmm14.pdf).

All excerpts are taken from field recordings uploaded to [www.freesound.org](www.freesound.org).
The files are pre-sorted into ten folds (folders named fold1-fold10) to help
in the reproduction of and comparison with the automatic classification results
reported in the article above.
"""

_CITATION = """
@inproceedings{Salamon:UrbanSound:ACMMM:14,
    Address = {Orlando, FL, USA},
    Author = {Salamon, J. and Jacoby, C. and Bello, J. P.},
    Booktitle = {22nd {ACM} International Conference on Multimedia (ACM-MM'14)},
    Month = {Nov.},
    Pages = {1041--1044},
    Title = {A Dataset and Taxonomy for Urban Sound Research},
    Year = {2014}
}
"""


class Urbansound8k(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for urbansound8k dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(urbansound8k): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'audio': tfds.features.Audio,
                'fdID': tfds.features.Tensor,
                'classID': tfds.features.ClassLabel(num_classes=10),
                'class': tfds.features.ClassLabel(
                    names=['air_conditioner', 'car_horn', 'children_playing',
                           'dog_barking', 'drilling', 'engine_idling',
                           'gun_shot', 'jackhammer', 'siren', 'street_mustic']
                )
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('audio', 'class'),  # Set to `None` to disable
            homepage='https://urbansounddataset.weebly.com/urbansound8k.html',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(urbansound8k): Downloads the data and defines the splits
        path = dl_manager.download_and_extract('https://todo-data-url')

        # TODO(urbansound8k): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train': self._generate_examples(path / 'train_imgs'),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(urbansound8k): Yields (key, example) tuples from the dataset
        for f in path.glob('*.jpeg'):
            yield 'key', {
                'image': f,
                'label': 'yes',
            }
