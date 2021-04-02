"""urbansound8k dataset."""

import tensorflow_datasets as tfds
from . import urbansound8k


class Urbansound8kTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for urbansound8k dataset."""
  # TODO(urbansound8k):
  DATASET_CLASS = urbansound8k.Urbansound8k
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
