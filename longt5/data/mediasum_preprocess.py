# Copyright 2022 The LongT5 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Converts MediaSum data into TF examples and writes out as TF records."""

import collections
import json
import os
from typing import List, Mapping, Sequence, Set, Tuple

from absl import app
from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string('input_path', None, 'Path to the input data.')
flags.DEFINE_string('output_path', None, 'Path to the output file.')


def get_id_splits() -> Tuple[Set[str], Set[str], Set[str]]:
  """Returns the id splits for train, test, and validate sets."""
  data_splits = []
  with open(
      os.path.join(FLAGS.input_path,
                   'mediasum_train_val_test_split.json')) as f:
    for r in f:
      data_splits.append(json.loads(r))
  return set(data_splits[0]['train']), set(data_splits[0]['test']), set(
      data_splits[0]['val'])


def get_data_splits() -> Tuple[List[Mapping[str, str]], List[Mapping[str, str]],
                               List[Mapping[str, str]]]:
  """Returns data split into train, test, and validate lists."""
  train_ids, test_ids, val_ids = get_id_splits()

  data = []
  with open(os.path.join(FLAGS.input_path, 'news_dialogue.json')) as f:
    for r in f:
      data.append(json.loads(r))

  train_data = []
  test_data = []
  val_data = []

  for value in data[0]:
    value_id = value['id']
    if value_id in test_ids:
      test_data.append(value)
    elif value_id in val_ids:
      val_data.append(value)
    elif value_id in train_ids:
      train_data.append(value)
    else:
      print(f'Invalid id: {value_id}')

  return train_data, test_data, val_data


def convert_and_write_tf_examples(data: List[Mapping[str, str]],
                                  filename: str) -> None:
  """Converts JSON data to TF example and write out to TF record.

  Args:
    data: Data in JSON format to be converted to tf examples.
    filename: Output filename to write data to.
  """
  tf_examples = []

  for value in data:
    speakers = value['speaker']
    utt = value['utt']
    line = []
    for s, u in zip(speakers, utt):
      line.append(s + ': ' + u)
    features = collections.OrderedDict()
    features['interview'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[' '.join(line).encode('utf-8')]))
    features['summary'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value['summary'].encode('utf-8')]))
    tf_examples.append(
        tf.train.Example(features=tf.train.Features(feature=features)))

  with tf.io.TFRecordWriter(os.path.join(FLAGS.output_path,
                                         filename)) as writer:
    for value in tf_examples:
      writer.write(value.SerializeToString())


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_data, test_data, val_data = get_data_splits()

  convert_and_write_tf_examples(train_data, 'train.tfrecords')
  convert_and_write_tf_examples(test_data, 'test.tfrecords')
  convert_and_write_tf_examples(val_data, 'val.tfrecords')


if __name__ == '__main__':
  app.run(main)
