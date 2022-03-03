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

"""Add Tasks to registry."""
import functools
import os

from longt5 import preprocessors as longt5_preprocessors

import seqio
import t5.data
from t5.evaluation import metrics
import tensorflow.compat.v1 as tf

MEDIASUM_DATA_DIR = "/path/to/processed/mediasum/"
NQ_DATA_DIR = "/path/to/processed/nq/"

NQ_FEATURES = {
    "id_": tf.io.FixedLenFeature([], dtype=tf.string),
    "title": tf.io.FixedLenFeature([], dtype=tf.string),
    "context": tf.io.FixedLenFeature([], dtype=tf.string),
    "question": tf.io.FixedLenFeature([], dtype=tf.string),
    "answer": tf.io.FixedLenFeature([], dtype=tf.string),
}

DEFAULT_TEMPERATURE = 1.0 / 0.3
DEFAULT_MIX_RATE = functools.partial(
    t5.data.rate_num_examples, temperature=DEFAULT_TEMPERATURE)

DEFAULT_VOCAB = t5.data.get_default_vocabulary()
DEFAULT_OUTPUT_FEATURES = {
    "inputs":
        seqio.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True, required=False),
    "targets":
        seqio.Feature(vocabulary=DEFAULT_VOCAB, add_eos=True)
}

# =========================== Pre-training Tasks/Mixtures ======================
for mean_value in (3, 5, 10, 20):
  seqio.TaskRegistry.add(
      "c4_v220_span_corruption_{}_mean_noise".format(mean_value),
      source=seqio.TfdsDataSource(tfds_name="c4/en:2.2.0"),
      preprocessors=[
          functools.partial(
              seqio.preprocessors.rekey,
              key_map={
                  "inputs": None,
                  "targets": "text"
              }),
          seqio.preprocessors.tokenize,
          seqio.CacheDatasetPlaceholder(),
          functools.partial(
              t5.data.preprocessors.span_corruption,
              mean_noise_span_length=mean_value),
          seqio.preprocessors.append_eos_after_trim,
      ],
      output_features=DEFAULT_OUTPUT_FEATURES,
      metric_fns=[])

seqio.TaskRegistry.add(
    "c4_v220_pegasus_parser",
    source=seqio.TfdsDataSource(tfds_name="c4/en:2.2.0"),
    preprocessors=[
        functools.partial(
            t5.data.preprocessors.rekey,
            key_map={
                "inputs": "text",
                "targets": "text",
            }),
        seqio.CacheDatasetPlaceholder(),
        longt5_preprocessors.pegasus_parse,
        # Pegasus parser adds EOS id, so no need for using seqio methods.
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    metric_fns=[])

seqio.MixtureRegistry.add(
    "c4_v220_pegasus_span_corruption", [
        "c4_v220_pegasus_parser",
        "c4_v220_span_corruption_3_mean_noise",
    ],
    default_rate=1.0)

# =========================== Fine-tuning Tasks/Mixtures =======================
# ----- QA tasks -----
# ----- NaturalQuestions (NQ) -----
seqio.TaskRegistry.add(
    "nq",
    source=seqio.TFExampleDataSource(
        reader_cls=tf.data.TFRecordDataset,
        split_to_filepattern={
            "train":
                os.path.join(NQ_DATA_DIR, "nq-train-????[0-8]-of-?????"),
            "validation":
                os.path.join(NQ_DATA_DIR, "nq-train-????9-of-?????"),
            "test":
                os.path.join(NQ_DATA_DIR, "nq-dev-?????-of-?????")
        },
        feature_description=NQ_FEATURES,
    ),
    preprocessors=[
        longt5_preprocessors.nq,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=t5.data.postprocessors.qa,
    metric_fns=[metrics.squad],
    output_features=DEFAULT_OUTPUT_FEATURES)

# ----- TriviaQA-----
seqio.TaskRegistry.add(
    "trivia_qa_full",
    source=seqio.TfdsDataSource(
        tfds_name="trivia_qa/rc:1.1.0",
        splits={
            "train": "train[:78785]",  # ~90%, matches numbers used by ORQA
            "validation": "train[78785:]",  # ~10%, matches numbers used by ORQA
            "test": "validation"
        },
    ),
    preprocessors=[
        longt5_preprocessors.triviaqa,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=t5.data.postprocessors.qa,
    metric_fns=[metrics.trivia_qa],
    output_features=DEFAULT_OUTPUT_FEATURES)

# This tasks trains on combined train and validation splits.
seqio.TaskRegistry.add(
    "trivia_qa_full_test",
    source=seqio.TfdsDataSource(
        tfds_name="trivia_qa/rc:1.1.0",
        splits={
            "train": "train+validation",
            "test": "test"
        },
    ),
    preprocessors=[
        longt5_preprocessors.triviaqa,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    postprocess_fn=t5.data.postprocessors.qa,
    metric_fns=[metrics.trivia_qa],
    output_features=DEFAULT_OUTPUT_FEATURES)

# ----- Summarization -----
seqio.TaskRegistry.add(
    "arxiv_summarization",
    source=seqio.TfdsDataSource(tfds_name="scientific_papers/arxiv:1.1.1"),
    preprocessors=[
        functools.partial(
            t5.data.preprocessors.summarize,
            article_key="article",
            summary_key="abstract"),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[metrics.rouge],
    output_features=DEFAULT_OUTPUT_FEATURES)

seqio.TaskRegistry.add(
    "pubmed_summarization",
    source=seqio.TfdsDataSource(tfds_name="scientific_papers/pubmed:1.1.1"),
    preprocessors=[
        functools.partial(
            t5.data.preprocessors.summarize,
            article_key="article",
            summary_key="abstract"),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[metrics.rouge],
    output_features=DEFAULT_OUTPUT_FEATURES)

seqio.TaskRegistry.add(
    "patent_summarization",
    source=seqio.TfdsDataSource(tfds_name="big_patent/all:2.1.2"),
    preprocessors=[
        functools.partial(
            t5.data.preprocessors.summarize,
            article_key="description",
            summary_key="abstract"),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[metrics.rouge],
    output_features=DEFAULT_OUTPUT_FEATURES)

seqio.TaskRegistry.add(
    "patent_summarization_reduced",
    source=seqio.TfdsDataSource(
        tfds_name="big_patent/all:2.1.2",
        splits={
            "train": "train",
            "validation": "validation[:1%]",
            "test": "test",
        }),
    preprocessors=[
        functools.partial(
            t5.data.preprocessors.summarize,
            article_key="description",
            summary_key="abstract"),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[metrics.rouge],
    output_features=DEFAULT_OUTPUT_FEATURES)

seqio.TaskRegistry.add(
    "cnn_summarization",
    source=seqio.TfdsDataSource(tfds_name="cnn_dailymail:3.1.0"),
    preprocessors=[
        functools.partial(
            t5.data.preprocessors.summarize,
            article_key="article",
            summary_key="highlights"),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[metrics.rouge],
    output_features=DEFAULT_OUTPUT_FEATURES)

seqio.TaskRegistry.add(
    "multi_news_summarization",
    source=seqio.TfdsDataSource(tfds_name="multi_news:1.0.0"),
    preprocessors=[
        functools.partial(
            t5.data.preprocessors.summarize,
            article_key="document",
            summary_key="summary"),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[metrics.rouge],
    output_features=DEFAULT_OUTPUT_FEATURES)

seqio.TaskRegistry.add(
    "media_summarization",
    source=seqio.TFExampleDataSource(
        split_to_filepattern={
            "train":
                os.path.join(MEDIASUM_DATA_DIR, "train.tfrecords"),
            "test":
                os.path.join(MEDIASUM_DATA_DIR, "test.tfrecords"),
            "validation":
                os.path.join(MEDIASUM_DATA_DIR, "val.tfrecords"),
        },
        feature_description={
            "interview": tf.FixedLenFeature([], tf.string),
            "summary": tf.FixedLenFeature([], tf.string),
        },
    ),
    preprocessors=[
        functools.partial(
            t5.data.preprocessors.summarize,
            article_key="interview",
            summary_key="summary"),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    metric_fns=[metrics.rouge],
    output_features=DEFAULT_OUTPUT_FEATURES)
