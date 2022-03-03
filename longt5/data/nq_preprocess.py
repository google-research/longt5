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

"""Converts NQ json lines to sstable file for long T5 fine-tuning."""

import collections
import json
import re

from absl import app
from absl import flags
import apache_beam as beam
import tensorflow as tf


FLAGS = flags.FLAGS
flags.DEFINE_string("input_path", None, "Path to the input data.")
flags.DEFINE_string("output_path", None, "Path to the output file.")
flags.DEFINE_bool(
    "include_html_tokens", False, "Whether to include html tokens.")
flags.DEFINE_bool(
    "use_full_document", False, "Whether to use full document as context.")


def parse_json_line(line):
  return json.loads(line)


def create_bytes_list_feature(value):
  try:
    feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
  except IOError:
    feat = tf.train.Feature(bytes_list=tf.train.BytesList(value=["none"]))
  return feat


def create_example(
    id_, title, context, question, answer, all_answers):
  """Create tf examples for the minspan task."""

  features = collections.OrderedDict()
  features["id_"] = create_bytes_list_feature([id_.encode("utf-8")])
  features["title"] = create_bytes_list_feature([title.encode("utf-8")])
  features["context"] = create_bytes_list_feature([context.encode("utf-8")])
  features["question"] = create_bytes_list_feature([question.encode("utf-8")])
  features["answer"] = create_bytes_list_feature([answer.encode("utf-8")])
  features["all_answers"] = create_bytes_list_feature(
      [answer.encode("utf-8") for answer in all_answers])

  return tf.train.Example(features=tf.train.Features(feature=features))


def _pad_punctuation(text):
  """Adds spaces around punctuation."""
  # Add space around punctuation.
  text = re.sub(r"(\W)", r" \1 ", text)
  # Collapse consecutive whitespace into one space.
  text = re.sub(r"\s+", " ", text)
  return text


def _has_short_answer(a):
  return bool(a["short_answers"])


def _is_yes_no_answer(a):
  return a["yes_no_answer"] in ("YES", "NO")


class ExampleFromJSON(beam.DoFn):
  """Extract example from json input."""

  def process(self, json_dict):
    del self  # unused for this method.

    # Question
    question = json_dict["question_text"].strip()

    # Title
    title = json_dict["document_title"].strip()

    # Context
    document_tokens = []

    if FLAGS.use_full_document:
      for t in json_dict["document_tokens"]:
        if bool(t["html_token"]) and not FLAGS.include_html_tokens:
          continue
        document_tokens.append(t["token"])
    else:
      for c in json_dict["long_answer_candidates"]:
        if not bool(c["top_level"]):
          continue
        for pos in range(c["start_token"], c["end_token"]):
          t = json_dict["document_tokens"][pos]
          if bool(t["html_token"]) and not FLAGS.include_html_tokens:
            continue
          document_tokens.append(t["token"])

    context = " ".join(document_tokens)

    # Answer -- train example has one annotation and dev example has up to five.
    all_answers = []
    for a in json_dict["annotations"]:
      answer_tokens = []
      if _has_short_answer(a):
        for sa in a["short_answers"]:
          for pos in range(sa["start_token"], sa["end_token"]):
            answer_tokens.append(json_dict["document_tokens"][pos]["token"])
          answer = " ".join(answer_tokens)
          all_answers.append(answer)
      elif _is_yes_no_answer(a):
        all_answers.append(a["yes_no_answer"])
      else:
        all_answers.append("NULL")
    all_answers = [_pad_punctuation(a) for a in all_answers]

    # pick the first non NULL answer if it exists
    answer = "NULL"
    for a in all_answers:
      if a != "NULL":
        answer = a
        break

    id_ = str(json_dict["example_id"])

    # Substitute newlines and tabs with spaces.
    context = re.sub(r"\n\t", " ", context)

    # Remove multiple spaces.
    context = _pad_punctuation(context)
    question = _pad_punctuation(question)

    yield id_, create_example(id_, title, context, question, answer,
                              all_answers)


def main(argv):
  del argv

  with beam.Pipeline() as p:
    _ = (
        p | "ReadData" >> beam.io.ReadFromText(
            FLAGS.input_path, validate=False)
        | "ParseJSON" >> beam.Map(parse_json_line)
        | "CreateExample" >> beam.ParDo(ExampleFromJSON())
        | "WriteToExample" >>  beam.io.tfrecordio.WriteToTFRecord(
            FLAGS.output_path,
            coder=beam.coders.ProtoCoder(tf.train.Example)))


if __name__ == "__main__":
  app.run(main)
