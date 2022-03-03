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

"""Preprocessors for long T5."""

from pegasus.data import parsers
import seqio
import t5.data
import tensorflow.compat.v2 as tf


def _string_join(lst):
  # Join on space, but collapse consecutive spaces.
  out = tf.strings.join(lst, separator=' ')
  return tf.strings.regex_replace(out, r'\s+', ' ')


def _normalize_text(text):
  """Lowercase and remove quotes from a TensorFlow string."""
  text = tf.strings.lower(text)
  text = tf.strings.regex_replace(text, "'(.*)'", r'\1')
  return text


@seqio.map_over_dataset
def nq(x):
  """Convert NQ TF examples to a text2text pair.

  NQ produces examples with this form:
    {'id_': <id>, 'title': <title>, context': <article>, 'question': <question>,
     'answer': <answer> }
  This function will return examples of the format:
    {'inputs': 'question: <question> context: <article>',
     'targets': '<answer>',
     'id': <id>, 'question': <question>, 'context': <context>,
     'answers': [<n answers>]},

  Args:
    x: an example to process.

  Returns:
    A preprocessed example with the format listed above.
  """
  inputs = _string_join(['question:', x['question'], 'context:', x['context']])

  return {
      'inputs': inputs,
      'targets': x['answer'],
      'id': x['id_'],
      'context': x['context'],
      'question': x['question'],
      'answers': [x['answer']]
  }


@seqio.map_over_dataset
def triviaqa(x, ignore_web=True, include_title=True):
  """Convert TriviaQA TF examples to a text2text pair.

  TriviaQA produces examples with this form:
    {'entity_pages': {dict of wiki entities},
     'search_results': <dict of web search results>,
     'answer': {dict of all answers}, 'question': <question>,
     'question_id': <question_id>, 'question_source': <question_source>}

  This function will return examples of the format:
    {'inputs': 'question: <question> context: <article>',
     'targets': '<answer>',
     'id': <id>, 'question': <question>, 'context': <context>,
     'answers': [<n answers>]},

  Args:
    x: an example to process.
    ignore_web: whether to ignore the web context
    include_title: whether to include the title

  Returns:
    A preprocessed example with the format listed above.
  """

  question = _normalize_text(x['question'])

  wiki_context = [_normalize_text(x['entity_pages']['wiki_context'])]
  if include_title:
    # Append the title before each context.
    wiki_context = [_normalize_text(x['entity_pages']['title'])] + wiki_context
    wiki_context = tf.transpose(tf.stack(wiki_context))
  wiki_context = tf.strings.reduce_join(wiki_context, separator=' ')
  context = wiki_context

  if not ignore_web:
    web_context = [_normalize_text(x['search_results']['search_context'])]
    if include_title:
      # Append the title before each context.
      web_context = [_normalize_text(x['search_results']['title'])
                    ] + web_context
      web_context = tf.transpose(tf.stack(web_context))
    web_context = tf.strings.reduce_join(web_context, separator=' ')
    context = _string_join([wiki_context, web_context])

  inputs = _string_join(['question:', question, 'context:', context])
  targets = _normalize_text(x['answer']['value'])

  return {
      'inputs': inputs,
      'targets': targets,
      'id': x['question_id'],
      'context': context,
      'question': question,
      'answers': x['answer']['aliases']
  }


# Preprocessor for PEGASUS type pretraining.
# Sentences/words are masked/replaced with different strategies. Details at
# https://arxiv.org/abs/1912.08777
pegasus_parser, _ = parsers.string_features_for_pretraining_parser(
    vocab_filename='gs://t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model',
    encoder_type='sentencepiece_noshift',  # Matches tokenizer used by T5.
    max_input_len=4096,
    max_target_len=910,
    max_total_words=0,
    parser_strategy='dynamic_rouge',
    parser_masked_sentence_ratio=0.2,
    parser_masked_words_ratio=0,
    parser_mask_word_option_prob=[0.8, 0.1, 0.1],
    parser_mask_sentence_option_prob=[.9, 0, .1, 0],
    parser_rouge_ngrams_size=1,
    parser_rouge_metric_type='F',
    parser_rouge_compute_option='standard',
    # The stopwords file used is here: https://gist.github.com/sebleier/554280
    parser_rouge_stopwords_filename='',
    shift_special_token_id=t5.data.DEFAULT_EXTRA_IDS - 2,  # 2's for eos and pad
    mode='',
    parser_rouge_noise_ratio=.2,
    parser_dynamic_mask_min_ratio=.33,
    input_feature='inputs',
    pretrain_target_filter_min=0)


@seqio.map_over_dataset
def pegasus_parse(x):
  """Parses an example with the Pegasus parser.

  As input, method receives:
    {
      'inputs': '<sent1> <sent2> .... <sentn>'
      'targets': None
    }
  This function will return examples of the format:
    {
      'inputs': '<sent1> <mask> .... <sentn>'
      'targets': '<sent2>'
    }
  though the returned example will have been tokenized with SPM and will
  contain EOS id at the end of both inputs and targets (as is also done in T5).

  Args:
    x: an example to process.

  Returns:
    A preprocessed example, where some of the input is masked and copied to the
    target. These values will have been tokenized with SPM.
  """

  # Add key 'supervised' as required by Pegasus parser.
  x['supervised'] = tf.constant(False, dtype=tf.bool)
  # Parse the input. Pegasus parser will return with some of the input masked
  # and copied to target (all having been tokenized).
  parsed = pegasus_parser(x)
  # Adjust outputs from Pegasus parser to work with T5. This involves taking
  # the elements at index 0 (to get the right shape needed) and casting from
  # int64 to int32.
  return {
      'inputs': tf.cast(parsed['inputs'][0], tf.int32),
      'targets': tf.cast(parsed['targets'][0], tf.int32)
  }
