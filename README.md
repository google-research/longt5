# LongT5: Efficient Text-To-Text Transformer for Long Sequences

LongT5 is an extension of the [T5 model](https://github.com/google-research/text-to-text-transfer-transformer) that handles long sequence inputs more efficiently. We integrated attention ideas from long-input transformers [ETC](https://arxiv.org/abs/2004.08483),and adopted pre-training strategies from summarization pre-training [PEGASUS](https://arxiv.org/abs/1912.08777) into  the scalable T5 architecture. The result is a new attention mechanism we call Transient Global(TGlobal), which  mimics ETC’s local/globalattention mechanism, but without requiring additional side-inputs. We are able to achieve state-of-the-art results on several summarization and question answering tasks, as well  as outperform the original T5 models on these tasks.

## Summarization Results

LongT5 achieves state-of-the-art performance on several summarization benchmarks that required longer context or multi-document understanding. The table is showing ROUGE-1 scores. LongT5 base models are all reported with 4k input tokens; large and xl models are trained with 16k tokens for arXiv, PubMed, BigPatent, 8k for MultiNews, and 4k for MediaSum and CNN/Daily News.

| Model | arXiv | PubMed | BigPatent | MultiNews | MediaSum | CNN/Daily Mail |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| DANCER PEGASUS | 45.01 | 46.34 | - | - | - | - |
| BigBird-PEGASUS (large) | 46.63 | 46.32 | 60.64 | - | - | - |
| HAT-BART | 46.68 | 48.36 | - | - | - |**44.48** |
| LED (large) | 46.64 | - | - | - | - | - |
| PRIMER | 47.6 | - | - | **49.9** | - | - |
| TG-MultiSum | - | - | - | 47.10 | - | - |
| BART (large) | - | - | - | - | 35.09 | - |
| LongT5 base | 44.87 | 47.77 | 60.95 | 46.01 | 35.09 | 42.15 |
| LongT5 large | 48.28 | 49.98 | 70.38 | 47.18 | 35.53 | 42.49 |
| LongT5 xl | **48.35** | **50.23** | **76.87** | 48.17 | **36.15** | 43.94 |

## QA Results

### Natural Questions

For NQ, we compare T5.1.1 and LongT5 with TGlobal attention. We decided to run T5.1.1 (1) with the default 512 input sequence length and (2) with the largest input sequence length that can fit into device memory, and use those as baselines. Since we are comparing against T5.1.1, for LongT5 experiments we report results at 512 input length for base and large, and the largest input length allowed by each model before running out of memory on the same hardware configuration used in our T5.1.1 experiments. For base and large models, we used 4x8 TPUv3 and no model partitioning; for xl model, we used 8x16 TPUv3 and 8 partitions.

| Model | EM | F1 |
| ---- | ---- | ---- |
| T5.1.1 base-512 | 50.93 | 52.54 |
| T5.1.1 base-6k | 56.73 | 56.73 |
| T5.1.1 large-512 | 57.29 | 60.68 |
| T5.1.1 large-3k | 60.09 | 64.17 |
| T5.1.1 xl-4k | 60.75 | 64.07 |
| LongT5 base-512 | 55.73 | 59.06 |
| LongT5 base-12k | 58.12 | 62.44 |
| LongT5 large-512 | 57.55 | 61.53 |
| LongT5 large-4k | 60.77 | 65.38 |
| LongT5 xl-8k | **62.66** | **66.61** |

Moreover, in our analysis for Input Length vs Speed and Input Length vs Performance sections using NQ, it shows that (1) at shorter sequence length T5.1.1 and LongT5 variants have similar speeds, but as we increase the sequence length, LongT5 becomes significantly faster, (2) T5.1.1 models reach their out-of-memory point much earlier than LongT5 models, and (3) performance increases significantly as input length increases.

### TriviaQA

For TriviaQA, we compare LongT5 with various top approaches on the leader board. All LongT5 models are reported with 16k input tokens.

| Model | EM | F1 |
| ---- | ---- | ---- |
| BigBird-ETC (random attn) | 80.86 | 84.5 |
| Fusion-in-Decoder | 80.09 | 84.35 |
| ReadTwice | 76.86 | 80.85 |
| LongT5 base | 74.67 | 78.9 |
| LongT5 large | 78.38 | 82.45 |
| LongT5 xl | **81.00** | **84.83** |

## Usage

### Data Preprocessing

Most of our tasks are using [Tensorflow Datasets](https://www.tensorflow.org/datasets) which works directly with the [SeqIO](https://github.com/google/seqio) used in the [T5
library](https://github.com/google-research/text-to-text-transfer-transformer). But for Natural Questions and MediaSum we provided our own data preprocessing code. To run the tasks corresponding to these datasets, please specify NQ_DATA_DIR and MEDIASUM_DATA_DIR to the output files produced by the preprocessing code in [tasks.py](longt5/tasks.py).

Example command for running NQ data preprocessing:

```sh
# Data path where the NQ json files are downloaded to.
INPUT_PATH="..."
# Data path where the output files will be generated.
OUTPUT_PATH="..."
LONGT5_DIR="..."  # directory where the LongT5 repo is cloned.

python3 ${LONGT5_DIR}/data/nq_preprocess.py \
  --input_path=${INUT_PATH} \
  --output_path=${OUTPUT_PATH}
```

### Training

The experiments are shown in the [tasks.py](longt5/tasks.py) file. Our architecture, model, and training configuration setups can be found in [Flaxformer](https://github.com/google/flaxformer) github repository.

## Released Model Checkpoints

We have released the following checkpoints for LongT5 pre-trained models:

* **LongT5-Local-Base** (250 million parameters): [gs://t5-data/pretrained_models/t5x/longt5/local_base](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/longt5/local_base/)
* **LongT5-TGlobal-Base** (250 million parameters): [gs://t5-data/pretrained_models/t5x/longt5/tglobal_base](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/longt5/tglobal_base/)
* **LongT5-Local-Large** (780 million parameters): [gs://t5-data/pretrained_models/t5x/longt5/local_large](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/longt5/local_large/)
* **LongT5-TGlobal-Large** (780 million parameters): [gs://t5-data/pretrained_models/t5x/longt5/tglobal_large](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/longt5/tglobal_large/)
* **LongT5-TGlobal-XL** (3 billion parameters): [gs://t5-data/pretrained_models/t5x/longt5/tglobal_xl](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/t5x/longt5/tglobal_xl/)
