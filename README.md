
# Awesome-Align-Modified

This repository contains a modified version of the original Awesome-Align framework. It includes additional fine-tuning on the Chinese-Vietnamese dataset and specific updates for our use case.

---

## Features

- Alignment tasks for multilingual sentence pairs.
- Fine-tuned models for better performance on custom datasets.
- Default pretrained model (`bert-base-multilingual-cased`) for general usage.
- Supports fine-tuning with your own datasets.

---

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/your-repo/awesome-align-modified.git
cd awesome-align-modified
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How to Use

### Running the Default Pretrained Model

To run alignment using the default pretrained model (`bert-base-multilingual-cased`):

```bash
python run_align.py     --data_file=test_data.txt     --output_file=alignment_output_default.txt     --output_word_file=alignment_words_default.txt     --model_name_or_path=bert-base-multilingual-cased     --batch_size=16     --extraction=softmax     --num_workers=0
```

### Running Fine-Tuned Model (Checkpoint-1000)

To run alignment using the fine-tuned model from checkpoint-1000:

```bash
python run_align.py     --data_file=test_data.txt     --output_file=alignment_output_1000.txt     --output_word_file=alignment_words_1000.txt     --model_name_or_path=./output_finetune/checkpoint-1000     --batch_size=16     --extraction=softmax     --num_workers=0
```

### Running Fine-Tuned Model (Checkpoint-2000)

To run alignment using the fine-tuned model from checkpoint-2000:

```bash
python run_align.py     --data_file=test_data.txt     --output_file=alignment_output_2000.txt     --output_word_file=alignment_words_2000.txt     --model_name_or_path=./output_finetune/checkpoint-2000     --batch_size=16     --extraction=softmax     --num_workers=0
```

---

## Fine-Tuning

To fine-tune the model further on your custom dataset, use the following command:

```bash
python run_train.py     --train_data_file=train_data.txt     --output_dir=./output_finetune     --model_name_or_path=./output_finetune/checkpoint-2000     --should_continue     --do_train     --train_tlm     --train_so     --num_train_epochs=3     --per_gpu_train_batch_size=2     --gradient_accumulation_steps=4     --save_steps=1000
```

---

## Dataset Format

The input data for alignment should follow this format:

```plaintext
source sentence ||| target sentence
```

### Example

```plaintext
你好世界 ||| Xin chào thế giới
我们正在学习对齐技术 ||| Chúng tôi đang học kỹ thuật căn chỉnh
```

---

## Outputs

After running alignment, you will get the following outputs:

- `alignment_output_*.txt`: Word index alignments.
- `alignment_words_*.txt`: Word-level alignments.

### Example of Word-Level Alignments:

```plaintext
source_word1target_word1 source_word2target_word2 ...
```

---

## Evaluation

For testing and evaluation, you can compare the results of different models:

- Default pretrained model (`bert-base-multilingual-cased`).
- Fine-tuned checkpoint models (e.g., checkpoint-1000, checkpoint-2000).

---

## Citation

If you use this tool, please cite the original Awesome-Align project:

```bibtex
@inproceedings{dou2021word,
  title={Word Alignment by Fine-tuning Embeddings on Parallel Corpora},
  author={Zi-Yi Dou and Antonios Anastasopoulos and Graham Neubig},
  booktitle={Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
  year={2021}
}
```

For this modified repository, please mention it along with the original citation.

---

## Acknowledgments

This project is built on the Awesome-Align framework. We extend our gratitude to Zi-Yi Dou, Antonios Anastasopoulos, and Graham Neubig for their groundbreaking work. This modified version incorporates fine-tuning and customizations for the Chinese-Vietnamese alignment task.
