# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and 
# The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.
# All rights reserved.
# Modifications copyright (C) 2020 Zi-Yi Dou
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

import argparse
import itertools
import os
import random
import shutil
import tempfile

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset
from tqdm import trange

# Vẫn giữ các import gốc từ awesome_align để không phá cấu trúc
from awesome_align import modeling
from awesome_align.configuration_bert import BertConfig
from awesome_align.modeling import BertForMaskedLM
from awesome_align.modeling_utils import PreTrainedModel
from awesome_align.tokenization_bert import BertTokenizer
from awesome_align.tokenization_utils import PreTrainedTokenizer


def seed_all(config):
    """
    Ấn định giá trị seed cho random, numpy, torch (cả CUDA) nếu config.seed >= 0.
    """
    if config.seed >= 0:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)


class ParallelTextDataset(IterableDataset):
    """
    Dataset này đọc từng dòng trong file: mỗi dòng chứa cặp câu src và tgt cách nhau bởi ' ||| '.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path, offsets=None):
        assert os.path.isfile(file_path), f"File {file_path} không tồn tại."
        print('Đang tải dữ liệu cho ParallelTextDataset...')
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.offsets = offsets

    def handle_line(self, worker_idx, line):
        """
        Xử lý một dòng, tách src/tgt, tokenize, chuyển sang ID và 
        trả về tuple (worker_idx, ids_src, ids_tgt, bpe_map_src, bpe_map_tgt, list_src, list_tgt).
        """
        line = line.strip()
        if not line or ' ||| ' not in line:
            return None

        src_text, tgt_text = line.split(' ||| ')
        if not src_text.strip() or not tgt_text.strip():
            return None

        # Tách từng từ
        src_tokens = src_text.strip().split()
        tgt_tokens = tgt_text.strip().split()

        # Tokenize
        src_subwords = [self.tokenizer.tokenize(w) for w in src_tokens]
        tgt_subwords = [self.tokenizer.tokenize(w) for w in tgt_tokens]

        # Chuyển subword -> ID
        src_ids_per_word = [self.tokenizer.convert_tokens_to_ids(x) for x in src_subwords]
        tgt_ids_per_word = [self.tokenizer.convert_tokens_to_ids(x) for x in tgt_subwords]

        # Tạo input_ids có [CLS], [SEP]
        ids_src = self.tokenizer.prepare_for_model(
            list(itertools.chain(*src_ids_per_word)),
            return_tensors='pt',
            max_length=self.tokenizer.max_len
        )['input_ids']

        ids_tgt = self.tokenizer.prepare_for_model(
            list(itertools.chain(*tgt_ids_per_word)),
            return_tensors='pt',
            max_length=self.tokenizer.max_len
        )['input_ids']

        # Bỏ qua trường hợp câu quá ngắn (chỉ [CLS], [SEP])
        if len(ids_src[0]) <= 2 or len(ids_tgt[0]) <= 2:
            return None

        # Tạo mapping bpe -> word
        bpe_map_src = []
        for idx, sbw_list in enumerate(src_subwords):
            bpe_map_src.extend([idx] * len(sbw_list))

        bpe_map_tgt = []
        for idx, sbw_list in enumerate(tgt_subwords):
            bpe_map_tgt.extend([idx] * len(sbw_list))

        return (
            worker_idx,
            ids_src[0],
            ids_tgt[0],
            bpe_map_src,
            bpe_map_tgt,
            src_tokens,
            tgt_tokens
        )

    def __iter__(self):
        """
        Nếu có offsets, mỗi worker sẽ đọc một đoạn riêng của file.
        """
        if self.offsets is not None:
            worker_info = torch.utils.data.get_worker_info()
            w_id = worker_info.id
            start_pos = self.offsets[w_id]
            end_pos = self.offsets[w_id+1] if (w_id+1) < len(self.offsets) else None
        else:
            w_id = 0
            start_pos, end_pos = 0, None

        with open(self.file_path, mode='r', encoding='utf-8') as f:
            f.seek(start_pos)
            line = f.readline()
            while line:
                processed = self.handle_line(w_id, line)
                if processed is None:
                    # Trường hợp dòng không phù hợp format
                    print(f"Dòng '{line.strip()}' (offset={f.tell()}) không đúng định dạng. Bỏ qua.")
                    dummy_ids = torch.tensor([self.tokenizer.cls_token_id, 999, self.tokenizer.sep_token_id])
                    yield (w_id, dummy_ids, dummy_ids, [-1], [-1], '', '')
                else:
                    yield processed

                if end_pos is not None and f.tell() >= end_pos:
                    break
                line = f.readline()


def split_offsets(filename, num_workers):
    """
    Tính toán offset cho từng worker để đọc file. 
    Mỗi worker xử lý một phần file để tăng tốc.
    """
    if num_workers <= 1:
        return None
    with open(filename, "r", encoding="utf-8") as f:
        file_size = os.fstat(f.fileno()).st_size
        chunk_size = file_size // num_workers
        offsets = [0]
        for i in range(1, num_workers):
            f.seek(chunk_size * i)
            while True:
                pos = f.tell()
                try:
                    f.readline()  # đọc để tìm vị trí cắt an toàn
                    break
                except UnicodeDecodeError:
                    pos -= 1
                    f.seek(pos)
            offsets.append(f.tell())
    return offsets


def init_output_writers(file_path, num_workers):
    """
    Khởi tạo writer (một file chính và một số file tạm).
    """
    main_writer = open(file_path, 'w+', encoding='utf-8')
    writer_list = [main_writer]
    if num_workers > 1:
        for _ in range(num_workers - 1):
            writer_list.append(tempfile.TemporaryFile(mode='w+', encoding='utf-8'))
    return writer_list


def unify_temp_files(writers):
    """
    Nối các file tạm (của worker>0) vào file chính (worker=0).
    """
    if len(writers) == 1:
        writers[0].close()
        return

    base_writer = writers[0]
    for w in writers[1:]:
        w.seek(0)
        shutil.copyfileobj(w, base_writer)
        w.close()
    base_writer.close()


def align_words(config, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    """
    Hàm chính thực hiện lấy alignments từ mô hình.
    """

    def collate_fn(samples):
        """
        Gom các sample thành batch, thực hiện padding, tách thành các list tương ứng.
        """
        (w_ids, src_ids, tgt_ids, bpe_s, bpe_t, raw_s, raw_t) = zip(*samples)
        padded_src = pad_sequence(src_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        padded_tgt = pad_sequence(tgt_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        return w_ids, padded_src, padded_tgt, bpe_s, bpe_t, raw_s, raw_t

    # Nếu nhiều worker, tính offset
    offsets = split_offsets(config.data_file, config.num_workers)

    # Tạo dataset và dataloader
    dataset = ParallelTextDataset(tokenizer, file_path=config.data_file, offsets=offsets)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers
    )

    model.to(config.device)
    model.eval()

    pbar = trange(0, desc="Extracting alignments", leave=True)

    # Mở file ghi alignments chính
    align_writers = init_output_writers(config.output_file, config.num_workers)

    prob_writers = None
    if config.output_prob_file is not None:
        prob_writers = init_output_writers(config.output_prob_file, config.num_workers)

    word_writers = None
    if config.output_word_file is not None:
        word_writers = init_output_writers(config.output_word_file, config.num_workers)

    # Lặp qua các batch
    for batch in loader:
        with torch.no_grad():
            (wk_ids, ids_src, ids_tgt, map_src, map_tgt, sents_src, sents_tgt) = batch
            # Lấy word alignment từ mô hình
            alignment_results = model.get_aligned_word(
                ids_src,
                ids_tgt,
                map_src,
                map_tgt,
                config.device,
                0,
                0,
                align_layer=config.align_layer,
                extraction=config.extraction,
                softmax_threshold=config.softmax_threshold,
                test=True,
                output_prob=(config.output_prob_file is not None)
            )

            # Ghi kết quả
            for w_id, aligns, src_words, tgt_words in zip(wk_ids, alignment_results, sents_src, sents_tgt):
                align_pairs = []
                prob_pairs = []
                word_pairs = []

                for align_item in aligns:
                    if align_item[0] != -1:
                        # Thêm "chỉ số - chỉ số"
                        align_pairs.append(f"{align_item[0]}-{align_item[1]}")

                        # Nếu có yêu cầu ghi xác suất
                        if config.output_prob_file:
                            prob_pairs.append(f"{aligns[align_item]}")

                        # Nếu có yêu cầu ghi cặp từ
                        if config.output_word_file:
                            word_pairs.append(f"{src_words[align_item[0]]}<sep>{tgt_words[align_item[1]]}")

                align_writers[w_id].write(' '.join(align_pairs) + '\n')

                if prob_writers:
                    prob_writers[w_id].write(' '.join(prob_pairs) + '\n')

                if word_writers:
                    word_writers[w_id].write(' '.join(word_pairs) + '\n')

            pbar.update(len(ids_src))

    # Nối file tạm -> file chính
    unify_temp_files(align_writers)
    if prob_writers:
        unify_temp_files(prob_writers)
    if word_writers:
        unify_temp_files(word_writers)


def run():
    parser = argparse.ArgumentParser()

    # Tham số bắt buộc
    parser.add_argument(
        "--data_file",
        default=None,
        type=str,
        required=True,
        help="File chứa dữ liệu đầu vào dạng text: 'src ||| tgt'."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="File đầu ra để ghi các cặp alignment (chỉ số)."
    )

    # Tham số tuỳ chọn
    parser.add_argument("--align_layer", type=int, default=8, help="Layer dùng để lấy alignment.")
    parser.add_argument("--extraction", default='softmax', type=str, help="Phương pháp trích xuất: softmax/entmax15.")
    parser.add_argument("--softmax_threshold", type=float, default=0.001, help="Ngưỡng xác suất cho softmax.")
    parser.add_argument("--output_prob_file", default=None, type=str, help="Nếu muốn ghi xác suất alignment, chỉ định file.")
    parser.add_argument("--output_word_file", default=None, type=str, help="Nếu muốn ghi cặp từ, chỉ định file.")
    parser.add_argument("--model_name_or_path", default=None, type=str, help="Checkpoint (hoặc tên) mô hình pretrained.")
    parser.add_argument("--config_name", default=None, type=str, help="Tên hoặc đường dẫn config (nếu khác model).")
    parser.add_argument("--tokenizer_name", default=None, type=str, help="Tên hoặc đường dẫn tokenizer (nếu khác model).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--batch_size", default=32, type=int, help="Kích thước batch.")
    parser.add_argument("--cache_dir", default=None, type=str, help="Cache folder cho các mô hình tải về.")
    parser.add_argument("--no_cuda", action="store_true", help="Không dùng GPU kể cả khi có CUDA.")
    parser.add_argument("--num_workers", type=int, default=4, help="Số worker dùng cho DataLoader.")

    args = parser.parse_args()
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    seed_all(args)

    # Chuẩn bị config, model, tokenizer
    config_class, model_class, tokenizer_class = BertConfig, BertForMaskedLM, BertTokenizer

    if args.config_name:
        config_obj = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config_obj = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config_obj = config_class()

    if args.tokenizer_name:
        tokenizer_obj = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer_obj = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "Không thể khởi tạo tokenizer mới từ đầu. Hãy chỉ định --tokenizer_name hoặc --model_name_or_path."
        )

    # Cập nhật ID cho các token đặc biệt
    modeling.PAD_ID = tokenizer_obj.pad_token_id
    modeling.CLS_ID = tokenizer_obj.cls_token_id
    modeling.SEP_ID = tokenizer_obj.sep_token_id

    if args.model_name_or_path:
        model_obj = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config_obj,
            cache_dir=args.cache_dir,
        )
    else:
        model_obj = model_class(config=config_obj)

    align_words(args, model_obj, tokenizer_obj)


if __name__ == "__main__":
    run()
