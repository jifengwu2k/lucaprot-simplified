import codecs
import os
from typing import List, Optional, Union

import esm
import numpy as np
import torch
from esm import Alphabet, BatchConverter
from esm.model.esm2 import ESM2
from subword_nmt.apply_bpe import BPE
from transformers import BatchEncoding, BertTokenizer

from lucaprot_model import SequenceAndStructureFusionNetwork

SSFN_MAX_SEQ_LENGTH = 2048
ESM_MAX_SEQ_LENGTH = 4096 - 2
ESM_MAX_EMBEDDING_LENGTH = 2048
PAD_TOKEN = 0
PAD_TOKEN_SEGMENT_ID = 0


class LucaProtPredictor:
    def __init__(
        self,
        ssfn_model_dir,
        tokenizer_dir,
        bpe_codes_path,
        device=None,
        esm_model_name="esm2_t36_3B_UR50D",
    ):  # type: (Union[str, os.PathLike], Union[str, os.PathLike], Union[str, os.PathLike], Optional[str], str) -> None
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.ssfn_model_dir = os.fspath(ssfn_model_dir)
        self.tokenizer_dir = os.fspath(tokenizer_dir)
        self.bpe_codes_path = os.fspath(bpe_codes_path)
        self.esm_model_name = esm_model_name

        self.ssfn_model = SequenceAndStructureFusionNetwork.from_pretrained(
            pretrained_model_name_or_path=self.ssfn_model_dir
        )
        self.ssfn_model.to(self.device)
        self.ssfn_model.eval()

        self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(
            self.esm_model_name
        )  # type: ESM2, Alphabet
        self.esm_model.to(self.device)
        self.esm_model.eval()

        self.bpe_encoder = BPE(
            codecs.open(self.bpe_codes_path), merges=-1, separator=""
        )
        self.ssfn_tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.tokenizer_dir,
            do_lower_case=False,
        )
        self.esm_converter = BatchConverter(
            self.alphabet, ESM_MAX_SEQ_LENGTH
        )  # type: BatchConverter

    def _prepare_sequence_inputs(self, protein_seq):  # type: (str) -> tuple
        seq_to_list = self.bpe_encoder.process_line(protein_seq).split()  # type: List[str]
        truncated_seq_to_list = seq_to_list[: SSFN_MAX_SEQ_LENGTH - 2]
        seq = " ".join(truncated_seq_to_list)

        inputs = self.ssfn_tokenizer.encode_plus(  # type: BatchEncoding
            seq,
            None,
            add_special_tokens=True,
            max_length=SSFN_MAX_SEQ_LENGTH,
            truncation=True,
        )

        input_ids = inputs["input_ids"]  # type: List[int]
        token_type_ids = inputs["token_type_ids"]  # type: List[int]
        attention_mask = [1] * len(input_ids)

        padding_length = SSFN_MAX_SEQ_LENGTH - len(input_ids)
        input_ids = input_ids + ([PAD_TOKEN] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([PAD_TOKEN_SEGMENT_ID] * padding_length)

        return input_ids, attention_mask, token_type_ids

    def _prepare_embedding_inputs(self, protein_seq, prot_id):  # type: (str, str) -> tuple
        protein_seq_for_esm = protein_seq[:ESM_MAX_SEQ_LENGTH]
        _, raw_seqs, tokens = self.esm_converter([(prot_id, protein_seq_for_esm)])

        with torch.no_grad():
            tokens = tokens.to(device=self.device, non_blocking=True)
            out = self.esm_model(
                tokens, repr_layers=[self.esm_model.num_layers], return_contacts=False
            )
            truncate_len = min(ESM_MAX_SEQ_LENGTH, len(raw_seqs[0]))
            embedding_info = (
                out["representations"][self.esm_model.num_layers]
                .to(device="cpu")[0, 1 : truncate_len + 1]
                .clone()
                .numpy()
            )

        embedding_length = embedding_info.shape[0]
        embedding_attention_mask = [1] * embedding_length

        if embedding_length > ESM_MAX_EMBEDDING_LENGTH:
            embedding_info = embedding_info[:ESM_MAX_EMBEDDING_LENGTH, :]
            embedding_attention_mask = [1] * ESM_MAX_EMBEDDING_LENGTH
        else:
            embedding_padding_length = ESM_MAX_EMBEDDING_LENGTH - embedding_length
            embedding_attention_mask = (
                embedding_attention_mask + [0] * embedding_padding_length
            )
            embedding_info = np.pad(
                embedding_info,
                [(0, embedding_padding_length), (0, 0)],
                mode="constant",
                constant_values=PAD_TOKEN,
            )

        return embedding_info, embedding_attention_mask

    def predict(self, protein_seq, prot_id="protein_1"):  # type: (str, str) -> float
        input_ids, attention_mask, token_type_ids = self._prepare_sequence_inputs(
            protein_seq
        )
        embedding_info, embedding_attention_mask = self._prepare_embedding_inputs(
            protein_seq, prot_id
        )

        with torch.no_grad():
            _, output = self.ssfn_model(
                input_ids=torch.tensor([input_ids], dtype=torch.long).to(self.device),
                attention_mask=torch.tensor([attention_mask], dtype=torch.long).to(
                    self.device
                ),
                token_type_ids=torch.tensor([token_type_ids], dtype=torch.long).to(
                    self.device
                ),
                embedding_info=torch.tensor(
                    np.array([embedding_info], dtype=np.float32), dtype=torch.float32
                ).to(self.device),
                embedding_attention_mask=torch.tensor(
                    [embedding_attention_mask], dtype=torch.long
                ).to(self.device),
            )

        return float(output.detach().cpu().item())
