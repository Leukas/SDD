import os
import re
import warnings
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import math

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

class DSTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        train_files,
        ds_factor=4,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        additional_special_tokens=None,
        **kwargs
    ) -> None:
        # Add extra_ids to the special token list
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None:
            # Check that we have the right number of extra_id special tokens
            extra_tokens = len(set(filter(lambda x: bool("extra_id" in str(x)), additional_special_tokens)))
            if extra_tokens != extra_ids:
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are provided to T5Tokenizer. "
                    "In this case the additional_special_tokens must include the extra_ids tokens"
                )

        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        self._extra_ids = extra_ids
        self.ds_factor = ds_factor
        self.vocab, self.idx2word = self.read_files(train_files)
        print(len(self.vocab))

    def read_files(self, files):
        vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2}
        idx2word = ["<pad>", "</s>", "<unk>"]
        for filepath in files:
            with open(filepath, 'r', encoding='utf8') as file:
                for line in file:
                    strline = line.strip()
                    ds_len = math.ceil(len(strline)/self.ds_factor)
                    words = [strline[i*self.ds_factor:(i+1)*self.ds_factor] for i in range(ds_len)]
                    for word in words:
                        if word not in vocab:
                            vocab[word] = len(vocab)
                            idx2word.append(word)
        return vocab, idx2word

    @property
    def vocab_size(self):
        return len(self.vocab) + self._extra_ids

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # normal case: some special tokens
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        """Do not add eos again if user already added it."""
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated eos tokens being added."
            )
            return token_ids
        else:
            return token_ids + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make
        use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:

        - single sequence: `X </s>`
        - pair of sequences: `A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, d):
        self.__dict__ = d

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        strtext = text.strip()
        ds_len = math.ceil(len(strtext)/self.ds_factor)
        words = [strtext[i*self.ds_factor:(i+1)*self.ds_factor] for i in range(ds_len)]
        return words

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1
        return self.vocab[token] if token in self.vocab else self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index < len(self.vocab):
            token = self.idx2word[index]
        else:
            token = f"<extra_id_{self.vocab_size - 1 - index}>"
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = ""
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                out_string += token + " "
            else:
                out_string += token
        return out_string.strip()

    # def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
    #     if not os.path.isdir(save_directory):
    #         logger.error(f"Vocabulary path ({save_directory}) should be a directory")
    #         return
    #     out_vocab_file = os.path.join(
    #         save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
    #     )

    #     if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
    #         copyfile(self.vocab_file, out_vocab_file)
    #         logger.info(f"Copy vocab file to {out_vocab_file}")

    #     return (out_vocab_file,)
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return ()



if __name__ == "__main__":
    folder = "data/iwslt2017/"
    files = [folder + "train.en", folder + "train.de"]
    tok = DSTokenizer(files)
    x = tok.encode("Thank you so much, Chris.")
    x = tok.encode("And it's truly a great honor to have the opportunity to come to this stage twice; I'm extremely grateful.")
    print(x, tok.convert_ids_to_tokens(x))
    print(tok.vocab_size)