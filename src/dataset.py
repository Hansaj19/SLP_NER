"""
PyTorch Dataset and Collate Function for Sequence Tagging.
"""
from typing import List, Dict, Any, Tuple
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.data_utils import PAD_IDX, UNK_IDX, TAG2IDX

class NERDataset(Dataset):
    """
    PyTorch Dataset for Named Entity Recognition.
    Converts list of token strings and tag strings to PyTorch LongTensors.
    """
    def __init__(self, data: List[Dict[str, Any]], word2idx: Dict[str, int], tag2idx: Dict[str, int] = TAG2IDX):
        self.data = data
        self.word2idx = word2idx
        self.tag2idx = tag2idx

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
        item = self.data[idx]
        tokens = item["tokens"]
        tags = item["tags"]

        token_ids = [self.word2idx.get(t, UNK_IDX) for t in tokens]
        tag_ids = [self.tag2idx.get(tg, self.tag2idx["O"]) for tg in tags]

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(tag_ids, dtype=torch.long),
            tokens,
            tags
        )

def pad_collate_fn(batch):
    """
    Pads variable-length sequences within a mini-batch to the max sequence length.
    Returns:
      padded_tokens: (batch_size, max_seq_len)
      padded_tags: (batch_size, max_seq_len)
      lengths: (batch_size)
      raw_tokens: List of original token lists
      raw_tags: List of original tag lists
    """
    tokens_list, tags_list, raw_tokens, raw_tags = zip(*batch)
    lengths = torch.tensor([len(t) for t in tokens_list], dtype=torch.long)

    # Pad with PAD_IDX (0)
    padded_tokens = pad_sequence(tokens_list, batch_first=True, padding_value=PAD_IDX)
    padded_tags = pad_sequence(tags_list, batch_first=True, padding_value=PAD_IDX)

    return padded_tokens, padded_tags, lengths, list(raw_tokens), list(raw_tags)
