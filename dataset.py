import torch
from torch.utils.data import Dataset
import json


def pad_and_truncate_sequences(sequences, max_len, padding_value=0):
    processed_sequences = []

    for seq in sequences:
        if len(seq) > max_len:
            processed_seq = seq[:max_len]
        else:
            num_padding = max_len - len(seq)
            processed_seq = seq + [padding_value] * num_padding

        processed_sequences.append(processed_seq)

    return processed_sequences


class StateTuningDataset(Dataset):
    def __init__(self, filename, max_len):
        self.texts = []
        self.masks = []

        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.texts.append(item['text'])
                self.masks.append(item['mask'])

        self.texts = pad_and_truncate_sequences(self.texts, max_len, padding_value=0)
        self.masks = pad_and_truncate_sequences(self.masks, max_len, padding_value=0)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text_tensor = torch.tensor(self.texts[index], dtype=torch.long)
        mask_tensor = torch.tensor(self.masks[index], dtype=torch.long)
        return text_tensor, mask_tensor
