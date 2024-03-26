"""
References:
https://github.com/BlinkDL/RWKV-LM
"""

import os
import math, warnings
import inspect
import time
from dataclasses import dataclass
import copy

import torch
import torch.nn as nn
from torch.nn import functional as F
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER


class RWKVInfer:
    def __init__(self, model, state=None, cuda=False, strategy='cuda fp16'):
        self.strategy = strategy
        os.environ['RWKV_JIT_ON'] = '1'
        if cuda:
            os.environ["RWKV_CUDA_ON"] = '1'
        else:
            os.environ["RWKV_CUDA_ON"] = '0'

        from rwkv.model import RWKV

        # download models: https://huggingface.co/BlinkDL
        self.model = RWKV(model=model,
                          strategy=strategy)
        self.tokenizer = TRIE_TOKENIZER('rwkv_vocab_v20230424.txt')

        if state:
            self.init_state = self.load_state(state)
            print('State loaded successfully.')
        else:
            self.init_state = None

    @torch.no_grad()
    def load_state(self, path):


        state_weights = torch.load(path)

        fp_16_trained_weight = []
        for w in state_weights:
            fp_16_trained_weight.append(w.to(torch.float16))

        return fp_16_trained_weight

    @staticmethod
    def topp_sampling(logits, top_p=0.5):

        probs = torch.softmax(logits, dim=0)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        cumulative_probs = torch.cumsum(sorted_probs, dim=0)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_keep = sorted_indices[~sorted_indices_to_remove]

        new_probs = torch.zeros_like(probs)
        new_probs[indices_to_keep] = probs[indices_to_keep]

        sampled_token_index = torch.multinomial(new_probs, 1).item()

        return sampled_token_index

    @torch.no_grad()
    def generate(self, prompt, topp=0.3, max_new_tokens=20, stop_token=None, recall=None):

        input_ids = self.tokenizer.encode(prompt)

        state = copy.deepcopy(self.init_state)

        _, state = self.model.forward(input_ids[:-1], state)

        for _ in range(max_new_tokens):
            logits, state = self.model.forward(input_ids[-1:], state)
            new_token = self.topp_sampling(logits, top_p=topp)
            if stop_token:
                # print(new_token)
                if new_token in stop_token:
                    break
            if recall:
                recall(self.tokenizer.decode([new_token]))
            input_ids.append(new_token)
        return input_ids
