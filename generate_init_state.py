import os
import argparse
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
import torch.nn as nn
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--cpu', type=bool, default=False)

    args = parser.parse_args()

    os.environ['RWKV_JIT_ON'] = '1'
    if args.cuda:
        os.environ["RWKV_CUDA_ON"] = '0'
    from rwkv.model import RWKV

    if args.cpu:
        model = RWKV(args.model_path, 'cpu fp32')
    else:
        model = RWKV(args.model_path, 'cuda fp16')
    tokenizer = TRIE_TOKENIZER('rwkv_vocab_v20230424.txt')

    input_ids = tokenizer.encode(args.prompt)

    _, state = model(input_ids, None)

    # real_state = [state[i] for i in range(len(state)) if i % 3 == 1]
    real_state = state

    d = {}
    for i in range(len(real_state)):
        d[f'blocks.{i}.att.state'] = nn.Parameter(real_state[i])

    torch.save(d, args.save_path)
    print('Successfully generated!')
