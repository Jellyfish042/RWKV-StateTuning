from dataclasses import dataclass
from typing import Union


@dataclass
class TrainConfig:
    # model_path: str = 'RWKV-x060-World-1B6-v2-20240208-ctx4096.pth'
    model_path: str = 'RWKV-x060-World-3B-v2-20240228-ctx4096.pth'
    # model_size: str = 'v6_1b5'
    model_size: str = 'v6_3b'
    init_state: Union[str, None] = None
    train_data: str = 'data/e2e_train_rwkvtok.jsonl'
    # train_data: str = 'data/wmt18_zh-en_rwkvtok_train_v3.jsonl'
    # val_data: Union[str, None] = None
    val_data: Union[str, None] = 'data/e2e_eval_rwkvtok.jsonl'
    # val_data: Union[str, None] = 'data/wmt18_zh-en_rwkvtok_val_v3.jsonl'
    log_file: str = '/root/tf-logs/train_state_test2'
    lr: int = 1e-3
    train_batch_size: int = 4
    eval_batch_size: int = 8
    accumulation_step: int = 1
    epoch: int = 1
    ctx: int = 256
    save_dir: str = 'weights'
    state_ckpt_prefix: str = 'v6_3b_en2zh'
    # train_ckpt_name_prefix: str = 'v6_encoder'
    save_steps: int = 1000
    print_log_steps: int = 50
    val_steps: int = 1000
    tensor_board: bool = True
    clip_grad_norm: float = 1.0
    random_state: int = 42


@dataclass
class RWKV6Config1B5:
    vocab_size: int = 65536
    n_layer: int = 24
    n_head: int = 32
    n_embd: int = 2048
    dropout: float = 0.0
    ctx_len: int = 4096
    head_size: int = 64
    head_size_a: int = 64
    dim_att: int = 2048
    dim_ffn: int = int(3.5 * 2048)
    head_size_divisor: int = 8


@dataclass
class RWKV6Config3B:
    vocab_size: int = 65536
    n_layer: int = 32
    n_head: int = 40
    n_embd: int = 2560
    dropout: float = 0.0
    ctx_len: int = 4096
    head_size: int = 64
    head_size_a: int = 64
    dim_att: int = 2560
    dim_ffn: int = int(3.5 * 2560)
    head_size_divisor: int = 8


SIZE_MAP = {
    'v6_1b5': RWKV6Config1B5(),
    'v6_3b': RWKV6Config3B(),
}
