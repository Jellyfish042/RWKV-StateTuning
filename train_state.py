from config import *
from rwkv_6_model import RWKV6_0
import argparse
from config import TrainConfig
from dataset import StateTuningDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
import os
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import random
import numpy as np
import re
import pdb


def set_random_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def extract_integer(s):
    matches = re.findall(r'\d+', s)
    if matches:
        return int(matches[0])
    else:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Train Configuration")
    parser.add_argument("--model_path", type=str, default=TrainConfig.model_path, help="Model file path")
    parser.add_argument("--model_size", type=str, default=TrainConfig.model_size, help="Size of the model")
    parser.add_argument("--init_state", type=str, default=TrainConfig.init_state, help="Initial state")
    parser.add_argument("--train_data", type=str, default=TrainConfig.train_data, help="Training data file path")
    parser.add_argument("--val_data", type=str, default=TrainConfig.val_data, help="Validation data file path")
    parser.add_argument("--log_file", type=str, default=TrainConfig.log_file, help="Log file path")
    parser.add_argument("--lr", type=float, default=TrainConfig.lr, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=TrainConfig.train_batch_size,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=TrainConfig.eval_batch_size,
                        help="Evaluation batch size")
    parser.add_argument("--accumulation_step", type=int, default=TrainConfig.accumulation_step,
                        help="Gradient accumulation steps")
    parser.add_argument("--epoch", type=int, default=TrainConfig.epoch, help="Number of epochs")
    parser.add_argument("--ctx", type=int, default=TrainConfig.ctx, help="Context length")
    parser.add_argument("--save_dir", type=str, default=TrainConfig.save_dir, help="Directory to save model weights")
    parser.add_argument("--state_ckpt_prefix", type=str, default=TrainConfig.state_ckpt_prefix,
                        help="State checkpoint prefix")
    parser.add_argument("--save_steps", type=int, default=TrainConfig.save_steps, help="Model save interval in steps")
    parser.add_argument("--print_log_steps", type=int, default=TrainConfig.print_log_steps,
                        help="Log printing interval in steps")
    parser.add_argument("--val_steps", type=int, default=TrainConfig.val_steps, help="Validation interval in steps")
    parser.add_argument("--tensor_board", type=bool, default=TrainConfig.tensor_board,
                        help="Enable TensorBoard logging")
    parser.add_argument("--clip_grad_norm", type=float, default=TrainConfig.clip_grad_norm,
                        help="Gradient clipping norm")
    parser.add_argument("--random_state", type=int, default=TrainConfig.random_state,
                        help="Random state for reproducibility")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    train_config = TrainConfig(**vars(args))
    print(train_config)

    if train_config.init_state is not None:
        raise NotImplementedError

    set_random_seeds(train_config.random_state)

    device = torch.device('cuda')

    model_config = SIZE_MAP[train_config.model_size]
    print(model_config)

    model = RWKV6_0(model_config,
                    train_config.model_path,
                    train_config.init_state
                    )
    model = model.to(device)

    model.freeze_model_except("state")

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    tokenizer = TRIE_TOKENIZER('rwkv_vocab_v20230424.txt')

    train_dataset = StateTuningDataset(train_config.train_data, train_config.ctx)
    if train_config.val_data:
        val_dataset = StateTuningDataset(train_config.val_data, train_config.ctx)
        val_loader = DataLoader(val_dataset, batch_size=train_config.eval_batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=train_config.train_batch_size, shuffle=True, drop_last=True)

    optimizer_params = []
    for name, param in model.named_parameters():
        print(name)
        if 'state' in name:
            optimizer_params.append(param)

    # only for test - group lr
    # factor = 1e-3
    # optimizer_params = []
    # for name, param in model.named_parameters():
    #     if 'state' in name:
    #         optimizer_params.append({'params': param, 'lr': train_config.lr + extract_integer(name) * factor})
    #     else:
    #         param.requires_grad = False

    optim = torch.optim.Adam(optimizer_params, lr=train_config.lr)
    # scheduler = lr_scheduler.StepLR(optim, step_size=1000, gamma=0.1)
    # scheduler = CosineAnnealingLR(optim, T_max=10000, eta_min=1e-5)

    if train_config.tensor_board:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(train_config.log_file)

    total_loss = 0
    global_step = 0

    for epoch in range(train_config.epoch):
        for ids, mask in train_loader:

            if global_step % train_config.val_steps == 0 and train_config.val_data:

                print("Evaluating...")

                val_loss = 0.0
                val_steps = 0
                model.eval()
                with torch.no_grad():
                    for ids, mask in tqdm(val_loader):
                        ids, mask = ids.to(device), mask.to(device)
                        input_ids, target_ids = ids[:, :-1].contiguous(), ids[:, 1:].contiguous()
                        mask_ids = mask[:, 1:].contiguous()

                        _, loss = model.forward(input_ids, targets=target_ids, mask=mask_ids)

                        val_loss += loss.item()
                        val_steps += 1

                    avg_val_loss = val_loss / val_steps
                    print(f"Validation Loss after {global_step} steps: {avg_val_loss:.4f}")

                if train_config.tensor_board:
                    writer.add_scalar('val loss', avg_val_loss, global_step=global_step)

            if global_step % train_config.save_steps == 0:

                weights_dir = train_config.save_dir
                if not os.path.exists(weights_dir):
                    os.makedirs(weights_dir)

                state_save_path = os.path.join(weights_dir, f'{train_config.state_ckpt_prefix}_{str(global_step)}.pth')
                model.save_state(state_save_path)

            ids, mask = ids.to(device), mask.to(device)
            input_ids, target_ids = ids[:, :-1].contiguous(), ids[:, 1:].contiguous()
            mask_ids = mask[:, 1:].contiguous()
            logits, loss = model.forward(input_ids, targets=target_ids, mask=mask_ids)
            total_loss += loss.item()
            loss = loss / train_config.accumulation_step
            loss.backward()
            if (global_step + 1) % train_config.accumulation_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.clip_grad_norm)
                optim.step()
                optim.zero_grad()

            if global_step % train_config.print_log_steps == 0 and global_step != 0:
                if train_config.tensor_board:
                    writer.add_scalar('training loss', total_loss / train_config.print_log_steps,
                                      global_step=global_step)
                print(
                    f"Epoch: {epoch + 1}, Global Step: {global_step}, Avg Loss: {total_loss / train_config.print_log_steps:.4f}")
                total_loss = 0

            global_step += 1

    if train_config.tensor_board:
        writer.close()
