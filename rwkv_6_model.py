import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pdb

from rwkv_6_0_kernel import RUN_CUDA_RWKV6_S


class StateEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        encoded_dim = 64
        dim_factor = 8

        self.n_head = args.dim_att // args.head_size_a

        self.encoded_state = nn.Parameter(torch.zeros(self.n_head, encoded_dim, dtype=torch.bfloat16))
        nn.init.normal_(self.encoded_state, mean=0.0, std=0.02)

        self.state_proj_1 = nn.Linear(encoded_dim, encoded_dim * dim_factor, dtype=torch.bfloat16)
        self.state_proj_2 = nn.Linear(encoded_dim * dim_factor, args.head_size_a * args.head_size_a, dtype=torch.bfloat16)

    # def forward(self):
    #     out = torch.tanh(self.state_proj_1(self.encoded_state))
    #     out = self.state_proj_2(out)
    #     out = out.reshape(self.n_head, self.args.head_size_a, self.args.head_size_a)
    #     return out

        self.state_ln1 = nn.LayerNorm(encoded_dim * dim_factor, dtype=torch.bfloat16)
        self.state_ln2 = nn.LayerNorm(args.head_size_a * args.head_size_a, dtype=torch.bfloat16)

    def forward(self):
        out = self.state_proj_1(self.encoded_state)
        out = torch.tanh(self.state_ln1(out))
        out = self.state_proj_2(out)
        out = self.state_ln2(out)
        out = out.reshape(self.n_head, self.args.head_size_a, self.args.head_size_a)
        return out

# class StateEncoder(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#
#         self.args = args
#         encoded_dim = 64
#         self.n_head = args.dim_att // args.head_size_a
#
#         self.encoded_state = nn.Parameter(torch.zeros(self.n_head, encoded_dim, encoded_dim, dtype=torch.bfloat16))
#
#
#     def forward(self):
#         return self.encoded_state


class RWKV_Tmix_x060(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            TIME_MIX_EXTRA_DIM = 32  # generate TIME_MIX for w,k,v,r,g
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, TIME_MIX_EXTRA_DIM * 5).uniform_(-1e-4, 1e-4))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, TIME_MIX_EXTRA_DIM, args.n_embd).uniform_(-1e-4, 1e-4))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, args.dim_att))

            TIME_DECAY_EXTRA_DIM = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, TIME_DECAY_EXTRA_DIM).uniform_(-1e-4, 1e-4))
            self.time_decay_w2 = nn.Parameter(torch.zeros(TIME_DECAY_EXTRA_DIM, args.dim_att).uniform_(-1e-4, 1e-4))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        # self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        # test
        self.time_shift = nn.ZeroPad2d((0, 0, 0, -1))
        self.token_shift_state = nn.Parameter(torch.zeros((1, args.n_embd),
                                                          requires_grad=True,
                                                          dtype=torch.bfloat16))

        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5) * (args.head_size_divisor ** 2))

        # self.state = nn.Parameter(torch.zeros(self.n_head, self.head_size, self.head_size, dtype=torch.bfloat16))

        self.state_encoder = StateEncoder(args)

        # self.state = nn.Parameter(torch.randn(self.n_head, self.head_size, self.head_size, dtype=torch.bfloat16) * 0.02)
        # self.state = nn.Parameter(
        #     torch.nn.init.uniform_(torch.empty(self.n_head, self.head_size, self.head_size, dtype=torch.bfloat16),
        #                            a=-0.1, b=0.1))
        # torch.nn.init.xavier_uniform_(self.state, gain=1.0)  # xavier_uniform_
        # torch.nn.init.kaiming_uniform_(self.state, mode='fan_in', nonlinearity='relu')
        # torch.nn.init.orthogonal_(self.state)

    def jit_func(self, x):
        B, T, C = x.size()

        token_shift_x = self.token_shift_state.repeat(B, 1, 1)

        # xx = self.time_shift(x) - x
        xx = torch.cat([token_shift_x, self.time_shift(x)], dim=1) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        return r, k, v, g, w

    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)

        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        state = self.state_encoder.forward()

        r, k, v, g, w = self.jit_func(x)
        x = RUN_CUDA_RWKV6_S(B, T, C, H, r, k, v, w, u=self.time_faaaa, s=state)

        return self.jit_func_2(x, g)


class RWKV_CMix_x060(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        # self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        # test
        self.time_shift = nn.ZeroPad2d((0, 0, 0, -1))
        self.token_shift_state = nn.Parameter(torch.zeros((1, args.n_embd),
                                                          requires_grad=True,
                                                          dtype=torch.bfloat16))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()

        token_shift_x = self.token_shift_state.repeat(B, 1, 1)

        xx = torch.cat([token_shift_x, self.time_shift(x)], dim=1) - x

        # xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x060(args, layer_id)
        self.ffn = RWKV_CMix_x060(args, layer_id)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p=args.dropout)
            self.drop1 = nn.Dropout(p=args.dropout)

    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)

        if self.args.dropout == 0:
            x = x + self.att(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            x = self.drop0(x + self.att(self.ln1(x)))
            x = self.drop1(x + self.ffn(self.ln2(x)))

        return x


class RWKV6_0(nn.Module):
    def __init__(self, args, model_weight=None, state_weight=None):
        super().__init__()
        self.args = args
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p=args.dropout)

        print(f"number of parameters in model: {self.get_num_params()}")

        if model_weight is None:
            print("No model path provided.")
        else:
            self.from_pretrained(model_weight)

        if state_weight is None:
            print("No state path provided.")
        else:
            self.load_state(state_weight)

    def forward(self, idx, targets=None, mask=None):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)

        if args.dropout > 0:
            x = self.drop0(x)
        for block in self.blocks:
            x = block(x)

        x = self.ln_out(x)

        logits = self.head(x)

        if targets is not None:
            if mask is not None:
                mask = mask.view(-1) == 1
                active_loss = mask.view(-1) == 1
                active_logits = logits.view(-1, logits.size(-1))[active_loss]
                active_labels = targets.view(-1)[active_loss]
                loss = F.cross_entropy(active_logits, active_labels, ignore_index=-1)
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            loss = None

        return logits, loss

    def freeze_model_except(self, keyword):
        for name, param in self.named_parameters():
            if keyword not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    @staticmethod
    def load_parameters(w, name, target_module):
        print(f'loading: {name}')
        if name.endswith('weight'):
            assert target_module.weight.data.shape == w[
                name].shape, f'{target_module.weight.data.shape} != {w[name].shape}'
            target_module.weight.data = w[name]
        elif name.endswith('bias'):
            assert target_module.bias.data.shape == w[name].shape, f'{target_module.bias.data.shape} != {w[name].shape}'
            target_module.bias.data = w[name]
        else:
            assert target_module.data.shape == w[
                name].shape, f'target: {target_module.data.shape} != weight: {w[name].shape}'
            target_module.data = w[name]

    def from_pretrained(self, path):

        w = torch.load(path, map_location='cpu')

        total_params = sum(p.numel() for p in w.values())
        print(f"Total number of parameters in state_dict: {total_params}")

        for p_name in w.keys():
            # print(p_name)
            if 'blocks' in p_name:
                block_idx = int(p_name.split('.')[1])
                target_block = self.blocks[block_idx]
                if 'ln0' in p_name:
                    target = target_block.ln0
                elif 'ln1' in p_name:
                    target = target_block.ln1
                elif 'ln2' in p_name:
                    target = target_block.ln2
                elif 'att.time_maa_x' in p_name:
                    target = target_block.att.time_maa_x
                elif 'att.time_maa_w1' in p_name:
                    target = target_block.att.time_maa_w1
                elif 'att.time_maa_w2' in p_name:
                    target = target_block.att.time_maa_w2
                elif 'att.time_maa_w' in p_name:
                    target = target_block.att.time_maa_w
                elif 'att.time_maa_k' in p_name:
                    target = target_block.att.time_maa_k
                elif 'att.time_maa_v' in p_name:
                    target = target_block.att.time_maa_v
                elif 'att.time_maa_r' in p_name:
                    target = target_block.att.time_maa_r
                elif 'att.time_maa_g' in p_name:
                    target = target_block.att.time_maa_g
                elif 'att.time_decay_w1' in p_name:
                    target = target_block.att.time_decay_w1
                elif 'att.time_decay_w2' in p_name:
                    target = target_block.att.time_decay_w2
                elif 'att.time_decay' in p_name:
                    target = target_block.att.time_decay
                elif 'att.time_faaaa' in p_name:
                    target = target_block.att.time_faaaa
                elif 'att.receptance' in p_name:
                    target = target_block.att.receptance
                elif 'att.key' in p_name:
                    target = target_block.att.key
                elif 'att.value' in p_name:
                    target = target_block.att.value
                elif 'att.output' in p_name:
                    target = target_block.att.output
                elif 'att.gate' in p_name:
                    target = target_block.att.gate
                elif 'att.ln_x' in p_name:
                    target = target_block.att.ln_x

                elif 'ffn.time_maa_k' in p_name:
                    target = target_block.ffn.time_maa_k
                elif 'ffn.time_maa_r' in p_name:
                    target = target_block.ffn.time_maa_r
                elif 'ffn.key' in p_name:
                    target = target_block.ffn.key
                elif 'ffn.receptance' in p_name:
                    target = target_block.ffn.receptance
                elif 'ffn.value' in p_name:
                    target = target_block.ffn.value
                else:
                    raise NotImplementedError
            else:
                if 'emb.weight' in p_name:
                    target = self.emb
                elif 'ln_out' in p_name:
                    target = self.ln_out
                elif 'head' in p_name:
                    target = self.head
                else:
                    raise NotImplementedError

            self.load_parameters(w, p_name, target)

        print('Model loaded successfully.')

    def save_state(self, path):

        state_list = []
        for block in self.blocks:
            time_mix_shift = block.att.token_shift_state.squeeze(0)
            channel_mix_shift = block.ffn.token_shift_state.squeeze(0)
            state = block.att.state_encoder.forward()

            state_list.append(time_mix_shift)
            state_list.append(state)
            state_list.append(channel_mix_shift)

        torch.save(state_list, path)

        print('State saved.')

    def load_state(self, path):

        saved_weights = torch.load(path)

        for name, param in self.named_parameters():
            pdb.set_trace()
            if name in saved_weights:
                param.data.copy_(saved_weights[name].data)

        print('State loaded successfully.')
