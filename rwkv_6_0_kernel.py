import torch
from torch.utils.cpp_extension import load

HEAD_SIZE = 64
CUDA_KERNEL_VERSION = 'v1'
T = 4096

wkv6state_cuda = load(name="wkv6state",
                      sources=["cuda/wkv6state_op.cpp", f"cuda/wkv6state_cuda_{CUDA_KERNEL_VERSION}.cu"],
                      verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3",
                                                       "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}",
                                                       f"-D_T_={T}"])


class WKV_6STATE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u, s):
        with torch.no_grad():
            assert r.dtype == torch.bfloat16
            assert k.dtype == torch.bfloat16
            assert v.dtype == torch.bfloat16
            assert w.dtype == torch.bfloat16
            assert u.dtype == torch.bfloat16
            assert s.dtype == torch.bfloat16
            assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            assert s.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            # ew = torch.sigmoid(-w.float()).contiguous()
            ctx.save_for_backward(r, k, v, ew, u, s)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16,
                            memory_format=torch.contiguous_format).uniform_(-100, 100)
            wkv6state_cuda.forward(B, T, C, H, r, k, v, ew, u, s, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == torch.bfloat16
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, ew, u, s = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16,
                             memory_format=torch.contiguous_format).uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16,
                             memory_format=torch.contiguous_format).uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16,
                             memory_format=torch.contiguous_format).uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16,
                             memory_format=torch.contiguous_format).uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16,
                             memory_format=torch.contiguous_format).uniform_(-100, 100)
            gs = torch.empty((B, H, C // H, C // H), device=gy.device, requires_grad=False, dtype=torch.bfloat16,
                             memory_format=torch.contiguous_format).uniform_(-100, 100)
            wkv6state_cuda.backward(B, T, C, H, r, k, v, ew, u, s, gy, gr, gk, gv, gw, gu, gs)
            gu = torch.sum(gu, 0).view(H, C // H)
            gs = torch.sum(gs, 0).view(H, C // H, C // H)
            return (None, None, None, None, gr, gk, gv, gw, gu, gs)


def RUN_CUDA_RWKV6_S(B, T, C, H, r, k, v, w, u, s):
    return WKV_6STATE.apply(B, T, C, H, r, k, v, w, u, s)
