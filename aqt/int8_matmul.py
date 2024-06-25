import triton
import triton.ops
import torch


def _int8_matmul(a: torch.Tensor, b: torch.Tensor):
    scale_a = a.abs().max(dim=1, keepdim=True).values / 127
    qa = (a / scale_a).to(torch.int8)
    scale_b = b.abs().max(dim=0, keepdim=True).values / 127
    qb = (b / scale_b).to(torch.int8)
    qout = triton.ops.matmul(qa, qb)
    return qout * scale_a * scale_b


class Int8Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        assert a.dim() == 2 and b.dim() == 2
        ctx.save_for_backward(a, b)
        return _int8_matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a = _int8_matmul(grad_output, b.t())
        if ctx.needs_input_grad[1]:
            grad_b = _int8_matmul(a.t(), grad_output)
        return grad_a, grad_b


int8_matmul = Int8Matmul.apply
