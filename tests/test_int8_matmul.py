import torch
from aqt import int8_matmul


def print_diff(name, a, b):
    print(f"{name}:")
    print(f"  max diff: {(a - b).abs().max()}")
    print(f"  mean diff: {(a - b).abs().mean()}")


def test_int8_matmul(dtype=torch.bfloat16, device="cuda"):
    a1 = torch.randn(64, 128, dtype=dtype, device=device)
    b1 = torch.randn(128, 256, dtype=dtype, device=device)
    a1.requires_grad = True
    b1.requires_grad = True

    a2 = a1.clone().detach().requires_grad_()
    b2 = b1.clone().detach().requires_grad_()

    qout = int8_matmul(a1, b1)
    qout.sum().backward()

    out = torch.matmul(a2, b2)
    out.sum().backward()

    print_diff("out", qout, out)
    print_diff("a.grad", a1.grad, a2.grad)
    print_diff("b.grad", b1.grad, b2.grad)


if __name__ == "__main__":
    test_int8_matmul()
