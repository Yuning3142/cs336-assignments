import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr,"betas":betas,"eps":eps,"weight_decay":weight_decay}
        super().__init__(params, defaults)


    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                if "m" not in state:
                    m = torch.zeros_like(p)
                else:
                    m = state["m"]
                if "v" not in state:
                    v = torch.zeros_like(p)
                else:
                    v = state["v"]
                grad = p.grad.data
                t += 1
                a_t = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)
                p.data -= lr * weight_decay * p.data
                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * grad ** 2
                p.data -= a_t * m/(torch.sqrt(v) + eps)
                state["t"] = t
                state["m"] = m
                state["v"] = v
        return loss

