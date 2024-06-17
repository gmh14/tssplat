import torch


class AdamUniform(torch.optim.Optimizer):
    """
    Variant of Adam with uniform scaling by the second moment.

    Instead of dividing each component by the square root of its second moment,
    we divide all of them by the max.
    """

    def __init__(self, params, grad_limit=False, grad_limit_values=[0.05, 0.01], grad_limit_iters=[4000], lr=0.1, betas=(0.9, 0.999)):
        defaults = dict(lr=lr, betas=betas)
        self.grad_limit = grad_limit
        super(AdamUniform, self).__init__(params, defaults)

        self.cc = 0
        
        if grad_limit:
            self.grad_limit_values = grad_limit_values
            self.grad_limit_iters = grad_limit_iters
            self.grad_limit_ptr = 0

    def __setstate__(self, state):
        super(AdamUniform, self).__setstate__(state)

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]

                state["step"] = 0
                state["g1"] = torch.zeros_like(p.data)
                state["g2"] = torch.zeros_like(p.data)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            # print(f"lr: {lr}")

            b1, b2 = group["betas"]
            for p in group["params"]:
                state = self.state[p]

                # Lazy initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["g1"] = torch.zeros_like(p.data)
                    state["g2"] = torch.zeros_like(p.data)

                g1 = state["g1"]
                g2 = state["g2"]
                state["step"] += 1

                grad = p.grad.data

                # print(f"ggrad:{torch.max(torch.abs(grad))}")

                g1.mul_(b1).add_(grad, alpha=1 - b1)
                g2.mul_(b2).add_(grad.square(), alpha=1 - b2)

                # print(f"g1:{torch.max(torch.abs(g1))}")
                # print(f"g2:{torch.max(torch.abs(g2))}")

                m1 = g1 / (1 - (b1 ** state["step"]))
                m2 = g2 / (1 - (b2 ** state["step"]))

                # print(f"m1:{torch.max(torch.abs(m1))}")
                # print(f"m2:{torch.max(torch.abs(m2))}")

                # This is the only modification we make to the original Adam algorithm
                gr = m1 / (1e-8 + m2.sqrt().max())

                if self.grad_limit:
                    m = self.grad_limit_values[self.grad_limit_ptr]
                    
                    if self.grad_limit_ptr < len(self.grad_limit_iters):
                        if self.cc >= self.grad_limit_iters[self.grad_limit_ptr]:
                            self.grad_limit_ptr += 1

                    # tet_spheres_ext.grad_limit(gr, 0.05, 0.05)
                    s = torch.max(torch.abs(gr))
                    if s > m:
                        gr.mul_(m / s)

                p.data.sub_(gr, alpha=lr)
                self.cc += 1
