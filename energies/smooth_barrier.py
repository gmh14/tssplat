import pypgo
import torch
import numpy as np
import math

from tet_spheres import tet_spheres_ext  # NOQA
# from .torch_energies import compute_energy, compute_G_matrix, compute_L_matrix, get_torch_sparse_mat

class SmoothnessBarrierFunc(torch.autograd.Function):
    @staticmethod
    def forward(x_cur, tet_sp, c1, c2, order):
        return tet_spheres_ext.forward(x_cur, tet_sp, c1, c2, order)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x_cur, tet_sp, c1, c2, order = inputs
        ctx.save_for_backward(x_cur)
        ctx.constants = (tet_sp, c1, c2, order)

    @staticmethod
    def backward(ctx, grad_output):
        # Here we must handle None grad_output tensor. In this case we
        # can skip unnecessary computations and just return None.
        if grad_output is None:
            return None, None, None, None, None

        (x_cur,) = ctx.saved_tensors
        tet_sp, c1, c2, order = ctx.constants
        grad_final = tet_spheres_ext.backward(
            grad_output, x_cur, tet_sp, c1, c2, int(order))
        return grad_final, None, None, None, None
    

class SmoothnessBarrierEnergy(torch.nn.Module):
    def __init__(self, tet_v, tet_f, FLAGS) -> None:
        super().__init__()

        v_flat = tet_v.flatten().astype(np.float32)
        f_flat = tet_f.flatten().astype(np.int32)
        self.tet_sp = tet_spheres_ext.TetSpheres(v_flat, f_flat)
        
        self.FLAGS = FLAGS

        # smoothfunc
        self.smooth_eng_func = SmoothnessBarrierFunc()

    def coeff_scheduler(self, it):
        smooth_coeff = self.FLAGS.smooth_eng_coeff
        barrier_coeff = self.FLAGS.barrier_coeff
        multiplier = math.pow(
            2,
            abs(math.sin(min(it / 300.0 / 4 * 0.5 * math.pi, 0.5 * math.pi)))
            * 4,
        )

        smooth_coeff *= multiplier
        barrier_coeff *= multiplier
        return smooth_coeff, barrier_coeff

    def forward(self, x, it, c1, c2):
        order = 2
        if it > self.FLAGS.increase_order_iter:
            order = 4

        eng = self.smooth_eng_func.apply(x, self.tet_sp, c1, c2, order)
        
        return eng
