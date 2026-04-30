from yunmeng.solvers.interfaces import IOperator
import torch
import numpy as np


class PressureProjection(IOperator, torch.autograd.Function):
    """
    自定义压力投影步骤，手动定义前向和反向传播
    """

    @staticmethod
    def forward(ctx, velocity, dt):
        # 前向：求解泊松方程 ∇·v = ∇²p
        pressure = solve_poisson_fft(velocity, dt)  # 伪谱法求解
        ctx.save_for_backward(velocity, pressure, dt)
        return pressure_corrected_velocity(velocity, pressure, dt)

    @staticmethod
    def backward(ctx, grad_output):
        # 反向：使用伴随方法（adjoint method）计算梯度
        # 避免存储整个前向计算的中间状态
        velocity, pressure, dt = ctx.saved_tensors

        # 求解伴随方程（adjoint equation）
        grad_input = solve_adjoint_poisson(grad_output, velocity, pressure)
        return grad_input, None
