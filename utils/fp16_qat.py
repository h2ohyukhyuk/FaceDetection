"""FP16 rounding-aware training with FP32 master weights and STE gradients.

Simulates FP16 representation at parameter/input/operator-output boundaries.
FP32 arithmetic and BatchNorm running statistics are retained: this is not
bit-exact emulation of CUDA/XNNPACK FP16 kernels or fused Conv-BN deployment.
No autocast or GradScaler is needed because backward computation stays FP32.
"""
import torch
from torch import nn, fx
from torch.func import functional_call


class _RoundFP16STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        rounded = value.to(torch.float16).to(value.dtype)
        if not torch.isfinite(rounded).all():
            raise FloatingPointError('FP16 QAT encountered NaN/Inf or FP16 overflow; inspect activations/learning rate.')
        return rounded

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def fake_quant_fp16(value):
    """Round floating tensors to FP16, preserving dtype and identity gradients."""
    if isinstance(value, torch.Tensor):
        return _RoundFP16STE.apply(value) if value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(fake_quant_fp16(v) for v in value)
    if isinstance(value, list):
        return [fake_quant_fp16(v) for v in value]
    if isinstance(value, dict):
        return {k: fake_quant_fp16(v) for k, v in value.items()}
    return value


class FP16QATModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        if any(p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
            raise ValueError('FP16 QAT requires FP32 master parameters; do not call model.half().')
        self.graph_model = fx.symbolic_trace(model)
        # Round inputs and every explicit operator output, including residual
        # additions and functional relu, rather than only module boundaries.
        graph = self.graph_model.graph
        for node in list(graph.nodes):
            if node.op not in ('placeholder', 'call_module', 'call_function', 'call_method'):
                continue
            with graph.inserting_after(node):
                rounded = graph.call_function(fake_quant_fp16, (node,))
            node.replace_all_uses_with(rounded)
            rounded.args = (node,)
        graph.lint()
        self.graph_model.recompile()

    def forward(self, *args, **kwargs):
        # Functional substitution retains the optimizer's FP32 parameters.
        parameters = {name: fake_quant_fp16(p) for name, p in self.graph_model.named_parameters()}
        return functional_call(self.graph_model, parameters, args, kwargs, strict=False)

    def state_dict(self, *args, **kwargs):
        # Preserve original Network checkpoint keys; fake quant ops have no state.
        return self.graph_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.graph_model.load_state_dict(state_dict, strict=strict, assign=assign)


def prepare_fp16_qat(model):
    return FP16QATModel(model)
