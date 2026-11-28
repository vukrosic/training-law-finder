"""Count training FLOPs exactly, the same on every machine.

count_step() runs one full training step (forward, backward and optimizer step, so Muon's Newton-Schulz matrix
multiplies are in it) with the real code and the real shapes under torch.utils.flop_counter.FlopCounterMode,
on the "meta" device: tensors with shapes but no numbers, so it needs no memory and takes a second.
FlopCounterMode counts matrix multiplies and attention. Every other op counts 1 FLOP per element of its largest
input or output (norms, softmax, activations, Adam updates, embedding lookups: under 1% of the baseline step).
Without that, a matrix multiply written as a broadcast multiply and a sum, (x[..., None] * W).sum(-2), would
count as 0 FLOPs. Only views and pure data movement (copies, concatenation) count 0. Evaluation is not part of
the budget.

Why meta and not the GPU: on mps, linear layers run their backward pass as one fused op that FlopCounterMode does
not know, so it misses about two thirds of the matrix FLOPs; on cpu the fused attention kernel counts as 0.
On meta every op is a plain matrix multiply, attention included, and the count matches attention_flops() below.
"""
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.flop_counter import FlopCounterMode

aten = torch.ops.aten
MOVES = {aten._unsafe_view, aten.clone, aten.copy_, aten._to_copy, aten.cat, aten.stack, aten.detach,
         aten.lift_fresh, aten.zeros_like, aten.ones_like, aten.empty_like, aten.fill_, aten.zero_}  # count 0


class OtherOps(TorchDispatchMode):
    """Counts 1 FLOP per element of the largest input or output of every op FlopCounterMode does not count."""

    def __init__(self, skip):
        super().__init__()
        self.skip, self.flops = skip, 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        op = func.overloadpacket
        if op in self.skip or op in MOVES or func.is_view:
            return out
        ins = [t.numel() for t in tree_flatten((args, kwargs))[0] if isinstance(t, torch.Tensor)]
        if ins:  # ops with no tensor input make new tensors (zeros, arange): 0
            self.flops += max(ins + [t.numel() for t in tree_flatten(out)[0] if isinstance(t, torch.Tensor)])
        return out


def count(fn):
    """Run fn() once and return its FLOPs: FlopCounterMode's matrix multiplies plus 1 per element of other ops."""
    with FlopCounterMode(display=False) as fc, sdpa_kernel(SDPBackend.MATH), OtherOps(set(fc.flop_registry)) as o:
        fn()
    return fc.get_total_flops() + o.flops


def count_step(build, train_step, batch_shape):
    """build() -> (model, optimizers); train_step(model, optimizers, x, y, dtype). Returns FLOPs of one step."""
    with torch.device("meta"):
        model, opts = build()
        x = torch.zeros(batch_shape, dtype=torch.long)
    return count(lambda: train_step(model, opts, x, x, torch.float32))


def attention_flops(batch, seq_len, d_model, n_layers):
    """Causal attention counted as full T x T: forward 4*B*T*T*d per layer (QK^T and AV), backward twice that."""
    return 12 * batch * seq_len * seq_len * d_model * n_layers


def estimate_6nd(n_params, tokens, seq_len, d_model, n_layers):
    """The usual estimate: 6*N*D, plus 12*L*T*d per token for attention (PaLM appendix B)."""
    return 6 * n_params * tokens, 6 * n_params * tokens + 12 * n_layers * seq_len * d_model * tokens
