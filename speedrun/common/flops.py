"""Count training FLOPs exactly, the same on every machine.

count_step() runs one full training step (forward, backward and optimizer step, so Muon's Newton-Schulz matrix
multiplies are in it) with the real code and the real shapes under torch.utils.flop_counter.FlopCounterMode,
on the "meta" device: tensors with shapes but no numbers, so it needs no memory and takes a second.
FlopCounterMode counts matrix multiplies and attention. It does not count elementwise work (norms, softmax,
activations, Adam updates), which is a small share at these sizes. Evaluation is not part of the budget.

Why meta and not the GPU: on mps, linear layers run their backward pass as one fused op that FlopCounterMode does
not know, so it misses about two thirds of the matrix FLOPs; on cpu the fused attention kernel counts as 0.
On meta every op is a plain matrix multiply, attention included, and the count matches attention_flops() below.
"""
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode


def count(fn):
    """Run fn() once and return the FLOPs FlopCounterMode saw."""
    with FlopCounterMode(display=False) as fc, sdpa_kernel(SDPBackend.MATH):
        fn()
    return fc.get_total_flops()


def count_step(build, train_step, batch_shape):
    """build() -> (model, optimizers); train_step(model, optimizers, x, y, dtype). Returns FLOPs of one step."""
    with torch.device("meta"):
        model, opts = build()
        x = torch.zeros(batch_shape, dtype=torch.long)
    return count(lambda: train_step(model, opts, x, x, torch.float32))


def count_step_inputs(build, train_step, make_inputs):
    """count_step for any inputs (frames, actions, observations): make_inputs() -> tuple of tensors, made on meta.
    train_step(model, optimizers, *inputs). Returns FLOPs of one step."""
    with torch.device("meta"):
        model, opts = build()
        inputs = make_inputs()
    return count(lambda: train_step(model, opts, *inputs))


def attention_flops(batch, seq_len, d_model, n_layers):
    """Causal attention counted as full T x T: forward 4*B*T*T*d per layer (QK^T and AV), backward twice that."""
    return 12 * batch * seq_len * seq_len * d_model * n_layers


def estimate_6nd(n_params, tokens, seq_len, d_model, n_layers):
    """The usual estimate: 6*N*D, plus 12*L*T*d per token for attention (PaLM appendix B)."""
    return 6 * n_params * tokens, 6 * n_params * tokens + 12 * n_layers * seq_len * d_model * tokens
