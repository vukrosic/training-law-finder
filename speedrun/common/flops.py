"""Count training FLOPs. Every op is charged, not only matrix multiplies.

Two ways to run the same counter (Meter, a TorchDispatchMode that sees every aten op after autograd, so the
backward pass and the optimizer step are in it):

    fast count (default, every research run)   one training step on the "meta" device: shapes, no numbers,
        flops.fast_count(build, train_step, ...)  about 1 s, cached per code and shape, so an lr sweep costs 0 s
    full count (--full-count, records)          every step on the real device while it trains, about 2% slower
        with meter.step(): train_step(...)

Each op is charged by its kind:

    matrix multiply, attention, convolution   2 FLOPs per multiply-add (torch.utils.flop_counter formulas),
                                              plus linear / linear_backward, which mps runs as single ops
    FFT                                       5 n log2(m) for n elements and transform length m
    sort, topk, kthvalue, median              n log2(m) for n elements and length m along the sorted dim
    eigh, eig, svd, qr, solve, inverse, ...   m * n * min(m, n) per matrix (about n^3)
    view ops (view, transpose, slice, ...)    0: they move no data
    gathers (embedding, index, gather)        1 FLOP per element read
    scatters (index_put, scatter_add, ...)    1 FLOP per element written
    every other op                            1 FLOP per element of its largest input or output
                                              (elementwise, reductions, cumsum, copies, optimizer updates)

For standard models (matrix multiplies, attention, convolution, embeddings, norms, activations, softmax,
elementwise) the fast count is exact: the full count on mps differs by 0.1% (bf16 casts). Ops outside that list
(FFT, decompositions, sorts, scans, scatters, heavy ops with no formula) go to report()["nonstandard"]; an entry
with any of them should check its real budget with --full-count. torch.compile and custom kernels hide their ops
from both counts and are not allowed in a step.
"""
import contextlib
import hashlib
import inspect
import json
import math
import os
from collections import defaultdict

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.flop_counter import flop_registry

aten = torch.ops.aten
COUNTER = "all-ops-v1"  # name of this counting rule; track.json and submission.json carry it
# words in an op name (split at "_") that mark heavy work; such an op without a formula is listed
HEAVY_WORDS = {"fft", "linalg", "conv", "convolution", "attention", "sort", "topk", "eig", "eigh", "svd", "qr",
               "solve", "lu", "cholesky", "inverse", "matmul", "mm", "einsum", "lstm", "gru", "rnn", "cdist", "kron",
               "det", "slogdet", "pinv", "lstsq"}
NOT_HEAVY = {"linalg_vector_norm", "linalg_matrix_norm", "linalg_norm"}  # norms: one pass over the input
# 0 FLOPs: no arithmetic, or views the schema does not mark as views
FREE = {"_local_scalar_dense", "detach", "alias", "lift_fresh", "_record_function_enter_new",
        "_record_function_exit", "empty", "empty_like", "empty_strided", "is_same_size", "sym_size", "sym_stride",
        "sym_numel", "sym_storage_offset", "set_", "resize_", "_unsafe_view", "_reshape_alias"}
# reads: 1 FLOP per element read, not per element of the table they read from
GATHERS = {"embedding", "index_select", "gather", "index", "take", "embedding_bag", "_embedding_bag",
           "_embedding_bag_forward_only", "masked_select"}
# writes into a big tensor: 1 FLOP per element written, not per element of the target
# ops a standard transformer does not use; an entry with any of them is told to run --full-count
NONSTANDARD = {"_fft_r2c", "_fft_c2r", "_fft_c2c", "sort", "argsort", "msort", "topk", "kthvalue", "median",
               "nanmedian", "mode", "cumsum", "cumsum_", "cumprod", "cumprod_", "logcumsumexp", "cummax", "cummin",
               "_cummax_helper", "_cummin_helper", "bincount", "histc", "unique", "_unique", "_unique2",
               "unique_consecutive", "unique_dim", "nonzero", "searchsorted", "bucketize"}
SCATTERS = {"scatter", "scatter_", "scatter_add", "scatter_add_", "scatter_reduce", "scatter_reduce_",
            "index_add", "index_add_", "index_put", "index_put_", "_index_put_impl_", "index_copy", "index_copy_",
            "masked_scatter", "masked_scatter_", "put", "put_", "index_reduce", "index_reduce_"}


def numel(x):
    return x.numel() if isinstance(x, torch.Tensor) else 0


def biggest(args):
    """Elements of the largest tensor argument; a list of tensors (foreach ops) counts as their sum."""
    best = 0
    for a in args:
        if isinstance(a, (list, tuple)):
            best = max(best, sum(numel(t) for t in a))
        else:
            best = max(best, numel(a))
    return best


def linear_flops(inp, weight, out):
    return 2 * out.numel() * weight.shape[-1]


def linear_backward_flops(inp, grad_out, weight, mask):
    mac = grad_out.numel() * weight.shape[-1]  # rows * out_features * in_features
    return 2 * mac * (int(mask[0]) + int(mask[1])) + (grad_out.numel() if mask[2] else 0)


def fft_flops(inp, out, dim):
    dims = [dim] if isinstance(dim, int) else list(dim)
    sizes = [max(inp.shape[d], out.shape[d]) for d in dims]
    m = math.prod(sizes)
    return int(5 * max(inp.numel(), out.numel()) * math.log2(max(m, 2)))


def sort_flops(inp, dim):
    m = inp.shape[dim] if inp.dim() else 1
    return int(inp.numel() * math.log2(max(m, 2)))


def matrix_flops(a):
    """m * n * min(m, n) per matrix for a (..., m, n)."""
    m, n = a.shape[-2], a.shape[-1]
    return a.numel() // max(m * n, 1) * m * n * min(m, n)


def sdpa_math_flops(q, k, v):
    """Attention for an op with no torch formula: QK^T and AV, full T x S."""
    b = q.numel() // (q.shape[-2] * q.shape[-1])
    return 2 * b * q.shape[-2] * k.shape[-2] * (q.shape[-1] + v.shape[-1])


def arg(func, args, kwargs, name, default):
    """The value of argument `name` of an aten op call, by its schema."""
    for i, a in enumerate(func._schema.arguments):
        if a.name == name:
            if i < len(args):
                return args[i]
            return kwargs.get(name, a.default_value if a.has_default_value() else default)
    return default


ATTENTION_LIKE = {"linear", "linear_backward", "_scaled_dot_product_attention_math_for_mps",
                  "_scaled_dot_product_flash_attention_for_cpu", "_scaled_dot_product_fused_attention_overrideable",
                  "_scaled_dot_product_flash_attention_for_cpu_backward",
                  "_scaled_dot_product_fused_attention_overrideable_backward"}
SPECIAL = ATTENTION_LIKE | {"_fft_r2c", "_fft_c2r", "_fft_c2c", "sort", "argsort", "msort", "topk", "kthvalue",
                            "median", "nanmedian", "mode", "_linalg_eigh", "linalg_eig", "_linalg_svd", "linalg_qr",
                            "linalg_cholesky_ex", "linalg_inv_ex", "linalg_solve_ex", "_linalg_solve_ex",
                            "linalg_lu_factor_ex", "linalg_lu", "_linalg_det", "_linalg_slogdet",
                            "linalg_solve_triangular", "triangular_solve", "cholesky_solve",
                            "_cholesky_solve_helper", "linalg_householder_product", "geqrf", "linalg_pinv",
                            "linalg_matrix_exp", "linalg_eigvals", "linalg_eigvalsh"}


def special(func, name, args, kwargs, out):
    """FLOPs for ops with their own formula, or None."""
    a = args
    if name == "linear":
        return linear_flops(a[0], a[1], out)
    if name == "linear_backward":
        return linear_backward_flops(*a[:4])
    if name in ("_fft_r2c", "_fft_c2r", "_fft_c2c"):
        return fft_flops(a[0], out, a[1])
    if name in ("sort", "argsort", "msort", "topk", "kthvalue", "median", "nanmedian", "mode"):
        x = a[0]
        return sort_flops(x, arg(func, args, kwargs, "dim", -1) if x.dim() else 0)
    if name in ("_linalg_eigh", "linalg_eig", "_linalg_svd", "linalg_qr", "linalg_cholesky_ex",
                "linalg_inv_ex", "linalg_solve_ex", "_linalg_solve_ex", "linalg_lu_factor_ex", "linalg_lu",
                "_linalg_det", "_linalg_slogdet", "linalg_solve_triangular", "triangular_solve",
                "cholesky_solve", "_cholesky_solve_helper", "linalg_householder_product", "geqrf",
                "linalg_pinv", "linalg_matrix_exp", "linalg_eigvals", "linalg_eigvalsh"):
        return matrix_flops(a[0])
    if name in ("_scaled_dot_product_attention_math_for_mps", "_scaled_dot_product_flash_attention_for_cpu",
                "_scaled_dot_product_fused_attention_overrideable"):
        return sdpa_math_flops(a[0], a[1], a[2])
    if name in ("_scaled_dot_product_flash_attention_for_cpu_backward",
                "_scaled_dot_product_fused_attention_overrideable_backward"):
        return 2 * sdpa_math_flops(a[1], a[2], a[3])
    return None


class Meter(TorchDispatchMode):
    """Counts the FLOPs of every op run inside `with meter.step():`."""

    def __init__(self):
        super().__init__()
        self.total = 0
        self.steps = []  # FLOPs of each counted step
        self.by_op = defaultdict(int)
        self.unknown_heavy = defaultdict(int)
        self.nonstandard = defaultdict(int)
        self._now = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        self._now += self.charge(func, args, kwargs, out)
        return out

    def charge(self, func, args, kwargs, out):
        name = func._overloadpacket.__name__
        if name in FREE or func.is_view:
            return 0
        packet = func._overloadpacket
        if packet in flop_registry:
            n = flop_registry[packet](*args, **kwargs, out_val=out)
        else:
            n = special(func, name, args, kwargs, out if isinstance(out, torch.Tensor) else
                        out[0] if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor)
                        else None)
            if n is None:
                outs, _ = tree_flatten(out)
                if name in GATHERS:
                    n = sum(numel(t) for t in outs)
                elif name in SCATTERS:
                    n = biggest(list(args[1:]) + list(kwargs.values()))
                else:
                    n = max(biggest(list(args) + list(kwargs.values())), sum(numel(t) for t in outs))
                if name not in NOT_HEAVY and HEAVY_WORDS & set(name.split("_")):
                    self.unknown_heavy[str(func)] += n
                    self.nonstandard[name] += int(n)
        n = int(n)
        if name in NONSTANDARD or name in SCATTERS or (name in SPECIAL and name not in ATTENTION_LIKE):
            self.nonstandard[name] += n
        self.by_op[name] += n
        return n

    @contextlib.contextmanager
    def step(self):
        """Count one training step. Steps may differ; every one is counted."""
        self._now = 0
        with self:
            yield self
        self.steps.append(self._now)
        self.total += self._now

    def count(self, fn):
        """Count fn() once without adding it to the training total (by_op still records it)."""
        total, steps = self.total, list(self.steps)
        with self.step():
            fn()
        n = self.steps[-1]
        self.total, self.steps = total, steps
        return n

    def report(self, top=12):
        kinds = sorted(set(self.steps))
        ops = sorted(self.by_op.items(), key=lambda kv: -kv[1])
        return dict(total=self.total, steps=len(self.steps), step_kinds=len(kinds),
                    per_step=kinds[:5], top_ops={k: v for k, v in ops[:top] if v},
                    unknown_heavy=dict(self.unknown_heavy), nonstandard=dict(self.nonstandard))


def code_key(module, skip=("LR", "COOLDOWN", "BUDGET")):
    """A key that changes when the code or a shape constant of a training script changes, but not its learning
    rates: the source of every class and function in it except main(), plus its UPPERCASE constants not starting
    with a prefix in skip."""
    parts = [COUNTER]
    for name, obj in sorted(vars(module).items()):
        if (inspect.isclass(obj) or inspect.isfunction(obj)) and obj.__module__ == module.__name__ and name != "main":
            parts.append(inspect.getsource(obj))
        elif name.isupper() and not name.startswith(skip) and isinstance(obj, (int, float, str, tuple)):
            parts.append(f"{name}={obj!r}")
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:16]


def fast_count(build, train_step, batch_shape, key=None, cache=None):
    """FLOPs of one training step counted on the meta device, and the report. With key and cache (a json path),
    the result is stored and reused: a second run with the same code and shapes costs nothing."""
    store = {}
    if key and cache and os.path.exists(cache):
        try:
            store = json.load(open(cache))
        except (OSError, ValueError):
            store = {}
        if key in store:
            return store[key]["flops"], store[key]["report"]
    with torch.device("meta"):
        model, opts = build()
        x = torch.zeros(batch_shape, dtype=torch.long)
    meter = Meter()
    n = meter.count(lambda: train_step(model, opts, x, x, torch.float32))
    rep = meter.report()
    if key and cache:
        store[key] = dict(flops=n, report=rep)
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        with open(cache, "w") as f:
            json.dump(store, f)
    return n, rep


def attention_flops(batch, seq_len, d_model, n_layers):
    """Causal attention counted as full T x T: forward 4*B*T*T*d per layer (QK^T and AV), backward twice that."""
    return 12 * batch * seq_len * seq_len * d_model * n_layers


def estimate_6nd(n_params, tokens, seq_len, d_model, n_layers):
    """The usual estimate: 6*N*D, plus 12*L*T*d per token for attention (PaLM appendix B)."""
    return 6 * n_params * tokens, 6 * n_params * tokens + 12 * n_layers * seq_len * d_model * tokens
