"""Load the official routing code, unchanged, from the pinned checkouts in data/ext/.

Megatron-Core (NVIDIA/Megatron-LM core_v0.16.1, 55ac708, BSD-3-Clause), file
megatron/core/transformer/moe/moe_utils.py: the top-level functions listed in MEGATRON_FUNCS
are taken from the file's syntax tree and executed as written. The file's package imports
(parallel_state, Transformer Engine, CUDA graphs, ...) are not needed by these functions
except torch.distributed.all_reduce in get_updated_expert_bias; for that a one-process gloo
group is started, so the all-reduce is the identity, and parallel_state is a stub that
returns the default group.

DeepSeek-V3 (deepseek-ai/DeepSeek-V3 9b4e978, MIT), inference/model.py: class Gate is
taken the same way. It is used only by tests/check_official.py to check that Megatron's
sigmoid + expert-bias + group-limited routing selects the same experts with the same weights.
"""
import ast, math, os, tempfile, types
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ext")
MEGATRON_FILE = os.path.join(ROOT, "megatron-core", "megatron", "core", "transformer", "moe", "moe_utils.py")
DSV3_FILE = os.path.join(ROOT, "deepseek-v3", "inference", "model.py")
MEGATRON_FUNCS = ["switch_load_balancing_loss_func", "z_loss_func", "get_capacity",
                  "group_limited_topk", "topk_routing_with_score_function",
                  "compute_routing_scores_for_aux_loss", "apply_router_token_dropping",
                  "get_updated_expert_bias"]


def _extract(path, names, namespace):
    src = open(path).read()
    tree = ast.parse(src)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            found[node.name] = ast.get_source_segment(src, node)
    missing = set(names) - set(found)
    assert not missing, f"not found in {path}: {missing}"
    for n in names:
        exec(compile(found[n], path, "exec"), namespace)
    return found


def _init_one_process_group():
    if dist.is_available() and not dist.is_initialized():
        store = tempfile.mktemp(prefix="moe_pg_")
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=0, world_size=1)


_parallel_state = types.SimpleNamespace(
    get_tensor_and_data_parallel_group=lambda **kw: None)  # None = default (one-process) group

mg = types.SimpleNamespace()
_ns = dict(torch=torch, math=math, Optional=Optional, Tuple=Tuple, List=List, Union=Union,
           HAVE_TE=False, parallel_state=_parallel_state,
           fused_moe_aux_loss=None, fused_topk_with_score_function=None,
           fused_compute_score_for_moe_aux_loss=None)
MEGATRON_SOURCES = _extract(MEGATRON_FILE, MEGATRON_FUNCS, _ns)
for _n in MEGATRON_FUNCS:
    setattr(mg, _n, _ns[_n])
_init_one_process_group()


def load_deepseek_gate():
    """Return DeepSeek-V3's Gate class (inference/model.py) with `linear` = F.linear (bf16/fp32 path)."""
    ns = dict(torch=torch, nn=nn, F=F, Tuple=Tuple, Optional=Optional, ModelArgs=object,
              linear=lambda x, w, b=None, scale_fmt=None: F.linear(x, w, b))
    _extract(DSV3_FILE, ["Gate"], ns)
    return ns["Gate"]
