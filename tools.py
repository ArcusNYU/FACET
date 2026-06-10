"""
tools.py

TEMPORARY, NON-ESSENTIAL debugging instrumentation for the FACET workspace.
  - tools.py is a scratch space for throwaway "is this actually correct?" probes:
    shape / dtype / device / value-range sanity checks sprinkled into the pipeline
    while a stage is being brought up. Nothing on the training or eval critical
    path should import from here; the calls are meant to be deleted (or left
    dormant behind a flag) once a stage has been verified.

Primary entry point:
    tools.inspect(title, {"name": tensor_or_list_or_dict, ...})
"""
# FIXME: 发行github时 tools.py的注释改成简要说明 因为最后tools.py的功能不止一个

from __future__ import annotations

from typing import Any, Mapping

import torch


# -----------------------------------------------------------------------------
# Single-tensor summary
# -----------------------------------------------------------------------------
def _tensor_line(t: torch.Tensor) -> str:
    """
    One-line dtype / device / shape / range / mean(+std) summary for a tensor.

    Stats are computed in float32 so bf16 / fp16 / integer tensors don't
    under/overflow, and non-finite values (NaN / Inf) are flagged loudly.
    """
    d = t.detach()
    shape = tuple(d.shape)
    head = f"shape={shape} dtype={d.dtype} device={d.device}"

    if d.numel() == 0:
        return f"{head} (empty)"

    f = d.float()
    finite = torch.isfinite(f)
    n_bad = int((~finite).sum().item())

    # Reduce over finite entries only, so a stray NaN doesn't poison min/max/mean.
    ff = f[finite] if n_bad else f
    if ff.numel() == 0:
        return f"{head}  [!] all non-finite ({n_bad} NaN/Inf)"

    vmin = ff.min().item()
    vmax = ff.max().item()
    vmean = ff.mean().item()
    vstd = ff.std(unbiased=False).item()  # unbiased=False -> 0.0 (not NaN) for 1 elem

    line = f"{head} min={vmin:+.4g} max={vmax:+.4g} mean={vmean:+.4g} std={vstd:.4g}"
    if n_bad:
        n_nan = int(torch.isnan(f).sum().item())
        n_inf = int(torch.isinf(f).sum().item())
        line += f"  [!] non-finite: nan={n_nan} inf={n_inf}"
    return line


# -----------------------------------------------------------------------------
# Recursive describe (tensor / list / tuple / dict / scalar)
# -----------------------------------------------------------------------------
def _describe(name: str, obj: Any, indent: int = 1) -> None:
    pad = "  " * indent
    if torch.is_tensor(obj):
        print(f"{pad}{name:<14}: {_tensor_line(obj)}")
    elif isinstance(obj, Mapping):
        print(f"{pad}{name} (dict, {len(obj)} keys)")
        for k, v in obj.items():
            _describe(str(k), v, indent + 1)
    elif isinstance(obj, (list, tuple)):
        kind = type(obj).__name__
        n = len(obj)
        all_tensors = n > 0 and all(torch.is_tensor(x) for x in obj)
        if all_tensors:
            shapes = [tuple(x.shape) for x in obj]
            dtypes = {str(x.dtype) for x in obj}
            devices = {str(x.device) for x in obj}
            same = "same" if len({tuple(s) for s in shapes}) == 1 else "ragged"
            print(
                f"{pad}{name} ({kind}[Tensor], len={n}, {same} "
                f"dtype={'/'.join(sorted(dtypes))} device={'/'.join(sorted(devices))})"
            )
            for i, x in enumerate(obj):
                _describe(f"[{i}]", x, indent + 1)
        else:
            print(f"{pad}{name} ({kind}, len={n})")
            for i, x in enumerate(obj):
                _describe(f"[{i}]", x, indent + 1)
    else:
        print(f"{pad}{name:<14}: {type(obj).__name__} = {obj!r}")


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------
def inspect(title: str, named: Mapping[str, Any], *, enabled: bool = True) -> None:
    """
    Pretty-print dtype / device / shape / range / mean for a bag of named values.

    Args:
        title:   header label for this probe (e.g. "prep + flowmatch @ step 0").
        named:   mapping of name -> tensor | list/tuple of tensors | dict | scalar.
        enabled: gate flag; pass False to make the call a cheap no-op so probes
                 can be left in place without editing the call site.

    Example:
        tools.inspect("prep", {"noisy": noisy, "t": t, "prompt": prompt_embeds})
    """
    if not enabled:
        return
    bar = "=" * 12
    print(f"\n{bar} [tools.inspect] {title} {bar}")
    for name, obj in named.items():
        _describe(name, obj)
    print("=" * (26 + len(title) + len(" [tools.inspect]  ")))
