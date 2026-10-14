"""Deterministic searches over native, serialized numerical representations.

Candidates are evaluated sequentially. Rows are independent; activation samples
are tiled on their existing device and copied only a tile at a time. No Hessian,
candidate stack, model download or inference-time operation is introduced.
"""
from __future__ import annotations

import math
import torch

from native_quantizer.errors import CalibrationError


def select(mask, candidate, original):
    """Select rows without float8 arithmetic (not implemented by torch.where)."""
    mask = mask.reshape(-1, *([1] * (original.ndim - 1)))
    if original.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu):
        return torch.where(mask, candidate.view(torch.uint8), original.view(torch.uint8)).view(original.dtype)
    return torch.where(mask, candidate, original)


class ErrorObjective:
    def __init__(self, weight, config, activations=None, transform=None):
        self.weight = weight.float()
        self.activations = activations if config.objective != 'weight' else None
        self.transform = transform
        if config.objective == 'activation' and activations is None:
            raise CalibrationError('activation MSE requires calibration for every targeted layer')
        if self.activations is not None:
            x = self.activations
            if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != weight.shape[1]:
                raise CalibrationError('MSE calibration must be nonempty [samples,input_features]')
            if not bool(torch.isfinite(x).all()):
                raise CalibrationError('MSE calibration contains nonfinite values')

    def tiles(self):
        # Bound both input and output tiles; never transfer the full calibration.
        size = max(1, min(16, 262144 // max(self.weight.shape)))
        for i in range(0, len(self.activations), size):
            x = self.activations[i:i + size].to(device=self.weight.device, dtype=torch.float32)
            yield self.transform(x) if self.transform is not None else x

    def score(self, reconstructed):
        delta = reconstructed.float() - self.weight
        if self.activations is None:
            result = delta.double().square().sum(-1)
        else:
            result = torch.zeros(len(delta), dtype=torch.float64, device=delta.device)
            for x in self.tiles():
                result += (delta @ x.T).double().square().sum(-1)
        return torch.where(torch.isfinite(result), result, torch.full_like(result, float('inf')))

    def fit(self, unit):
        """Exact LS outer scale for fixed decoded codes, optionally in X space."""
        if self.activations is None:
            numerator = (unit.double() * self.weight.double()).sum(-1)
            denominator = unit.double().square().sum(-1)
        else:
            numerator = torch.zeros(len(unit), dtype=torch.float64, device=unit.device)
            denominator = torch.zeros_like(numerator)
            for x in self.tiles():
                u, target = (unit @ x.T).double(), (self.weight @ x.T).double()
                numerator += (u * target).sum(-1)
                denominator += u.square().sum(-1)
        return numerator, denominator


def row_scale_search(weight, data, scale, decode, config, activations=None, transform=None):
    """Search row scales using exact codes returned by decode(scale).

    decode returns (serialized codes, decoded unit values). Baseline wins ties.
    LS proposals are re-encoded and re-scored, never accepted on an ideal grid.
    """
    objective = ErrorObjective(weight, config, activations, transform)
    best_data, best_scale = data, scale.reshape(-1, 1)
    _, unit = decode(best_scale)
    # The caller's original codes may differ at ties; decode the baseline once
    # before calling this helper when a native encoder has a different tie rule.
    best_error = objective.score(unit * best_scale)
    initial_scale = best_scale

    def consider(proposal):
        nonlocal best_data, best_scale, best_error
        proposal = proposal.float().clamp(min=1e-30)
        candidate, values = decode(proposal)
        error = objective.score(values * proposal)
        valid = torch.isfinite(proposal).all(-1) & (error < best_error)
        best_data = select(valid, candidate, best_data)
        best_scale = select(valid, proposal, best_scale)
        best_error = torch.where(valid, error, best_error)

    for factor in config.factors():
        consider(initial_scale * factor)
    for _ in range(config.iterations):
        _, values = decode(best_scale)
        numerator, denominator = objective.fit(values)
        proposed = (numerator / denominator.clamp(min=1e-300)).float().reshape(-1, 1)
        consider(proposed)
    return best_data.contiguous(), best_scale.contiguous()


def fp8_values(weight, scale, dtype):
    limit = torch.finfo(dtype).max
    codes = (weight / scale).clamp(-limit, limit).to(dtype)
    return codes, codes.float() * scale


def tensor_scale_search(rows, initial_scale, dtype, config, activations=None):
    """Exact streamed FP8 tensor objective; one tensor scale across all chunks."""
    def error(scale):
        total = 0.0
        for weight in rows():
            _, dq = fp8_values(weight, scale, dtype)
            total += float(ErrorObjective(weight, config, activations).score(dq).sum())
        return total

    best, best_error = initial_scale, error(initial_scale)
    for factor in config.factors():
        candidate = initial_scale * factor
        candidate_error = error(candidate)
        if math.isfinite(candidate_error) and candidate_error < best_error:
            best, best_error = candidate, candidate_error
    for _ in range(config.iterations):
        numerator, denominator = 0.0, 0.0
        for weight in rows():
            codes, _ = fp8_values(weight, best, dtype)
            n, d = ErrorObjective(weight, config, activations).fit(codes.float())
            numerator += float(n.sum())
            denominator += float(d.sum())
        candidate = torch.full_like(best, max(numerator / max(denominator, 1e-300), 1e-30))
        candidate_error = error(candidate)
        if math.isfinite(candidate_error) and candidate_error < best_error:
            best, best_error = candidate, candidate_error
    return best


def nearest_grid(weight, levels, channel):
    """Nearest index in a sorted per-group grid; lower index wins ties."""
    rows, groups, size = weight.shape
    grid = levels.reshape(rows * groups, -1).contiguous()
    target = (weight / channel[:, None, None]).reshape(rows * groups, size).contiguous()
    pos = torch.searchsorted(grid, target)
    lo, hi = (pos - 1).clamp(0, grid.shape[1] - 1), pos.clamp(0, grid.shape[1] - 1)
    a, b = torch.gather(grid, 1, lo), torch.gather(grid, 1, hi)
    return torch.where((target - b).abs() < (target - a).abs(), hi, lo).reshape(rows, groups, size)


def optimize_grouped(weight, tensors, fmt, group_size, config, activations=None, transform=None):
    """Refine FP8 relative scales and FP32 channel scales on the INT8 decode grid.

    Keep the existing fitted W4 codebook / Kitchen W6 search as the seed. Group
    candidates use rounded/clamped INT8 levels, not ideal real-valued levels.
    """
    from native_quantizer.codecs.packing import unpack_w4
    from native_quantizer.codecs.w6a8 import unpack_w6, pack_w6
    from native_quantizer.identity import FORMAT_W6A8
    six = fmt == FORMAT_W6A8
    n, k = weight.shape
    grouped = weight.float().reshape(n, -1, group_size)
    table = (torch.arange(64, device=weight.device).float() - 32 if six
             else tensors['_codebook'].float())
    rel, channel = tensors['_s_rel'], tensors['_s_channel']
    codes = (unpack_w6(tensors['']) if six else unpack_w4(tensors[''])).reshape_as(grouped).long()
    objective = ErrorObjective(weight, config, activations, transform)

    def values(c, r):
        return (table[c] * r.float().unsqueeze(-1)).round().clamp(-127, 127).reshape(n, k)

    best_codes, best_rel, best_channel = codes, rel, channel
    best_error = objective.score(values(codes, rel) * channel[:, None])

    def consider(c, r, ch):
        nonlocal best_codes, best_rel, best_channel, best_error
        error = objective.score(values(c, r) * ch[:, None])
        mask = error < best_error
        best_codes, best_rel, best_channel = (select(mask, c, best_codes), select(mask, r, best_rel),
                                            select(mask, ch, best_channel))
        best_error = torch.where(mask, error, best_error)

    # Independent block-MSE proposal plus rowwise candidates for correlated X.
    group_codes, group_rel = codes, rel
    group_error = (values(codes, rel).reshape_as(grouped) * channel[:, None, None] - grouped).double().square().sum(-1)
    for factor in config.factors():
        r = (rel.float() * factor).clamp(0, torch.finfo(rel.dtype).max).to(rel.dtype)
        levels = (table[None, None, :] * r.float().unsqueeze(-1)).round().clamp(-127, 127)
        c = nearest_grid(grouped, levels, channel)
        err = (values(c, r).reshape_as(grouped) * channel[:, None, None] - grouped).double().square().sum(-1)
        better = err < group_error
        group_codes = torch.where(better[..., None], c, group_codes)
        group_rel = torch.where(better, r.view(torch.uint8), group_rel.view(torch.uint8)).view(rel.dtype)
        group_error = torch.where(better, err, group_error)
        consider(c, r, channel)
    consider(group_codes, group_rel, channel)
    # Fixed integer levels have an analytic optimal outer scale. The existing
    # group search remains authoritative; no duplicate codebook fitting.
    for _ in range(config.iterations):
        unit = values(best_codes, best_rel)
        numerator, denominator = objective.fit(unit)
        ch = (numerator / denominator.clamp(min=1e-300)).float().clamp(min=1e-30)
        levels = (table[None, None, :] * best_rel.float().unsqueeze(-1)).round().clamp(-127, 127)
        c = nearest_grid(grouped, levels, ch)
        consider(c, best_rel, ch)
    unsigned = best_codes.reshape(n, k)
    packed = (pack_w6(unsigned) if six else
              ((unsigned[:, 0::2] & 15) | ((unsigned[:, 1::2] & 15) << 4)).to(torch.int8))
    return {**tensors, '': packed.contiguous(), '_s_rel': best_rel.contiguous(),
            '_s_channel': best_channel.contiguous()}


def optimize_block_native(weight, spec, payload, reconstructed, kitchen, config, activations=None):
    """NVFP4 E2M1/FP8 scales and MXFP8 E4M3/E8M0 scales, native swizzle.

    Kitchen still owns swizzling and the format's float packing. Search never
    changes a block scale to an unrepresentable real value.
    """
    from comfy_kitchen.float_utils import from_blocked, to_blocked, _f32_to_floatx_unpacked, pack_uint4
    nv = spec.cli == 'nvfp4'
    n, k = weight.shape
    size = 16 if nv else 32
    groups = k // size
    raw_scale = payload['_scale'] if nv else payload['_scale'].view(torch.uint8)
    plain = from_blocked(raw_scale, num_rows=n, num_cols=groups)
    global_scale = payload['_scale_2'] if nv else None
    objective = ErrorObjective(weight, config, activations)
    best_error = objective.score(reconstructed)
    best_q, best_plain, best_dq = payload[''], plain, reconstructed
    grouped = weight.reshape(n, groups, size)

    def encode(p):
        scale = p.float() * global_scale if nv else torch.pow(2.0, p.float() - 127)
        scaled = grouped / torch.where(scale > 0, scale, torch.ones_like(scale))[..., None]
        if nv:
            unpacked = _f32_to_floatx_unpacked(scaled.clamp(-6, 6).reshape(n, k), 2, 1)
            q = pack_uint4(unpacked, hi_first=True)
        else:
            q = scaled.clamp(-448, 448).reshape(n, k).to(torch.float8_e4m3fn)
        swizzled = to_blocked(p, flatten=False)
        params = (kitchen.Params(scale=global_scale, block_scale=swizzled, orig_dtype=torch.float32, orig_shape=(n, k)) if nv
                  else kitchen.Params(scale=swizzled.view(torch.float8_e8m0fnu), orig_dtype=torch.float32, orig_shape=(n, k)))
        return q, kitchen.dequantize(q, params).float()

    group_plain = plain
    group_error = (reconstructed.reshape_as(grouped) - grouped).double().square().sum(-1)
    candidates = ((plain.float() * factor).clamp(0, 448).to(plain.dtype) for factor in config.factors()) if nv else (
        (plain.to(torch.int32) + offset).clamp(0, 254).to(torch.uint8) for offset in (-2, -1, 1, 2))
    for p in candidates:
        q, dq = encode(p)
        err = (dq.reshape_as(grouped) - grouped).double().square().sum(-1)
        mask = err < group_error
        group_plain = torch.where(mask, p.view(torch.uint8), group_plain.view(torch.uint8)).view(plain.dtype)
        group_error = torch.where(mask, err, group_error)
        error = objective.score(dq)
        better = error < best_error
        best_q, best_plain, best_dq = select(better, q, best_q), select(better, p, best_plain), select(better, dq, best_dq)
        best_error = torch.where(better, error, best_error)
    q, dq = encode(group_plain)
    error = objective.score(dq)
    better = error < best_error
    best_q, best_plain, best_dq = select(better, q, best_q), select(better, group_plain, best_plain), select(better, dq, best_dq)
    best_error = torch.where(better, error, best_error)
    # NVFP4's tensor scale is also optimizable, while MXFP8 has no such field.
    if nv:
        for _ in range(config.iterations):
            numerator, denominator = objective.fit(best_dq / global_scale)
            proposed = (numerator.sum() / denominator.sum().clamp(min=1e-300)).float().clamp(min=1e-30)
            params = kitchen.Params(scale=proposed, block_scale=to_blocked(best_plain, flatten=False),
                                    orig_dtype=torch.float32, orig_shape=(n, k))
            dq = kitchen.dequantize(best_q, params).float()
            error = objective.score(dq)
            if bool(torch.isfinite(proposed)) and float(error.sum()) < float(best_error.sum()):
                global_scale, best_dq, best_error = proposed, dq, error
    return {**payload, '': best_q.contiguous(), '_scale': to_blocked(best_plain, flatten=False).contiguous(),
            **({'_scale_2': global_scale} if nv else {})}, best_dq
