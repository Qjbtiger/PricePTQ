"""INT and FP weight fake quantizers used in Price-KL experiments."""

from __future__ import annotations

import torch


def get_quant_params(quant_type: str) -> tuple[int, bool]:
    """Return the finite-format parameters for a quantizer name."""
    quant_params = {
        "fp8": (8, True),
        "fp4": (4, True),
        "int8": (8, False),
        "int4": (4, False),
        "int2": (2, False),
    }
    if quant_type not in quant_params:
        raise ValueError(f"Unknown quantization type: {quant_type}")
    return quant_params[quant_type]



def fake_quantize_int(
    tensor: torch.Tensor,
    num_bits: int,
    granularity: str,
) -> torch.Tensor:
    """Return integer-grid fake quantization with the requested bitwidth and scale sharing."""
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1

    if granularity == "per-tensor":
        max_val = tensor.abs().max().clamp(min=1e-8)
        scale = max_val / qmax
        q_tensor = torch.clamp(torch.round(tensor / scale), qmin, qmax)
        return q_tensor * scale

    if granularity == "per-channel":
        if tensor.dim() == 1:
            max_val = tensor.abs().max().clamp(min=1e-8)
            scale = max_val / qmax
            q_tensor = torch.clamp(torch.round(tensor / scale), qmin, qmax)
            return q_tensor * scale

        original_shape = tensor.shape
        tensor_2d = tensor.view(tensor.shape[0], -1)
        max_vals = tensor_2d.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
        scale = max_vals / qmax
        q_tensor = torch.clamp(torch.round(tensor_2d / scale), qmin, qmax)
        return (q_tensor * scale).view(original_shape)

    raise ValueError(f"Unknown granularity: {granularity}")



def fake_quantize_float(
    tensor: torch.Tensor,
    num_bits: int,
    granularity: str,
) -> torch.Tensor:
    """Return local-ULP floating-point fake quantization with the requested format and scale sharing."""
    if num_bits == 8:
        mantissa_bits = 3
        max_val = 448.0
    elif num_bits == 4:
        mantissa_bits = 1
        max_val = 6.0
    else:
        raise ValueError(f"Unsupported float quantization bits: {num_bits}")

    num_levels = 2**mantissa_bits

    if granularity == "per-tensor":
        tensor_max = tensor.abs().max().clamp(min=1e-8)
        scale = tensor_max / max_val
        scaled_tensor = tensor / scale
        sign = scaled_tensor.sign()
        abs_tensor = scaled_tensor.abs().clamp(min=1e-8, max=max_val)
        log_scale = torch.floor(torch.log2(abs_tensor + 1e-8))
        log_scale = torch.clamp(log_scale, min=-14)
        mantissa = abs_tensor / (2**log_scale)
        q_mantissa = torch.round(mantissa * num_levels) / num_levels
        q_tensor = sign * q_mantissa * (2**log_scale)
        return q_tensor * scale

    if granularity == "per-channel":
        if tensor.dim() == 1:
            return fake_quantize_float(tensor, num_bits, "per-tensor")

        original_shape = tensor.shape
        tensor_2d = tensor.view(tensor.shape[0], -1)
        channel_max = tensor_2d.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
        scale = channel_max / max_val
        scaled_tensor = tensor_2d / scale
        sign = scaled_tensor.sign()
        abs_tensor = scaled_tensor.abs().clamp(min=1e-8, max=max_val)
        log_scale = torch.floor(torch.log2(abs_tensor + 1e-8))
        log_scale = torch.clamp(log_scale, min=-14)
        mantissa = abs_tensor / (2**log_scale)
        q_mantissa = torch.round(mantissa * num_levels) / num_levels
        q_tensor = sign * q_mantissa * (2**log_scale)
        return (q_tensor * scale).view(original_shape)

    raise ValueError(f"Unknown granularity: {granularity}")



def fake_quantize(
    tensor: torch.Tensor,
    quant_type: str,
    granularity: str,
) -> torch.Tensor:
    """Dispatch weight fake quantization by finite format and scale-sharing granularity."""
    num_bits, is_float = get_quant_params(quant_type)
    if is_float:
        return fake_quantize_float(tensor, num_bits, granularity)
    return fake_quantize_int(tensor, num_bits, granularity)

