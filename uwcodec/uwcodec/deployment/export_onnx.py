"""ONNX export for encoder deployment.

Export the lightweight encoder to ONNX for deployment on edge devices.
Supports INT8 quantization for STM32N6570.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def export_encoder_onnx(
    encoder,
    output_path: str | Path,
    input_size: int = 128,
    opset_version: int = 13,
    dynamic_batch: bool = False,
    simplify: bool = True,
) -> Path:
    """Export encoder model to ONNX format.

    Args:
        encoder: PyTorch encoder model (AppearanceEncoder).
        output_path: Path to save ONNX file.
        input_size: Input image resolution.
        opset_version: ONNX opset version.
        dynamic_batch: Whether to support dynamic batch size.
        simplify: Whether to simplify the ONNX graph.

    Returns:
        Path to exported ONNX file.
    """
    import torch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoder.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size)
    if next(encoder.parameters()).is_cuda:
        dummy_input = dummy_input.cuda()

    input_names = ["image"]
    output_names = ["latent"]

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"image": {0: "batch"}, "latent": {0: "batch"}}

    torch.onnx.export(
        encoder,
        dummy_input,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Simplify if requested
    if simplify:
        try:
            import onnxsim
            import onnx
            model = onnx.load(str(output_path))
            model_simplified, check = onnxsim.simplify(model)
            if check:
                onnx.save(model_simplified, str(output_path))
                print(f"ONNX model simplified successfully.")
        except ImportError:
            print("onnxsim not installed, skipping simplification.")

    print(f"Exported encoder to ONNX: {output_path}")
    print(f"  Input:  (1, 3, {input_size}, {input_size})")

    # Report size
    size_kb = output_path.stat().st_size / 1024
    print(f"  Size:   {size_kb:.1f} KB")

    return output_path


def export_quantized_onnx(
    onnx_path: str | Path,
    output_path: str | Path | None = None,
    quantization: str = "int8",
) -> Path:
    """Quantize an ONNX model for edge deployment.

    Args:
        onnx_path: Path to the ONNX model.
        output_path: Path for quantized model. Defaults to *_quant.onnx.
        quantization: Quantization type ("int8" or "uint8").

    Returns:
        Path to quantized ONNX file.
    """
    onnx_path = Path(onnx_path)
    if output_path is None:
        output_path = onnx_path.with_name(onnx_path.stem + f"_{quantization}.onnx")
    output_path = Path(output_path)

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        qtype = QuantType.QInt8 if quantization == "int8" else QuantType.QUInt8

        quantize_dynamic(
            str(onnx_path),
            str(output_path),
            weight_type=qtype,
        )

        orig_size = onnx_path.stat().st_size / 1024
        quant_size = output_path.stat().st_size / 1024
        print(f"Quantized {onnx_path.name}: {orig_size:.1f}KB → {quant_size:.1f}KB "
              f"({quant_size/orig_size*100:.1f}%)")

        return output_path

    except ImportError:
        print("onnxruntime.quantization not available. Install onnxruntime.")
        return onnx_path
