#!/usr/bin/env python3
"""
Requantizes the 'output.weight' tensor from f16 to q8_0 in a GGUF file.

Usage:
    python requantize_output.py <input_gguf> [output_gguf]

If no output path is given, writes to <input_gguf>.q8_0.gguf
"""

from __future__ import annotations

import logging
import sys

import numpy as np
from pathlib import Path

# Allow running this script from the examples/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf.gguf_reader import GGUFReader
from gguf.gguf_writer import GGUFWriter
from gguf.constants import GGMLQuantizationType
from gguf.quants import quantize, dequantize

logger = logging.getLogger("requantize")
logging.basicConfig(level=logging.INFO, format="%(message)s")

TARGET_TENSOR = "output.weight"
TARGET_QTYPE = GGMLQuantizationType.Q8_0
SOURCE_QTYPE = GGMLQuantizationType.BF16

def requantize(gguf_path: str, output_path: str | None = None):
    reader = GGUFReader(gguf_path)

    # Find the output.weight tensor
    tensor = None
    for t in reader.tensors:
        if t.name == TARGET_TENSOR:
            tensor = t
            break

    if tensor is None:
        logger.error("Tensor %r not found in %s", TARGET_TENSOR, gguf_path)
        sys.exit(1)

    logger.info("Found tensor: %s", TARGET_TENSOR)
    logger.info("  Shape:   %s", "x".join(map(str, tensor.shape)))
    logger.info("  Type:    %s", tensor.tensor_type.name)
    logger.info("  Elements: %d", tensor.n_elements)
    logger.info("  Bytes:    %d", tensor.n_bytes)

    if tensor.tensor_type != SOURCE_QTYPE:
        logger.warning(
            "Tensor is already %s (not %s). "
            "Requantization may produce unexpected results.",
            tensor.tensor_type.name,
            SOURCE_QTYPE.name,
        )

    # Quantize the tensor data in-place
    # GGUFReader does not decode BF16 to float32 (falls into the else branch,
    # returning a uint8 array with byte-shaped dims).  Decode BF16 to float32
    # so that `quantize()` receives actual float values.
    if tensor.tensor_type == SOURCE_QTYPE:
        original_data = dequantize(tensor.data, SOURCE_QTYPE)
    else:
        original_data = tensor.data.copy()
    quantized_data = quantize(original_data, TARGET_QTYPE)

    logger.info("  Quantized shape: %s", "x".join(map(str, tensor.shape)))
    logger.info("  quantized_data shape: %s", "x".join(map(str, quantized_data.shape)))

    # GGUFWriter stores dimensions in reverse order (last dim first), so the
    # shape written to file is reversed compared to numpy.  Reshape here so
    # the output GGUF tensor has the desired reversed shape.
    # reversed_shape = quantized_data.shape[::-1]
    # quantized_data = quantized_data.reshape(reversed_shape)
    # logger.info("  reshaped quantized_data: %s", "x".join(map(str, quantized_data.shape)))

    # Determine output path
    if output_path is None:
        output_path = str(Path(gguf_path).with_suffix(gguf_path.suffix + ".q8_0" + gguf_path.suffix))

    logger.info("Writing to: %s", output_path)

    # Copy architecture from source
    arch_field = reader.get_field("general.architecture")
    if arch_field is None:
        logger.error("Cannot determine architecture from source file")
        sys.exit(1)
    arch = arch_field.contents()

    # Write the new GGUF file
    writer = GGUFWriter(output_path, arch)

    # Copy key-value metadata (deduplicate: skip keys already added by writer or derived from header)
    # GGUFReader adds GGUF.version, GGUF.tensor_count, GGUF.kv_count from the header,
    # so if the source KV section also has them, they'd be duplicates.
    skipped_keys = {"general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
    seen_keys = set(skipped_keys)
    for key, field in reader.fields.items():
        if key in seen_keys:
            continue  # skip duplicates
        seen_keys.add(key)
        value = field.contents()
        types = field.types
        if not types:
            continue
        vtype = types[0]
        sub_type = types[1] if len(types) > 1 else None
        writer.add_key_value(key, value, vtype, sub_type)

    # Write all tensors
    for t in reader.tensors:
        if t.name == TARGET_TENSOR:
            writer.add_tensor(t.name, quantized_data, raw_dtype=TARGET_QTYPE)
        else:
            writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)

    # write_header_to_file() and write_kv_data_to_file() must be called first
    # to advance the state machine past NO_FILE before write_tensors_to_file()
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    logger.info("Done. Output: %s", output_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: requantize_output.py <input_gguf> [output_gguf]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    requantize(input_path, output_path)
