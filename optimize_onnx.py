import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

def optimize(model_name):
    onnx_file = f"{model_name}.onnx"
    if not os.path.exists(onnx_file):
        print(f"[-] ONNX file not found: {onnx_file}. Skipping optimization for {model_name}.")
        return

    # 1. Generate FP16 variant (WebGPU acceleration)
    try:
        print(f"[+] Converting {model_name} to FP16...")
        model_fp32 = onnx.load(onnx_file)
        model_fp16 = float16.convert_float_to_float16(model_fp32)
        onnx.save(model_fp16, f"{model_name}_fp16.onnx")
        print(f"[+] Created FP16: {model_name}_fp16.onnx")
    except Exception as e:
        print(f"[-] FP16 conversion failed for {model_name}: {e}")

    # 2. Generate INT8 variant (WASM/CPU fallback)
    try:
        print(f"[+] Quantizing {model_name} to INT8...")
        quantize_dynamic(
            model_input=onnx_file,
            model_output=f"{model_name}_int8.onnx",
            weight_type=QuantType.QUInt8
        )
        print(f"[+] Created INT8: {model_name}_int8.onnx")
    except Exception as e:
        print(f"[-] INT8 conversion failed for {model_name}: {e}")

if __name__ == "__main__":
    optimize("bird_mixit_4source")
    optimize("mel_band_roformer")
