# Model Setup & ONNX Conversion Guide (Bird-MixIT & Mel-Band Roformer)

This guide details how to select, download, convert, and install environmental audio source separation models for the **BirdNET Live Radar PWA (Pipeline B)**, focusing on **Bird-MixIT** and **Mel-Band Roformer (UVR)**.

---

## 1. Model Selection Analysis

For client-side execution in a web browser (PWA) via **ONNX Runtime Web**, we must balance **separation quality** with **performance overhead (RAM, storage, GPU cycles)**.

### Comparison Matrix

| Model | Size (FP32) | Latency (Mobile) | Target Sample Rate | Strengths | Weaknesses |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Bird‑MixIT (4‑source, bird‑specific)** | ~30–60 MB | **Low** (<350 ms) | 22,050 Hz | Official Google model trained on large bird soundscape datasets; robust to outdoor noise; designed to improve downstream bird classification. | Unsupervised (outputs unlabeled channels); needs custom tf2onnx conversion from TF checkpoints. |
| **Mel-Band Roformer (UVR)** | ~40–80 MB | **Medium** (<600 ms) | 44,100 Hz | State-of-the-art transformer-based separation; excellent separation boundaries for high-pitched bird calls. | Slightly higher RAM usage; requires config YAML for conversion. |
| **Demucs v4** | ~100–300 MB | **High** (>2000 ms) | 44,100 Hz | High-quality time-domain separation (music-oriented). | Too heavy for mobile browser CPUs/GPUs (prone to WebGPU OOM); not bird‑specific. |

### Recommendations for this PWA
* **Primary: Bird‑MixIT (bird‑specific 4‑source ONNX variant).** This is the best current unsupervised separator/denoiser for birds: it is trained on bird soundscapes and has been shown to improve bird classification when you run the classifier on both original and separated audio and take the max score per species.
* **Alternative: Mel-Band Roformer (UVR).** Excellent choice if you need high-fidelity extraction of high-pitched overlapping bird vocalizations. It operates in the frequency-domain using rotary transformer blocks.

---

## 2. Step-by-Step Setup Plan

### Step A: Download Pre-Trained Weights

#### Option 1: Google Research Bird‑MixIT (Bird‑Specific 4‑Source)
Download the official TensorFlow checkpoints provided by Google Research:
```bash
# Verify gsutil is installed, then download checkpoints
gsutil -m cp -r gs://gresearch/sound_separation/bird_mixit_model_checkpoints .

# The directory contains output_sources4 (4-source bird-specific model, recommended for this PWA)
# and output_sources8 (8-source model, optional).
```

#### Option 2: Mel-Band Roformer (UVR)
Download the PyTorch model checkpoint (`.ckpt` or `.pth`) and the corresponding model configuration `.yaml` file from Hugging Face (e.g. from the UVR resources repositories or official Kimberley Jensen/Viperx Roformer releases):
* **Checkpoint (`.ckpt`):** Contains the model weights.
* **Config (`.yaml`):** Contains the neural network architecture parameters (e.g., number of bands, transformer layers).

---

### Step B: Convert to ONNX Format

Ensure you have a Python environment. Since `tensorflow` is not yet available as a pre-compiled wheel for Python 3.14 on macOS, **use Python 3.12** to set up your virtual environment:

```bash
# Recreate venv with Python 3.12
rm -rf venv
/opt/homebrew/bin/python3.12 -m venv venv
venv/bin/pip install --no-cache-dir tensorflow tf2onnx onnx onnxruntime onnxconverter-common pyyaml
```

#### 1. Convert TensorFlow (Bird‑MixIT) to ONNX
Run `tf2onnx` on the `output_sources4` SavedModel folder:
```bash
python -m tf2onnx.convert \
    --saved-model path/to/bird_mixit_model_checkpoints/output_sources4 \
    --output bird_mixit_4source.onnx \
    --opset 15
```

#### 2. Convert PyTorch (Mel-Band Roformer) to ONNX
Create a Python script `convert_roformer.py`:
```python
import torch
import torch.onnx
import yaml
# Assuming the Roformer source code is in your python path:
# from melband_roformer import MelBandRoformer

# 1. Load config and instantiate model
# with open("model_config.yaml", "r") as f:
#     config = yaml.safe_load(f)
# model = MelBandRoformer(**config["model"])

# 2. Load checkpoint weights
# checkpoint = torch.load("mel_band_roformer.ckpt", map_location="cpu")
# model.load_state_dict(checkpoint)
# model.eval()

# Dummy representation for export demonstration:
class DummyMelBandRoformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 2, kernel_size=15, padding=7)
    def forward(self, x):
        # Input shape [1, samples] -> returns [1, 2, samples]
        x_expanded = x.unsqueeze(1)
        return self.conv(x_expanded)

model = DummyMelBandRoformer().eval()

# 3. Dummy input representing 9 seconds of audio at 44.1kHz (396,900 samples)
dummy_input = torch.randn(1, 396900)

# 4. Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "mel_band_roformer.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {1: "samples"},
        "output": {2: "samples"}
    },
    opset_version=15
)
print("Successfully exported Mel-Band Roformer to ONNX!")
```

---

### Step C: Create Optimized Precision Variants (FP16 & INT8)

For fast execution on mobile WebGPU/WASM, create half-precision (FP16) and quantized (INT8) variants of the exported ONNX models.

Create a python script `optimize_onnx.py`:
```python
import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_dynamic, QuantType

def optimize(model_name):
    # 1. Generate FP16 variant (WebGPU acceleration)
    try:
        model_fp32 = onnx.load(f"{model_name}.onnx")
        model_fp16 = float16.convert_float_to_float16(model_fp32)
        onnx.save(model_fp16, f"{model_name}_fp16.onnx")
        print(f"Created FP16: {model_name}_fp16.onnx")
    except Exception as e:
        print(f"FP16 conversion failed for {model_name}:", e)

    # 2. Generate INT8 variant (WASM/CPU fallback)
    try:
        quantize_dynamic(
            model_input=f"{model_name}.onnx",
            model_output=f"{model_name}_int8.onnx",
            weight_type=QuantType.QUInt8
        )
        print(f"Created INT8: {model_name}_int8.onnx")
    except Exception as e:
        print(f"INT8 conversion failed for {model_name}:", e)

optimize("bird_mixit_4source")
optimize("mel_band_roformer")
```
Run the optimization:
```bash
venv/bin/python optimize_onnx.py
```

---

### Step D: Move Model Files to the PWA

Place the generated model files under the `public/models` directory:

```bash
# Move Bird-MixIT 4-source variants
mv bird_mixit_4source.onnx public/models/bird_mixit_4source.onnx
mv bird_mixit_4source_fp16.onnx public/models/bird_mixit_4source_fp16.onnx
mv bird_mixit_4source_int8.onnx public/models/bird_mixit_4source_int8.onnx

# Move Mel-Band Roformer variants
mv mel_band_roformer.onnx public/models/mel_band_roformer.onnx
mv mel_band_roformer_fp16.onnx public/models/mel_band_roformer_fp16.onnx
mv mel_band_roformer_int8.onnx public/models/mel_band_roformer_int8.onnx
```

---

### Step E: Update the Model Catalog Schema

Ensure that `public/models/models.json` is updated to expose the model paths and their sample rates:

```json
[
  {
    "id": "bird_mixit_4source",
    "name": "Bird-MixIT 4-Source (ONNX)",
    "path": "models/bird_mixit_4source.onnx",
    "outputSampleRate": 22050,
    "precisions": {
      "fp32": "models/bird_mixit_4source.onnx",
      "fp16": "models/bird_mixit_4source_fp16.onnx",
      "int8": "models/bird_mixit_4source_int8.onnx"
    }
  },
  {
    "id": "mel_band_roformer",
    "name": "Mel-Band Roformer (UVR)",
    "path": "models/mel_band_roformer.onnx",
    "outputSampleRate": 44100,
    "precisions": {
      "fp32": "models/mel_band_roformer.onnx",
      "fp16": "models/mel_band_roformer_fp16.onnx",
      "int8": "models/mel_band_roformer_int8.onnx"
    }
  }
]
```