# Part 1: Project Brief (The Blueprint)

This project is an **advanced, offline-first bird identification radar** hosted as a Progressive Web App (PWA) via GitHub Pages. By executing heavy Machine Learning models locally via WebAssembly and WebGPU, it continuously analyzes microphone input to detect and isolate bird calls.

The defining innovation is its **modular dual-pipeline audio architecture**. It pairs a real-time identification layer with a parallel, computationally heavy source-separation (segmentation) layer. Designed specifically for modern flagship devices, the goal is to create a privacy-first, highly autonomous bioacoustics tool that entirely eliminates environmental noise, untangles overlapping calls, and operates cross-platform without a backend server.

## Core Architecture

* **Delivery:** GitHub Pages hosted PWA with IndexedDB/Cache API for full offline capabilities.
* **Pipeline A (Live Radar):** A highly responsive, 3-second rolling-window inference engine using the **BirdNET V2.4 FP32 (TFLite)** model for immediate user feedback (marked as "Pending").
* **Pipeline B (High-Fidelity Segmentation):** A delayed, modular Web Worker process that takes larger audio buffers (6–9 seconds). It uses a heavy source-separation model (via ONNX WebGPU) to split overlapping birds and noise into distinct audio channels, re-analyzing each channel with the BirdNET FP32 model.
* **The Consensus Engine:** An internal logic layer that compares Pipeline B's isolated results against Pipeline A's pending drafts to verify hits, drop false positives, and output clean, isolated audio tracks for the user.

## Distinct Advantages

* **Zero Server Dependency:** Unlike BirdNET-Pi or BirdNET-Go, it requires no hardware setup or server maintenance.
* **Passive & Continuous:** Unlike the official BirdNET App, it acts as a hands-off radar rather than a manual recording tool.
* **Isolated Playback & Extreme Accuracy:** Unlike the official real-time PWA, the segmentation layer extracts distinct sound layers. This prevents false positives caused by urban noise and allows users to isolate and listen to individual birds from a busy soundscape.

---

# Part 2: The Technical Implementation Plan

To keep the segmentation pipeline modular while supporting heavier models, the architecture must heavily abstract the audio routing. We will fork `birdnet-team/real-time-pwa` to handle the device hardware APIs, but gut and rebuild its processing logic.

### Phase 1: Foundation & Hardware Routing

*Forking the official PWA and establishing the dual-routing system.*

1. **Repository Setup:** Fork the `real-time-pwa`. Update the PWA manifest, service workers, and Tailwind UI shell to reflect the new project identity.
2. **Central Audio Buffer (The Main Thread):** Modify the existing `AudioWorkletNode`. Instead of feeding a single worker, it must now stream raw PCM data into a central, rolling `RingBuffer` in the main thread.
3. **Dual-Dispatch System:** Write a dispatcher script that reads from the `RingBuffer`:
* **Dispatch A:** Sends the last 3 seconds of audio to `live-worker.js` every 1 second.
* **Dispatch B:** Sends the last 9 seconds of audio to `segmentation-worker.js` every 4.5 seconds (allowing for a 50% overlap to ensure no calls are cut off).



### Phase 2: Pipeline A (The Live Tracker)

*Establishing the real-time "Draft" interface.*

1. **Worker Initialization:** Implement `@tensorflow/tfjs-tflite` in `live-worker.js` and load the `BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite` model.
2. **Meta-Model Integration:** Retain the geolocation and date-based meta-model from the original fork to adjust base probabilities.
3. **Pending UI State:** When `live-worker.js` detects a bird, render it in the UI with a distinct visual state (e.g., a pulsing border or "Analyzing..." badge). It is logged into a temporary state object, awaiting Pipeline B's confirmation.

### Phase 3: Pipeline B (High-Fidelity Modular Segmentation)

*This is the core innovation. We will use ONNX Runtime Web with the WebGPU execution provider to handle a heavier separation model.*

1. **The Modularity Interface:** Create an abstract class `AudioSeparator` in `segmentation-worker.js`. This ensures the exact model can be swapped later without breaking the app. It must strictly accept a `Float32Array` (the mixed audio) and return `Array<Float32Array>` (the isolated tracks).
2. **Model Selection (Targeting High-End):** Since we are targeting modern devices, convert a general-purpose, high-quality model like **Demucs v4** (specifically a version fine-tuned on environmental/DCASE data) or a robust PyTorch implementation of **BioCPPNet** to `.onnx`.
3. **WebGPU Execution:** Instantiate `onnxruntime-web/webgpu`. This offloads the massive matrix multiplication of the separation model to the iPhone/Pixel's GPU, preventing the browser tab from crashing.
4. **Re-Analysis:** Once the audio is split into tracks (e.g., Track 1, Track 2), pass *each* track through a secondary instance of the BirdNET FP32 model located inside this same worker.

### Phase 4: The Consensus Engine & Playback UI

*Reconciling the two pipelines.*

1. **Data Structure:** `segmentation-worker.js` returns an object containing: `[Timestamp, Track_Audio_Buffer, Identified_Species, Confidence_Score]`.
2. **The Validator:** The main thread compares this object against Pipeline A's pending list.
* *Match:* If Pipeline A flagged a "Robin" and Pipeline B confirms it with high confidence on an isolated track, the UI element upgrades from "Pending" to "Verified."
* *Correction/False Positive:* If Pipeline B identifies a sound as noise (or just fails to find a bird), the pending UI element is silently unmounted.


3. **Isolated Playback:** Attach the `Track_Audio_Buffer` to the verified UI element. Use the WebAudio API to allow the user to click a "Play Isolated Call" button, letting them hear the bird without the background noise.

### Phase 5: Caching & Deployment

1. **Model Storage:** Because the ONNX separation models and the FP32 BirdNET model are large, they cannot be downloaded every time. Implement the `Cache API` to store the `.tflite` and `.onnx` files permanently on the device after the first load.
2. **Web Worker Fallbacks:** Implement a feature check on load. If the device's browser does not support WebGPU (e.g., an older iOS version), fallback to WebAssembly (WASM) execution for ONNX, with a warning to the user that segmentation will run slower.
3. **Deployment:** Deploy the static bundle to GitHub Pages.

---

additional notes:
To ensure Pipeline B remains completely modular, we should define the exact Input/Output (I/O) data contract between the Main Thread and the Segmentation Worker.