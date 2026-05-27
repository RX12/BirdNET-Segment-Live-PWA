# BirdNET PWA Pipeline & Consensus Architecture

This document serves as the primary source of truth for the classification pipelines, environmental source separation engine, and the consensus fusion logic within the BirdNET Live Radar PWA. It details both the **Live Identifier** stream and the **Audio Upload** scanning workflows.

---

## 1. High-Level System Architecture

The application runs entirely client-side using two parallel pipelines:
1. **Pipeline A (Draft Detection / Live Classifier):** Runs continuous classification using the official BirdNET global model on raw audio to establish draft detections.
2. **Pipeline B (Verification / Source Separator):** Isolates multiple concurrent audio sources using a 4-source MixIT model and verifies draft detections by running classification on each isolated track.

```
                  ┌────────────────────────────────────────┐
                  │               Audio Source             │
                  │   (Live Microphone or Uploaded File)   │
                  └───────────────────┬────────────────────┘
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
         ┌──────────────────┐                  ┌──────────────────┐
         │    Pipeline A    │                  │    Pipeline B    │
         │ (Live Classifier)│                  │(Source Separator)│
         └─────────┬────────┘                  └─────────┬────────┘
                   │                                     │
           [Draft Detections]                     [Isolated Tracks]
         (e.g., Nuthatch 37.4%)                          │
                   │                                     ▼
                   │                           ┌──────────────────┐
                   │                           │    Classifier    │
                   │                           │ (Isolated Tracks)│
                   │                           └─────────┬────────┘
                   │                                     │
                   │                           [Channel Predictions]
                   │                          (e.g., Wren & Nuthatch)
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                       ┌────────────────────────────┐
                       │      Consensus Engine      │
                       │ (Verify / Reject Matches)  │
                       └──────────────┬─────────────┘
                                      ▼
                        ┌──────────────────────────┐
                        │      Final Results       │
                        └──────────────────────────┘
```

---

## 2. Core Components & Model Specifications

### A. Pipeline A Classifier
* **Script / Worker:** [live-worker.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/live-worker.js)
* **Model File:** `BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite`
* **Runtime:** TensorFlow.js TFLite WebAssembly runner (`tfjs-tflite`) running on CPU backend (`tf.setBackend('cpu')`).
* **Input Specifications:** 3.0-second chunk of raw Float32 mono PCM audio at 48kHz (`144,000` samples) shaped as `[1, 144000]`.
* **Output Specifications:** 1D array of class confidence probabilities (size `3332`). *Note: The raw TFLite model output contains unnormalized log-odds (logits) ranging from negative to positive values. The worker explicitly transforms these logits using the standard Sigmoid activation function ($p = \frac{1}{1 + e^{-x}}$) to generate proper probability scores in the range `[0.0, 1.0]`.*

### B. Pipeline B Separator
* **Script / Worker:** [segmentation-worker.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/segmentation-worker.js)
* **Model File:** `bird_mixit_4source.onnx` (custom patched Conv-TasNet / MixIT 4-source separator model).
* **Runtime:** ONNX Runtime WebAssembly/WebGPU (`ort.min.js`) running on WebGPU executor with CPU WebAssembly SIMD fallback.
* **Input Specifications:** `9.0` seconds (default `pipelineBWindow`) resampled to 22.050kHz (`198,450` samples) shaped as `[1, 1, 198450]`.
* **Output Specifications:** Single output tensor containing 4 separated audio channels of shape `[1, 4, 198450]`.

### C. Pipeline B Classifier
* **Script / Worker:** [segmentation-worker.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/segmentation-worker.js)
* **Model File:** Same TFLite model (`BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite`).
* **Runtime:** `tfjs-tflite` CPU runtime inside the segmentation worker.
* **Inference Method:** Slices each 9.0s separated channel back into 3.0-second frames with 1.5-second overlap (5 frames total), runs inference frame by frame, applies standard Sigmoid transformation to the raw logits to get true probabilities, and pools them using **Max Pooling** to preserve transient calls.

---

## 3. Detailed Data Flow Traces

### Workflow A: Live Identifier (Microphone Stream)
1. **Audio Capture:** [app.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/app.js) captures audio from the microphone and routes it via [audio-router.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/audio-router.js).
2. **Pipeline A Execution:**
   * Every 1.0 second, [audio-router.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/audio-router.js#L240-L245) grabs the last 3.0s of raw 48kHz audio and sends it to [live-worker.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/live-worker.js) using the `predict` action.
   * [live-worker.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/live-worker.js#L184-L239) runs the TFLite model and posts the results back to the main thread.
   * If a species has confidence $\ge 0.15$ (default `detectionThreshold`), it is flagged as a draft and added to `pendingDetections` in [app.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/app.js#L544).
3. **Pipeline B Execution:**
   * When `pendingDetections` has active items, [audio-router.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/audio-router.js) feeds a 9.0-second overlapping audio window (decimated and resampled to 22.050kHz) to [segmentation-worker.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/segmentation-worker.js) using `PROCESS_SEGMENT`.
   * The segmentation worker runs MixIT to separate the 4 audio sources, resamples them back to 48kHz, peak normalizes them to `[-0.8, 0.8]`, runs the TFLite classifier on 3.0s slices, and max-pools the predictions.
   * The worker sends a `SEGMENT_RESULT` containing the predictions back to the main thread.
4. **Consensus Fusion:** [app.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/app.js#L531-L570) receives the channel results. If a species exists in `pendingDetections` AND has confidence $\ge 0.15$ on at least one separated channel, it is verified and displayed. If a draft remains unverified when its time window exits the pipeline, it is discarded as a false positive.

### Workflow B: Audio Upload (Offline Scan)
1. **Audio Decoding & Resampling:** The user uploads a file. [upload.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/upload.js#L278-L304) decodes it using Web Audio API and resamples the entire track to 48kHz mono, creating `audioBuffer48k`.
2. **Scan Interval Loop:** An interval timer increments `scanTimeSec` by `0.5s` (simulated speed).
3. **Pipeline A Execution:**
   * Every 1.0 second of simulated audio, a 3.0s window is sliced and sent to [live-worker.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/live-worker.js).
   * Draft species with confidence $\ge 0.15$ are registered in `pendingDetections` and mapped to a simulated timestamp.
4. **Pipeline B Execution:**
   * Every 4.5 seconds (defined by `pipelineBStride`), a 9.0s window is sliced and sent to [segmentation-worker.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/segmentation-worker.js).
   * The worker separates, resamples, peak-normalizes, classifies, and pools the predictions for each channel, posting `SEGMENT_RESULT` back to the main thread.
5. **Consensus Fusion:** [upload.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/upload.js#L699-L748) reconciles predictions. It verifies drafts if they match isolated channel predictions $\ge 0.15$, and discards them if they are not verified in the temporal segment window.

---

## 4. Consensus and Fusion Logic Detail

The PWA integrates a gating consensus strategy to guarantee high-fidelity detections while eliminating false positives from overlapping calls:

1. **Draft Thresholding:** In the raw signal, many overlapping calls blend together, yielding weak confidences. Pipeline A flags any species crossing $15\%$ confidence as a "draft".
2. **Separation Gating:** Pipeline B extracts distinct vocal tracks. By isolating the calling bird, background noise and overlapping frequencies are attenuated.
3. **Peak Normalization:** Quiet separated channels are boosted up to `20x` (capped gain) in both workers ([live-worker.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/live-worker.js) and [segmentation-worker.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/segmentation-worker.js)) to ensure the classifier has optimal input levels.
4. **Max-Pooling Inference:** Inference runs on overlapping 3-second slices of each isolated channel. Max-pooling is applied across these frames. This ensures that transient bird calls (which might only appear in a single frame of the 9.0s window) are captured at their peak confidence, rather than being averaged out.
5. **Fusion Verification:** If an isolated track classification confirms the draft species with $\ge 15\%$ confidence, it is marked as `VERIFIED`.
6. **Timeline Reconciliation:** If a draft species is detected but cannot be verified on any separated channel within that time frame, the consensus engine assumes the raw model was confused by overlapping frequencies (a false positive) and rejects it.
7. **Early Exit Fallback:** If the raw mix confidence (Pipeline A) is exceptionally high (e.g. $\ge 90\%$), the pipeline can bypass source separation to save battery and processing time. In this case, the raw 3-second audio slice is mapped as the verification buffer for that species, enabling play and download capabilities for that detection window.

---

## 5. Root Causes of Diagnostic Discrepancies

### Issue A: Why Pipeline A Only on separated WAV files fails to detect Wren (while BirdNET Demo succeeds)
* **Lack of Peak Normalization in Pipeline A:** The isolated channel WAV files generated by MixIT are structurally very quiet. For instance, channel 0 had a peak amplitude of only `0.0951` (under 10% volume). 
* When running "Pipeline A Only" inside the PWA, [live-worker.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/live-worker.js) performs **no gain adjustments** or peak normalization on the audio before classification. The weak signal is fed directly to the TFLite model, leading to suppressed confidences below the `0.15` threshold.
* The official **BirdNET Web Demo** performs automatic gain control (AGC) or peak normalization to scale the signal up before feeding it to the model. Similarly, the PWA's **Pipeline B worker** peak normalizes each separated track before classifying. This is why the Wren is easily detected in both of those environments but missed by Pipeline A in the PWA.

### Issue B: Why the entire Pipeline A+B consensus run yields zero detections (missing even Kleiber)
* **The Float32Array.map Serialization Bug:** There is a critical JavaScript typed array bug in [segmentation-worker.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/segmentation-worker.js#L570-L578):
  ```javascript
  const formattedPredictions = pooledPredictions
      .map((confidence, idx) => ({
          speciesCode: birds[idx] ? birds[idx].scientificName : `class_${idx}`,
          commonName: birds[idx] ? (birds[idx].commonNameI18n || birds[idx].commonName) : `Class ${idx}`,
          scientificName: birds[idx] ? birds[idx].scientificName : `Class ${idx}`,
          confidence: confidence
      }))
  ```
  In JavaScript, calling `.map()` on a typed array (like `pooledPredictions` which is a `Float32Array`) returns a new typed array of the *same type* (`Float32Array`).
  Because the map callback returns objects, JavaScript attempts to cast these objects to numbers, resolving all of them to `NaN`.
  * This resulted in `formattedPredictions` being a `Float32Array` filled with `NaN`s.
  * In the worker log, formatting the slice of predictions threw no errors but outputted nothing because `p.commonName` and `p.confidence` were undefined.
  * The serialization sent a `Float32Array` of `NaN`s to the main thread.
  * The main thread consensus logic looped through `res.predictions` and checked `pred.scientificName` and `pred.confidence`. Since both were undefined/falsy, **no species was ever verified by Pipeline B**.
  * Consequently, the timeline reconciliation block rejected all drafts (including Kleiber/Eurasian Nuthatch which originally had 37.4% confidence in Pipeline A) as false positives, leaving the user with a blank results page.

---

## 6. Browser-Side Playback & WAV Export Integration

### A. Namespace Decoupling
To prevent namespace pollution and page clashing, the global handlers on the upload and live pages are decoupled:
* **Audio Upload Page ([upload.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/upload.js)):** Registers page-specific handlers `window.playUploadIsolatedAudio`, `window.stopUploadIsolatedAudio`, and `window.downloadUploadIsolatedAudio`.
* **Live Dashboard Page ([app.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/app.js)):** Only registers the standard global hooks `window.playIsolatedAudio`, `window.stopIsolatedAudio`, and `window.downloadIsolatedAudio` if `#liveSpectrogram` is present in the DOM.

### B. Best-Channel Buffer Selection
When the user clicks "Play Isolated" or "Download" for a species result:
1. The app invokes `getBestIsolatedBufferForSpecies(scientificName)`.
2. It queries `separatedChannelPredictions` (accumulated during the scan) to locate which of the 4 separated channels scored the highest confidence score for that species across the entire audio timeline.
3. It retrieves the Float32Array buffers for that channel from `separatedChannelBuffers` and concatenates them into a single continuous track.
4. If the species was verified via an early-exit fallback (no separation buffer available), it falls back to the raw mix audio slice.

### C. Peak Normalization on Output
Before playback or export, the concatenated Float32Array buffer is peak normalized:
* The peak absolute value is calculated.
* An audio gain factor is computed to scale the peak value to `0.8` (capped at a maximum boost of `20.0x` to prevent amplifying pure static noise).
* The normalized array is generated by multiplying the samples by the gain factor.

### D. Audio Playback
Playback is managed via a native browser `AudioContext`. To ensure compatibility with Safari and iOS devices:
* The standard `AudioContext` or `webkitAudioContext` is created dynamically (wrapped in a try-catch fallback to ignore `sampleRate` constructor limitations).
* `playbackAudioContext.resume()` is explicitly awaited before triggering the source node.
* A native `AudioBuffer` is allocated using `playbackAudioContext.sampleRate` and populated with the normalized samples.

### E. WAV File Encoding & Download
Instead of relying on heavy third-party libraries, the application uses a lightweight inline WAV encoder (`bufferToWav`):
* Allocates a 44-byte `ArrayBuffer` for the standard RIFF/WAVE header.
* Fills in metadata: Mono channel count, sample rate, bit depth (16-bit PCM), and block alignment.
* Maps Float32 samples `[-1.0, 1.0]` into 16-bit signed integers `[-32768, 32767]`.
* Packs the header and PCM buffer into a single `Blob` of type `audio/wav`.
* Programmatically triggers a temporary anchor element click to trigger a local browser download (`[Species_Name]_isolated.wav`).

---

## 7. Robustness, Caching, and UI Utilities

### A. WebGPU JSEP Executor Shape Compatibility
ONNX Runtime's WebGPU execution provider compiles dynamic GPU shader kernels when it encounters varying input shapes. Under short audio segments or tail end segments, the shape of the audio input array changes, causing driver compilation freezes or `Failed to run JSEP kernel` crashes.
* **Symmetric Padding**: Inside [segmentation-worker.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/segmentation-worker.js), the resampled input array is symmetrically padded (with zeros) or truncated to a constant target size matching the selected `pipelineBWindow` (e.g. `198,450` samples for `9.0s`).
* After source separation inference completes, the 4 output channels are sliced back to their original duration (discarding the padding) before being resampled back to 48kHz, keeping WebGPU execution shape constant and crash-free.

### B. Sequential Processing Queue
To prevent concurrent execution clashes in WebGPU (which causes WASM heap corruption and driver crashes under fast scan speeds), the segmentation worker queues incoming segments inside a FIFO array and processes them sequentially, locking execution until the current session run is fully completed.

### C. Service Worker Cache Reload
Since core assets are cached offline, PWA updates can leave pages running outdated client-side scripts.
* **Update Gating**: The Service Worker `APP_VERSION` is incremented on changes (forcing a new cache directory installation).
* **Automatic Reload**: [base.njk](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/src/_includes/layouts/base.njk) registers a `controllerchange` event listener on `navigator.serviceWorker`. The instant the new service worker activates and claims clients, the browser automatically triggers `window.location.reload()`, ensuring the user is immediately running the latest scripts.

### D. Audio Channels Debugger UI
A dedicated debug panel (`#separationDebuggerCard`) in the upload page displays the 4 raw audio tracks produced by the separator.
* Stitches and peak-normalizes each track.
* Generates native HTML `<audio>` elements for direct listening.
* Embeds explicit WAV downloads for each track.
* Details the raw class predictions (and confidences) for each channel to verify model execution transparency.

---

## 8. Live Identification Pipeline Orchestration & Timeline Analysis

This section analyzes the detailed orchestration, temporal windows, overlap rates, backpressure mechanisms, and latency profiles of the **Live Identification** mode.

### A. Temporal Timeline & Overlap Matrix
Live Mode operates on two decoupled periodic dispatch loops that extract sliding windows from a shared circular ring buffer:

| Parameter | Pipeline A (Live Classifier) | Pipeline B (Consensus Separator) |
| :--- | :--- | :--- |
| **Window Duration** | 3.0 seconds (`144,000` samples) | 9.0 seconds (`432,000` samples) |
| **Stride / Dispatch Interval** | 1.0 second | 4.5 seconds (default `pipelineBStride`) |
| **Slicing Overlap Rate** | **66.6% (2.0s)** between consecutive runs | **50.0% (4.5s)** between consecutive runs |
| **Buffer Source** | Main Thread RingBuffer (15s capacity) | Main Thread RingBuffer (15s capacity) |
| **Internal Framing** | Exactly 1 frame per run | 5 overlapping frames of 3.0s (1.5s overlap / hop) |

### B. Execution Sequence & Timeline Trace

```
Time (s)  0.0     1.0     2.0     3.0     4.0     5.0     6.0     7.0     8.0     9.0     10.0
─────────────────────────────────────────────────────────────────────────────────────────────
Mic Input ══════════════════════════════════════════════════════════════════════════════════> (Continuous)
RingBuffer [────────────────── 15.0s Rolling Window ──────────────────]

Pipeline A:
- Run A1:                         [  3.0s window  ] ──> TFLite (Inference: ~150ms)
- Run A2:                                 [  3.0s window  ] ──> TFLite
- Run A3:                                         [  3.0s window  ] ──> TFLite

Pipeline B:
- Run B1:                         [─────────────── 9.0s window ───────────────] ──> MixIT (Inference: ~1.5s - ~4.0s)
                                                                                  └──> TFLite on 4 channels (5 frames each)
```

1. **Microphone Capture**: The `AudioContext` streams raw audio. [audio-router.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/audio-router.js) writes PCM blocks into a 15-second `RingBuffer` continuously.
2. **Draft Generation**: Every **1.0 second**, Pipeline A slices the last 3.0s of the RingBuffer and sends it to [live-worker.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/live-worker.js). If any prediction confidence $\ge 0.15$ (default `detectionThreshold`), it is added to the `pendingDetections` map with `timestamp = Date.now()`.
3. **Consensus Trigger**: Every **4.5 seconds**, Pipeline B checks if any draft exists in `pendingDetections`. If yes:
   * It extracts the last 9.0s of audio and dispatches it to [segmentation-worker.js](file:///Users/rubendietrich/Documents/Div/ProjectsCode/BirdNET-Live-Radar-PWA/public/js/segmentation-worker.js) via `PROCESS_SEGMENT`.
   * It marks the router as busy (`segmentBusy = true`) to prevent launching concurrent separation tasks (backpressure gate).
4. **Verification & Timeline Reconciliation**:
   * The segmentation worker runs MixIT (4 channels) and classifies five 3.0s overlapping frames per channel (hop size = 1.5s).
   * It aggregates predictions using **Max Pooling** and sends them back to the main thread.
   * `app.js` matches the verified channel species with the drafts in `pendingDetections`.
   * **Reconciliation Window**: It validates or discards any drafts whose timestamp falls in the `[timestamp - 9.0s, timestamp]` window. If confirmed, it is marked as `VERIFIED` and rendered. If the window passes without confirmation, it is discarded as a false positive.
   * Marks `segmentBusy = false`, opening the gate for the next stride.

### C. Gating, Backpressure & Latency Profiles

* **Gate 1: Acoustic Activity Detection (AAD)**: Before sending a 9.0s window to the separator, it calculates the RMS amplitude and zero-crossing Spectral Centroid. If it falls below the thresholds (`RMS < 0.01` or `Centroid < 1000Hz`), Pipeline B is bypassed to conserve device battery.
* **Gate 2: Early Exit**: If Pipeline A finds a species with high confidence ($\ge 90\%$), it bypasses separation. The raw mix segment is used as the playback buffer for that detection.
* **Gate 3: Mobile Backpressure**: If a low-end mobile device takes $> 4.5$s to run separation, the next stride is skipped. The ring buffer continues to grow, and the next executed segment (e.g. at 9.0s) will read the last 9.0s of audio, ensuring **zero audio gaps** are introduced and all draft detections are still eventually reconciled.
* **Consensus Latency**:
  * **Best Case (WebGPU)**: `~2.8` seconds delay (instantaneous dispatch + MixIT run).
  * **Worst Case (WASM CPU Fallback)**: `~9.8` seconds delay (delayed stride + longer CPU inference).
