/**
 * BirdNET Live - Standalone Audio File Analyzer
 * 
 * Handles file drop/upload, OfflineAudioContext decoding/resampling,
 * paced pipeline simulation (Pipeline A & B), consensus engine verification,
 * waveform canvas drawing, playhead sweeps, and live debug console logging.
 */

(function () {
  /* ==========================================================================
     1. GLOBAL CONFIGURATION & STATE
     ========================================================================== */

  const SAMPLE_RATE = 48000;
  const WINDOW_SAMPLES = 144000; // 3 seconds at 48kHz
  const SEGMENT_SAMPLES = 432000; // 9 seconds at 48kHz
  const prefix = window.PATH_PREFIX || "/";

  // State Variables
  let audioBuffer48k = null;     // Float32Array PCM audio buffer
  let audioContext = null;       // Playback context for preview
  let previewSource = null;      // AudioSourceNode for playing the uploaded file
  let previewStartTime = 0;      // Start timestamp of preview
  let previewOffset = 0;         // Current offset of preview playback
  let isPlayingPreview = false;

  let liveWorker = null;
  let segmentationWorker = null;
  let liveWorkerReady = false;
  let segmentationWorkerReady = false;
  
  let isScanning = false;
  let isPipelineAOnly = false;
  let scanIntervalId = null;
  let scanTimeSec = 0;           // Simulated time in seconds
  let scanDurationSec = 0;
  let simulatedEpochStart = 1000000000000; // Base epoch for simulation
  let activePipelineARequests = 0;
  let activePipelineBRequests = 0;
  let isTimelineFeedFinished = false;

  // Consensus state maps
  let verifiedDetections = new Map(); // scientificName -> { confidence, audioBuffer, timestamp }
  let pendingDetections = new Map();  // scientificName -> detectedAt (ms)
  let latestDetections = [];          // Displayed array of detections

  // Isolated Playback state
  let playbackAudioContext = null;
  let activeAudioSource = null;
  let currentlyPlayingSpecies = null;

  // Separation Debugger state
  let separatedChannelBuffers = new Map(); // channelId -> Array of Float32Array
  let separatedChannelPredictions = new Map(); // channelId -> Map of scientificName -> { commonName, confidence }
  let createdObjectURLs = [];
  
  const store = {
    get: (k, def) => localStorage.getItem(k) ?? def,
    getFloat: (k, def) => { const v = localStorage.getItem(k); return v === null ? def : parseFloat(v); },
    getBool: (k, def) => { const v = localStorage.getItem(k); return v === null ? def : v === "true"; }
  };

  let detectionThreshold = store.getFloat("bn_threshold", 0.15);
  if (detectionThreshold > 1.0) detectionThreshold = 0.15;
  
  // Separator Model State
  let separatorModelCatalog = [];
  let selectedSeparatorModel = store.get("bn_separator_model", "bird_mixit_4source");
  let customModelBytes = null;
  let customModelName = "";
  let maxLiveConfidenceInInterval = 0;

  // New Pipeline B configs:
  let aadEnabled = store.getBool("bn_aad_enabled", true);
  let aadRmsThreshold = store.getFloat("bn_aad_rms_threshold", 0.010);
  let aadCentroidMin = store.getFloat("bn_aad_centroid_min", 1000);
  let earlyExitEnabled = store.getBool("bn_early_exit_enabled", true);
  let earlyExitConfidence = store.getFloat("bn_early_exit_confidence", 0.90);
  let pipelineBWindow = store.getFloat("bn_pipeline_b_window", 9.0);
  let pipelineBStride = store.getFloat("bn_pipeline_b_stride", 4.5);
  let separatorPrecision = store.get("bn_separator_precision", "fp32");
  let webgpuEnabled = store.getBool("bn_webgpu_enabled", true);
  
  // Elements
  const dropZone = document.getElementById("dropZone");
  const audioUpload = document.getElementById("audioUpload");
  const uploadLabel = document.getElementById("uploadLabel");
  const analysisContainer = document.getElementById("analysisContainer");
  const waveformCanvas = document.getElementById("waveformCanvas");
  const scanPlayhead = document.getElementById("scanPlayhead");
  const playbackBtn = document.getElementById("playbackBtn");
  const scanBtn = document.getElementById("scanBtn");
  const scanBtnPipelineA = document.getElementById("scanBtnPipelineA");
  const scanStatusText = document.getElementById("scanStatusText");
  const scanProgressBar = document.getElementById("scanProgressBar");
  const scanProgressText = document.getElementById("scanProgressText");
  const uploadDetectionsList = document.getElementById("uploadDetectionsList");
  const debugLogConsole = document.getElementById("debugLogConsole");
  const separationDebuggerCard = document.getElementById("separationDebuggerCard");
  const separatedChannelsContainer = document.getElementById("separatedChannelsContainer");

  /* ==========================================================================
     2. EVENT BINDINGS
     ========================================================================== */

  // Drag & Drop
  if (dropZone) {
    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.style.borderColor = "#3b82f6";
      dropZone.style.backgroundColor = "rgba(59, 130, 246, 0.05)";
    });

    dropZone.addEventListener("dragleave", (e) => {
      e.preventDefault();
      dropZone.style.borderColor = "#cbd5e1";
      dropZone.style.backgroundColor = "transparent";
    });

    dropZone.addEventListener("drop", async (e) => {
      e.preventDefault();
      dropZone.style.borderColor = "#cbd5e1";
      dropZone.style.backgroundColor = "transparent";

      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        await handleFileSelection(files[0]);
      }
    });
  }

  if (audioUpload) {
    audioUpload.addEventListener("change", async () => {
      const files = audioUpload.files;
      if (files && files.length > 0) {
        await handleFileSelection(files[0]);
      }
    });
  }

  // Playback Control
  if (playbackBtn) {
    playbackBtn.addEventListener("click", () => {
      if (isPlayingPreview) {
        stopPreview();
      } else {
        startPreview();
      }
    });
  }

  // Scan Control
  if (scanBtn) {
    scanBtn.addEventListener("click", () => {
      if (isScanning) {
        stopScan();
      } else {
        startScan(false);
      }
    });
  }

  if (scanBtnPipelineA) {
    scanBtnPipelineA.addEventListener("click", () => {
      if (isScanning) {
        stopScan();
      } else {
        startScan(true);
      }
    });
  }

  // Load separator models catalog
  fetch(prefix + "models/models.json")
    .then(r => r.json())
    .then(catalog => {
      separatorModelCatalog = catalog;
      populateSeparatorDropdown();
    })
    .catch(err => {
      console.warn("[Upload] Failed to load separator models catalog models.json, using fallback presets.", err);
      separatorModelCatalog = [
        { "id": "bird_mixit_4source", "name": "Bird-MixIT 4-Source (ONNX)", "path": "models/bird_mixit_4source.onnx" }
      ];
      populateSeparatorDropdown();
    });

  function populateSeparatorDropdown() {
    const select = document.getElementById("separatorModelSelect");
    if (!select) return;

    select.innerHTML = "";

    separatorModelCatalog.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.name;
      select.appendChild(opt);
    });

    const dspOpt = document.createElement("option");
    dspOpt.value = "dsp";
    dspOpt.textContent = "DSP Crossover Filter (CPU)";
    select.appendChild(dspOpt);

    const customOpt = document.createElement("option");
    customOpt.value = "custom";
    customOpt.textContent = customModelName ? `Custom: ${customModelName}` : "Load Custom Local ONNX Model...";
    select.appendChild(customOpt);

    select.value = selectedSeparatorModel;

    select.addEventListener("change", () => {
      const val = select.value;
      if (val === "custom") {
        const fileInput = document.getElementById("customModelUpload");
        if (fileInput) {
          fileInput.click();
        }
      } else {
        selectedSeparatorModel = val;
        customModelBytes = null;
        customOpt.textContent = "Load Custom Local ONNX Model...";
        logConsole("System", `Separator model changed to: ${val}`, "info");
      }
    });

    const fileInput = document.getElementById("customModelUpload");
    if (fileInput) {
      const newInput = fileInput.cloneNode(true);
      fileInput.parentNode.replaceChild(newInput, fileInput);

      newInput.addEventListener("change", () => {
        const files = newInput.files;
        if (files && files.length > 0) {
          const file = files[0];
          customModelName = file.name;
          customOpt.textContent = `Custom: ${customModelName}`;
          
          selectedSeparatorModel = "custom";
          select.value = "custom";

          logConsole("System", `Loading custom local model: ${file.name}...`, "info");

          const reader = new FileReader();
          reader.onload = (e) => {
            customModelBytes = e.target.result;
            logConsole("System", `Custom local model loaded from disk. Ready to scan.`, "success");
          };
          reader.readAsArrayBuffer(file);
        } else {
          select.value = selectedSeparatorModel;
        }
      });
    }
  }

  /* ==========================================================================
     3. FILE PROCESSING & RESAMPLING
     ========================================================================== */

  async function handleFileSelection(file) {
    if (!file) return;
    
    // Stop any ongoing scan or playback
    stopScan();
    stopPreview();
    stopUploadIsolatedAudio();

    // Reset maps
    verifiedDetections.clear();
    pendingDetections.clear();
    latestDetections = [];
    renderDetections();

    uploadLabel.textContent = "Decoding and resampling audio...";
    logConsole("System", `Decoding file: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)...`, "info");

    try {
      const arrayBuffer = await file.arrayBuffer();
      
      // Decode audio
      const tempCtx = new (window.AudioContext || window.webkitAudioContext)();
      const decodedBuffer = await tempCtx.decodeAudioData(arrayBuffer);
      tempCtx.close();

      logConsole("System", `Decoded successfully. Sample Rate: ${decodedBuffer.sampleRate}Hz, Channels: ${decodedBuffer.numberOfChannels}, Duration: ${decodedBuffer.duration.toFixed(1)}s. Resampling to 48000Hz mono...`, "info");

      // Resample to 48000Hz mono using OfflineAudioContext
      scanDurationSec = decodedBuffer.duration;
      const offlineCtx = new OfflineAudioContext(
        1,
        Math.ceil(scanDurationSec * SAMPLE_RATE),
        SAMPLE_RATE
      );

      const source = offlineCtx.createBufferSource();
      source.buffer = decodedBuffer;
      source.connect(offlineCtx.destination);
      source.start();

      const resampledBuffer = await offlineCtx.startRendering();
      audioBuffer48k = resampledBuffer.getChannelData(0);

      logConsole("System", `Resampled to 48kHz mono successfully. Buffer length: ${audioBuffer48k.length} samples.`, "success");
      uploadLabel.textContent = `Active File: ${file.name}`;
      
      // Enable UI
      analysisContainer.classList.remove("d-none");
      playbackBtn.disabled = false;
      
      // Reset playhead & progress
      scanPlayhead.style.left = "0%";
      scanProgressBar.style.width = "0%";
      scanProgressText.textContent = "0%";
      scanStatusText.textContent = `Ready (${scanDurationSec.toFixed(1)}s)`;

      // Draw Waveform
      drawWaveform(audioBuffer48k);
    } catch (err) {
      console.error(err);
      uploadLabel.textContent = "Failed to load audio file.";
      logConsole("System", `File loading failed: ${err.message}`, "error");
    }
  }

  function drawWaveform(buffer) {
    if (!waveformCanvas) return;
    const canvas = waveformCanvas;
    
    // Set internal canvas resolution to match display size
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;

    const ctx = canvas.getContext("2d");
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const width = rect.width;
    const height = rect.height;
    ctx.clearRect(0, 0, width, height);

    // Dark background gradient
    const bgGrad = ctx.createLinearGradient(0, 0, 0, height);
    bgGrad.addColorStop(0, "#0b1329");
    bgGrad.addColorStop(1, "#030712");
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, width, height);

    // Wave color gradient
    const waveGrad = ctx.createLinearGradient(0, 0, 0, height);
    waveGrad.addColorStop(0, "#06b6d4"); // Cyan
    waveGrad.addColorStop(0.5, "#3b82f6"); // Blue
    waveGrad.addColorStop(1, "#06b6d4");

    ctx.lineWidth = 1;
    ctx.strokeStyle = waveGrad;

    const step = Math.ceil(buffer.length / width);
    const amp = height / 2;

    ctx.beginPath();
    ctx.moveTo(0, amp);

    for (let i = 0; i < width; i++) {
      let min = 1.0;
      let max = -1.0;
      for (let j = 0; j < step; j++) {
        const datum = buffer[i * step + j];
        if (datum < min) min = datum;
        if (datum > max) max = datum;
      }
      ctx.lineTo(i, amp + min * amp);
      ctx.lineTo(i, amp + max * amp);
    }

    ctx.lineTo(width, amp);
    ctx.stroke();

    // Center divider
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.beginPath();
    ctx.moveTo(0, amp);
    ctx.lineTo(width, amp);
    ctx.stroke();
  }

  /* ==========================================================================
     4. SIMULATED TIMELINE SCANNER
     ========================================================================== */

  async function startScan(pipelineAOnly = false) {
    if (!audioBuffer48k || isScanning) return;
    
    stopPreview();
    stopUploadIsolatedAudio();

    // Reset maps
    verifiedDetections.clear();
    pendingDetections.clear();
    latestDetections = [];
    renderDetections();

    // Revoke old object URLs and reset separation debugger
    if (createdObjectURLs && createdObjectURLs.length > 0) {
      createdObjectURLs.forEach(url => {
        try { URL.revokeObjectURL(url); } catch (_) {}
      });
      createdObjectURLs = [];
    }
    separatedChannelBuffers.clear();
    separatedChannelPredictions.clear();
    if (separationDebuggerCard) separationDebuggerCard.classList.add("d-none");
    if (separatedChannelsContainer) separatedChannelsContainer.innerHTML = "";

    isScanning = true;
    isPipelineAOnly = pipelineAOnly;

    if (isPipelineAOnly) {
      if (scanBtnPipelineA) {
        scanBtnPipelineA.textContent = "Stop Pipeline A Scan";
        scanBtnPipelineA.className = "btn btn-sm btn-danger";
      }
      if (scanBtn) scanBtn.disabled = true;
    } else {
      if (scanBtn) {
        scanBtn.textContent = "Stop Scan";
        scanBtn.className = "btn btn-sm btn-danger";
      }
      if (scanBtnPipelineA) scanBtnPipelineA.disabled = true;
    }
    
    scanStatusText.textContent = "Spawning Web Workers...";
    logConsole("System", isPipelineAOnly ? "Starting offline Pipeline A scan..." : "Starting offline consensus scan...", "info");

    activePipelineARequests = 0;
    activePipelineBRequests = 0;
    isTimelineFeedFinished = false;
    liveWorkerReady = false;
    segmentationWorkerReady = false;

    // Fetch settings threshold if available
    try {
      const storedThreshold = localStorage.getItem("bn_threshold");
      if (storedThreshold) {
        detectionThreshold = parseFloat(storedThreshold);
      }
    } catch (_) {}

    // Determine language to use for bird labels
    let currentLabelLang = "en_us";
    try {
      const storedLang = localStorage.getItem("bn_lang");
      if (storedLang) {
        currentLabelLang = storedLang;
      } else {
        const locale = navigator.language || "en-US";
        const l = locale.toLowerCase();
        const base = l.split(/[-_]/)[0];
        const supported = ["en_us", "en_uk", "de", "fr", "es", "it", "nl", "pt", "fi", "sv", "no", "da", "pl", "ru", "uk", "cs", "sk", "sl", "hu", "ro", "tr", "ar", "ja", "ko", "th", "zh", "af"];
        if (supported.includes(l)) {
          currentLabelLang = l;
        } else {
          switch (base) {
            case "en": currentLabelLang = l.includes("gb") || l.includes("uk") ? "en_uk" : "en_us"; break;
            case "de": currentLabelLang = "de"; break;
            case "fr": currentLabelLang = "fr"; break;
            case "es": currentLabelLang = "es"; break;
            case "it": currentLabelLang = "it"; break;
            case "nl": currentLabelLang = "nl"; break;
            case "pt": currentLabelLang = "pt"; break;
            case "fi": currentLabelLang = "fi"; break;
            case "sv": currentLabelLang = "sv"; break;
            case "no": currentLabelLang = "no"; break;
            case "da": currentLabelLang = "da"; break;
            case "pl": currentLabelLang = "pl"; break;
            case "ru": currentLabelLang = "ru"; break;
            case "uk": currentLabelLang = "uk"; break;
            case "cs": currentLabelLang = "cs"; break;
            case "sk": currentLabelLang = "sk"; break;
            case "sl": currentLabelLang = "sl"; break;
            case "hu": currentLabelLang = "hu"; break;
            case "ro": currentLabelLang = "ro"; break;
            case "tr": currentLabelLang = "tr"; break;
            case "ar": currentLabelLang = "ar"; break;
            case "ja": currentLabelLang = "ja"; break;
            case "ko": currentLabelLang = "ko"; break;
            case "th": currentLabelLang = "th"; break;
            case "zh": currentLabelLang = "zh"; break;
            case "af": currentLabelLang = "af"; break;
            default: currentLabelLang = "en_us";
          }
        }
      }
    } catch (_) {}

    // Determine separator model parameters
    let modelPath = "";
    let modelSR = 48000;
    if (selectedSeparatorModel !== "dsp" && selectedSeparatorModel !== "custom") {
      const model = separatorModelCatalog.find(m => m.id === selectedSeparatorModel);
      if (model) {
        modelSR = model.outputSampleRate || 48000;
        const precPath = model.precisions ? model.precisions[separatorPrecision] : null;
        modelPath = prefix + (precPath || model.path);
      } else {
        modelPath = prefix + "models/bird_mixit_4source.onnx"; // fallback
        modelSR = 22050;
      }
    }

    if (isPipelineAOnly) {
      logConsole("System", "Loading Pipeline A only (Live model)...", "info");
      liveWorker = new Worker(prefix + "js/live-worker.js");
      segmentationWorkerReady = true; // pretend B is ready so checkWorkersReadyAndStart triggers
    } else {
      logConsole("System", "Loading Pipeline A (Live model) and Pipeline B (Secondary classifier)...", "info");
      liveWorker = new Worker(prefix + "js/live-worker.js");
      segmentationWorker = new Worker(prefix + "js/segmentation-worker.js");
    }

    liveWorker.onerror = (e) => {
      console.error("[Upload] Live Worker compilation/loading error:", e);
      logConsole("Pipeline A", `Startup Error: ${e.message || "Failed to load script (check browser console)"}`, "error");
      stopScan();
    };

    if (!isPipelineAOnly) {
      segmentationWorker.onerror = (e) => {
        console.error("[Upload] Segmentation Worker compilation/loading error:", e);
        logConsole("Pipeline B", `Startup Error: ${e.message || "Failed to load script (check browser console)"}`, "error");
        stopScan();
      };
    }

    // Send initialization parameters via message passing
    liveWorker.postMessage({
      message: 'init',
      lang: currentLabelLang
    });

    if (!isPipelineAOnly) {
      segmentationWorker.postMessage({
        type: 'INIT',
        payload: {
          separator: selectedSeparatorModel,
          modelPath: modelPath,
          runOnGPU: webgpuEnabled,
          outputSampleRate: modelSR,
          lang: currentLabelLang,
          pipelineBWindow: pipelineBWindow
        }
      });

      if (selectedSeparatorModel === "custom" && customModelBytes) {
        segmentationWorker.postMessage({
          type: "SET_MODEL_BYTES",
          payload: {
            modelBytes: customModelBytes.slice(0)
          }
        }, [customModelBytes.slice(0)]);
      }
    }

    // Setup Worker Events
    liveWorker.onmessage = (e) => {
      const data = e.data || {};
      if (data.message === "pooled" && Array.isArray(data.pooled)) {
        activePipelineARequests--;
        
        // Log top Pipeline A predictions to console
        const topA = data.pooled
          .filter(p => p.confidence > 0.01)
          .sort((a, b) => b.confidence - a.confidence)
          .slice(0, 5)
          .map(p => `${p.commonName} (${(p.confidence * 100).toFixed(1)}%)`)
          .join(', ');
        console.log(`[Upload] Pipeline A predictions: ${topA || "None"}`);

        // Register detections in pendingDetections mapping to the scanner timeline
        const scanTimestamp = simulatedEpochStart + scanTimeSec * 1000;
        let maxConf = 0;
        data.pooled.forEach(p => {
          if (p.confidence > maxConf) maxConf = p.confidence;
          if (p.confidence >= detectionThreshold && p.scientificName) {
            if (isPipelineAOnly) {
              if (!verifiedDetections.has(p.scientificName)) {
                // Slices the original audio for the 3.0s window
                const sliceStart = Math.max(0, Math.round(scanTimeSec * SAMPLE_RATE) - WINDOW_SAMPLES);
                const sliceEnd = Math.min(audioBuffer48k ? audioBuffer48k.length : 0, Math.round(scanTimeSec * SAMPLE_RATE));
                let copy = null;
                if (audioBuffer48k && sliceEnd > sliceStart) {
                  const slice = audioBuffer48k.subarray(sliceStart, sliceEnd);
                  copy = new Float32Array(slice);
                }
                verifiedDetections.set(p.scientificName, {
                  confidence: p.confidence,
                  audioBuffer: copy,
                  timestamp: scanTimestamp
                });
                logConsole("Pipeline A", `VERIFIED (Pipeline A Only): ${p.commonName} (${(p.confidence * 100).toFixed(0)}%) at ${formatTime(scanTimeSec)}`, "success");
              }
            } else {
              if (!verifiedDetections.has(p.scientificName) && !pendingDetections.has(p.scientificName)) {
                pendingDetections.set(p.scientificName, scanTimestamp);
                logConsole("Pipeline A", `Draft detected: ${p.commonName} (${(p.confidence * 100).toFixed(0)}%) at ${formatTime(scanTimeSec)}`, "warning");
              }
            }
          }
        });
        maxLiveConfidenceInInterval = Math.max(maxLiveConfidenceInInterval, maxConf);
        
        // Append new detections to latestDetections
        data.pooled.forEach(p => {
          if (p.confidence >= detectionThreshold) {
            const exists = latestDetections.some(d => d.scientificName === p.scientificName);
            if (!exists) {
              latestDetections.push(p);
            } else {
              // Update confidence
              const match = latestDetections.find(d => d.scientificName === p.scientificName);
              if (match) match.confidence = p.confidence;
            }
          }
        });

        renderDetections();
        checkIfScanFinished();
      } else if (data.message === "loaded") {
        logConsole("Pipeline A", "Live Worker loaded and warmed up successfully.", "success");
        liveWorkerReady = true;
        checkWorkersReadyAndStart();
      } else if (["load_model", "warmup", "load_geomodel", "load_labels"].includes(data.message)) {
        if (typeof data.progress === "number") {
          logConsole("Pipeline A", `Loading: ${data.progress}%`, "info");
        }
      } else if (data.message === "worker_error") {
        activePipelineARequests--;
        logConsole("Pipeline A", `Error: ${data.error}`, "error");
        if (!liveWorkerReady || !segmentationWorkerReady) {
          stopScan();
          logConsole("System", "Scan failed: Pipeline A worker error during startup.", "error");
        } else {
          checkIfScanFinished();
        }
      }
    };

    if (!isPipelineAOnly) {
      segmentationWorker.onmessage = (e) => {
        const { type, payload } = e.data || {};
        
        if (type === "PIPELINE_STATUS") {
          if (payload.status === "ready") {
            logConsole("Pipeline B", "Pipeline B Worker loaded and warmed up successfully.", "success");
            segmentationWorkerReady = true;
            checkWorkersReadyAndStart();
          } else if (payload.status === "error") {
            logConsole("Pipeline B", `Initialization error: ${payload.message}`, "error");
            if (!liveWorkerReady || !segmentationWorkerReady) {
              stopScan();
              logConsole("System", "Scan failed: Pipeline B worker error during startup.", "error");
            }
          } else {
            logConsole("Pipeline B", `Status update: ${payload.message}`, "warning");
          }
        } else if (type === "SEGMENT_RESULT") {
          activePipelineBRequests--;
          const { segmentId, timestamp, results, error } = payload || {};
          if (error) {
            logConsole("Pipeline B", `Error in segment ${segmentId}: ${error}`, "error");
            checkIfScanFinished();
            return;
          }

          const elapsedSec = (timestamp - simulatedEpochStart) / 1000;
          logConsole("Pipeline B", `Completed segment separation for timestamp ${formatTime(elapsedSec)}`, "info");

          if (results && results.length > 0) {
            let updated = false;
            const verifiedThisSegment = new Set();

            results.forEach(res => {
              // Accumulate raw buffers for the separation debugger UI
              if (res.audioBuffer) {
                if (!separatedChannelBuffers.has(res.channelId)) {
                  separatedChannelBuffers.set(res.channelId, []);
                }
                separatedChannelBuffers.get(res.channelId).push(res.audioBuffer);
              }

              // Accumulate predictions for the separation debugger UI (max pooled across segments)
              if (res.predictions && res.predictions.length > 0) {
                if (!separatedChannelPredictions.has(res.channelId)) {
                  separatedChannelPredictions.set(res.channelId, new Map());
                }
                const channelPredsMap = separatedChannelPredictions.get(res.channelId);
                res.predictions.forEach(p => {
                  if (p.scientificName) {
                    const existing = channelPredsMap.get(p.scientificName);
                    if (!existing) {
                      channelPredsMap.set(p.scientificName, {
                        commonName: p.commonName || p.scientificName,
                        confidence: p.confidence
                      });
                    } else {
                      existing.confidence = Math.max(existing.confidence, p.confidence);
                    }
                  }
                });
              }

              // Log separated channel predictions to console
              const topB = res.predictions
                .sort((a, b) => b.confidence - a.confidence)
                .slice(0, 3)
                .map(p => `${p.commonName} (${(p.confidence * 100).toFixed(3)}%)`)
                .join(', ');
              console.log(`[Upload] Pipeline B Channel ${res.channelId} predictions: ${topB}`);

              if (res.predictions && res.predictions.length > 0) {
                res.predictions.forEach(pred => {
                  if (pred.scientificName && pred.confidence >= detectionThreshold) {
                    logConsole("Consensus", `VERIFIED: ${pred.commonName} (${(pred.confidence * 100).toFixed(0)}%) on isolated channel ${res.channelId} at ${formatTime(elapsedSec)}`, "success");
                    
                    console.log(`[Consensus] Storing verified detection: ${pred.scientificName}, audioBuffer:`, res.audioBuffer);
                    verifiedDetections.set(pred.scientificName, {
                      confidence: pred.confidence,
                      audioBuffer: res.audioBuffer, // Float32Array isolated channel audio
                      timestamp: timestamp
                    });
                    verifiedThisSegment.add(pred.scientificName);
                    pendingDetections.delete(pred.scientificName);

                    // Append verified species to latestDetections so it appears in the UI
                    const exists = latestDetections.some(d => d.scientificName === pred.scientificName);
                    if (!exists) {
                      latestDetections.push({
                        scientificName: pred.scientificName,
                        commonName: pred.commonName,
                        commonNameI18n: pred.commonNameI18n || pred.commonName,
                        confidence: pred.confidence
                      });
                    } else {
                      const match = latestDetections.find(d => d.scientificName === pred.scientificName);
                      if (match) {
                        match.confidence = Math.max(match.confidence, pred.confidence);
                      }
                    }
                    updated = true;
                  }
                });
              }
            });

            // Timeline reconciliation: reject unverified detections
            const segmentStart = timestamp - (pipelineBWindow * 1000);
            const segmentEnd = timestamp;

            pendingDetections.forEach((detectedAt, sciName) => {
              if (detectedAt >= segmentStart && detectedAt <= segmentEnd) {
                if (!verifiedThisSegment.has(sciName)) {
                  logConsole("Consensus", `REJECTED: False positive unmounted (${sciName}) at ${formatTime(elapsedSec)}`, "error");
                  pendingDetections.delete(sciName);
                  
                  // Remove from latestDetections
                  latestDetections = latestDetections.filter(d => d.scientificName !== sciName);
                  updated = true;
                }
              }
            });

            if (updated) {
              renderDetections();
            }
          }
          checkIfScanFinished();
        }
      };
    }

    function checkWorkersReadyAndStart() {
      if (liveWorkerReady && segmentationWorkerReady) {
        beginScan();
      }
    }

    function beginScan() {
      scanStatusText.textContent = "Scanning...";
      logConsole("System", isPipelineAOnly ? "Pipeline A worker initialized. Commencing scan..." : "Both workers initialized. Commencing scan...", "info");

      // Determine if we need to run Pipeline B immediately on short files
      let runPipelineBImmediately = false;
      if (!isPipelineAOnly && scanDurationSec < pipelineBWindow) {
        runPipelineBImmediately = true;
      }
      
      if (runPipelineBImmediately) {
        const copy = new Float32Array(audioBuffer48k);
        const simulatedTimestamp = simulatedEpochStart + scanDurationSec * 1000;
        activePipelineBRequests++;
        segmentationWorker.postMessage({
          type: "PROCESS_SEGMENT",
          payload: {
            segmentId: `seg-upload-short-${simulatedTimestamp}`,
            timestamp: simulatedTimestamp,
            sampleRate: SAMPLE_RATE,
            audioBuffer: copy,
            meta: {}
          }
        }, [copy.buffer]);
        logConsole("Pipeline B", `Triggered single segment run for short audio file (duration: ${scanDurationSec.toFixed(1)}s)`, "info");
      }

      // Get Scanning Speed pacing
      const speedSelect = document.querySelector('input[name="scanSpeed"]:checked');
      const speed = speedSelect ? speedSelect.value : "normal";
      
      let pacingMs = 500; // time interval for step executions
      let timeStepSec = 0.5; // step increment in simulated seconds
      if (speed === "fast") pacingMs = 50; // 10x faster
      if (speed === "turbo") pacingMs = 5; // ~100x faster

      scanTimeSec = 0.0;
      let nextPipelineBTime = pipelineBWindow;
      maxLiveConfidenceInInterval = 0;
      
      // Start interval scan
      scanIntervalId = setInterval(async () => {
        scanTimeSec += timeStepSec;
        
        if (scanTimeSec > scanDurationSec) {
          clearInterval(scanIntervalId);
          scanIntervalId = null;
          
          // Post one final Pipeline A slice for the very end of the file if duration >= 3.0
          if (scanDurationSec >= 3.0) {
            const startSample = audioBuffer48k.length - WINDOW_SAMPLES;
            const slice = audioBuffer48k.subarray(startSample, audioBuffer48k.length);
            const copy = new Float32Array(slice);
            activePipelineARequests++;
            liveWorker.postMessage({
              message: "predict",
              pcmAudio: copy,
              overlapSec: 1.5,
              sensitivity: 1.0
            }, [copy.buffer]);
          } else {
            // If the file is shorter than 3.0s, pad it to 3.0s and run Pipeline A once
            const slice = new Float32Array(WINDOW_SAMPLES);
            slice.set(audioBuffer48k);
            activePipelineARequests++;
            liveWorker.postMessage({
              message: "predict",
              pcmAudio: slice,
              overlapSec: 1.5,
              sensitivity: 1.0
            });
          }

          // Post one final Pipeline B slice for the very end of the file if duration >= pipelineBWindow
          if (!isPipelineAOnly && scanDurationSec >= pipelineBWindow) {
            const windowSamples = Math.round(pipelineBWindow * SAMPLE_RATE);
            const startSample = audioBuffer48k.length - windowSamples;
            const slice = audioBuffer48k.subarray(startSample, audioBuffer48k.length);
            const copy = new Float32Array(slice);
            activePipelineBRequests++;
            
            const simulatedTimestamp = simulatedEpochStart + scanDurationSec * 1000;
            segmentationWorker.postMessage({
              type: "PROCESS_SEGMENT",
              payload: {
                segmentId: `seg-upload-final-${simulatedTimestamp}`,
                timestamp: simulatedTimestamp,
                sampleRate: SAMPLE_RATE,
                audioBuffer: copy,
                meta: {}
              }
            }, [copy.buffer]);
          }

          isTimelineFeedFinished = true;
          checkIfScanFinished();
          return;
        }

        // Update UI Progress
        const pct = (scanTimeSec / scanDurationSec) * 100;
        scanPlayhead.style.left = `${pct}%`;
        scanProgressBar.style.width = `${pct}%`;
        scanProgressText.textContent = `${Math.min(100, Math.round(pct))}%`;
        scanStatusText.textContent = `Scanning: ${formatTime(scanTimeSec)} / ${formatTime(scanDurationSec)}`;

        // Calculate sample endpoints
        const endSample = Math.round(scanTimeSec * SAMPLE_RATE);

        // Pipeline A (Live): run every 1.0s on the last 3.0s of audio
        if (Number.isInteger(scanTimeSec) && scanTimeSec >= 3.0) {
          const startSample = endSample - WINDOW_SAMPLES;
          if (startSample >= 0 && startSample + WINDOW_SAMPLES <= audioBuffer48k.length) {
            const slice = audioBuffer48k.subarray(startSample, startSample + WINDOW_SAMPLES);
            // Transfer copy to worker
            const copy = new Float32Array(slice);
            activePipelineARequests++;
            liveWorker.postMessage({
              message: "predict",
              pcmAudio: copy,
              overlapSec: 1.5,
              sensitivity: 1.0
            }, [copy.buffer]);
          }
        }

        // Pipeline B (Segmented): run dynamically based on window and stride
        if (!isPipelineAOnly && scanTimeSec >= nextPipelineBTime - 0.001) {
          const windowSamples = Math.round(pipelineBWindow * SAMPLE_RATE);
          const startSample = endSample - windowSamples;
          
          if (startSample >= 0 && startSample + windowSamples <= audioBuffer48k.length) {
            const slice = audioBuffer48k.subarray(startSample, startSample + windowSamples);
            
            let bypassB = false;

            // AAD Gate Check
            if (aadEnabled) {
              const rms = calculateRMS(slice);
              const centroid = calculateSpectralCentroid(slice, SAMPLE_RATE);
              if (rms < aadRmsThreshold || centroid < aadCentroidMin) {
                logConsole("Pipeline B", `Bypassed via gate at ${formatTime(scanTimeSec)} (rms: ${rms.toFixed(4)} < ${aadRmsThreshold} OR centroid: ${Math.round(centroid)}Hz < ${aadCentroidMin}Hz)`, "info");
                bypassB = true;
              }
            }

            // Early Exit Check
            if (!bypassB && earlyExitEnabled && maxLiveConfidenceInInterval >= earlyExitConfidence) {
              logConsole("Pipeline B", `Bypassed via early exit at ${formatTime(scanTimeSec)} (max live confidence: ${(maxLiveConfidenceInInterval * 100).toFixed(1)}% >= ${(earlyExitConfidence * 100).toFixed(0)}%)`, "info");
              bypassB = true;

              // Immediately verify all pending drafts in this segment window using the raw audio slice as fallback
              const windowSize = pipelineBWindow;
              const segmentStart = (simulatedEpochStart + scanTimeSec * 1000) - (windowSize * 1000);
              const segmentEnd = simulatedEpochStart + scanTimeSec * 1000;
              const sliceCopy = new Float32Array(slice);

              pendingDetections.forEach((detectedAt, sciName) => {
                if (detectedAt >= segmentStart && detectedAt <= segmentEnd) {
                  logConsole("Consensus", `VERIFIED (Early Exit Fallback): ${sciName} using raw audio slice`, "success");
                  const match = latestDetections.find(d => d.scientificName === sciName);
                  const conf = match ? match.confidence : maxLiveConfidenceInInterval;

                  verifiedDetections.set(sciName, {
                    confidence: conf,
                    audioBuffer: sliceCopy,
                    timestamp: segmentEnd
                  });
                  pendingDetections.delete(sciName);
                }
              });
              renderDetections();
            }

            maxLiveConfidenceInInterval = 0; // reset for next interval

            if (!bypassB) {
              // Transfer copy to worker
              const copy = new Float32Array(slice);
              const simulatedTimestamp = simulatedEpochStart + scanTimeSec * 1000;
              
              activePipelineBRequests++;
              segmentationWorker.postMessage({
                type: "PROCESS_SEGMENT",
                payload: {
                  segmentId: `seg-upload-${simulatedTimestamp}`,
                  timestamp: simulatedTimestamp,
                  sampleRate: SAMPLE_RATE,
                  audioBuffer: copy,
                  meta: {}
                }
              }, [copy.buffer]);
            }
          }
          nextPipelineBTime += pipelineBStride;
        }
      }, pacingMs);
    }
  }

  function stopScan() {
    if (scanIntervalId) {
      clearInterval(scanIntervalId);
      scanIntervalId = null;
    }
    isScanning = false;
    liveWorkerReady = false;
    segmentationWorkerReady = false;
    if (scanBtn) {
      scanBtn.textContent = "Start Scan";
      scanBtn.className = "btn btn-sm btn-success";
      scanBtn.disabled = false;
    }
    if (scanBtnPipelineA) {
      scanBtnPipelineA.textContent = "Start Pipeline A Scan (BirdNET only)";
      scanBtnPipelineA.className = "btn btn-sm btn-outline-success";
      scanBtnPipelineA.disabled = false;
    }
    if (scanStatusText) {
      scanStatusText.textContent = "Scan stopped.";
    }

    // Terminate workers
    if (liveWorker) {
      liveWorker.terminate();
      liveWorker = null;
    }
    if (segmentationWorker) {
      segmentationWorker.terminate();
      segmentationWorker = null;
    }
  }

  function completeScan() {
    stopScan();
    logConsole("System", "Analysis complete! All segments parsed through consensus validation.", "success");
    if (scanStatusText) {
      scanStatusText.textContent = `Completed (${scanDurationSec.toFixed(1)}s)`;
    }
    
    // Force progress UI to 100% on complete
    if (scanProgressBar) {
      scanProgressBar.style.width = "100%";
    }
    if (scanProgressText) {
      scanProgressText.textContent = "100%";
    }
    if (scanPlayhead) {
      scanPlayhead.style.left = "100%";
    }

    // Reconcile remaining unverified elements
    if (isPipelineAOnly) {
      pendingDetections.clear();
    } else {
      pendingDetections.forEach((detectedAt, sciName) => {
        logConsole("Consensus", `CLEANUP: Removing unverified species: ${sciName}`, "error");
        latestDetections = latestDetections.filter(d => d.scientificName !== sciName);
      });
      pendingDetections.clear();
    }
    renderDetections();

    // Render the separated channels debugger UI
    if (!isPipelineAOnly && separatedChannelBuffers.size > 0) {
      if (separatedChannelsContainer && separationDebuggerCard) {
        separatedChannelsContainer.innerHTML = "";
        separationDebuggerCard.classList.remove("d-none");



        // Loop over accumulated channels in order
        const sortedChannelIds = Array.from(separatedChannelBuffers.keys()).sort((a, b) => a - b);
        sortedChannelIds.forEach(channelId => {
          const arrays = separatedChannelBuffers.get(channelId);
          if (arrays && arrays.length > 0) {
            const rawTrack = concatenateFloat32Arrays(arrays);
            
            // Peak normalize the track so it's clearly audible
            let maxVal = 0;
            for (let i = 0; i < rawTrack.length; i++) {
              const abs = Math.abs(rawTrack[i]);
              if (abs > maxVal) maxVal = abs;
            }

            const normalized = new Float32Array(rawTrack.length);
            let gain = 1.0;
            if (maxVal > 0.0001) {
              gain = 0.8 / maxVal;
              const clampedGain = Math.min(20.0, gain); // up to 20x gain boost
              gain = clampedGain;
              for (let i = 0; i < rawTrack.length; i++) {
                normalized[i] = rawTrack[i] * clampedGain;
              }
            } else {
              normalized.set(rawTrack);
            }

            // Convert to WAV Blob
            const wavBlob = bufferToWav(normalized, SAMPLE_RATE);
            const objectURL = URL.createObjectURL(wavBlob);
            createdObjectURLs.push(objectURL);

            // Build predictions list html for this channel (no threshold filter, show all)
            let predictionsHtml = "";
            const channelPreds = separatedChannelPredictions.get(channelId);
            if (channelPreds && channelPreds.size > 0) {
              const sortedPreds = Array.from(channelPreds.values())
                .sort((a, b) => b.confidence - a.confidence);
              
              predictionsHtml = `
                <div class="mt-2 pt-2 border-top">
                  <span class="text-muted d-block mb-1 small fw-bold" style="font-size: 0.7rem;">
                    <i class="bi bi-tags me-1"></i>Detected Species (All Confidences):
                  </span>
                  <div class="d-flex flex-wrap gap-1">
                    ${sortedPreds.map(p => {
                      let badgeClass = "bg-secondary-subtle text-secondary border border-secondary-subtle";
                      if (p.confidence >= 0.15) {
                        badgeClass = "bg-success-subtle text-success border border-success";
                      } else if (p.confidence >= 0.05) {
                        badgeClass = "bg-warning-subtle text-warning border border-warning";
                      }
                      return `<span class="badge ${badgeClass} py-1 px-2 rounded-2" style="font-size: 0.7rem; font-weight: normal;">
                        ${p.commonName}: ${(p.confidence * 100).toFixed(1)}%
                      </span>`;
                    }).join("")}
                  </div>
                </div>
              `;
            } else {
              predictionsHtml = `
                <div class="mt-2 pt-2 border-top">
                  <span class="text-muted d-block small" style="font-size: 0.7rem;">No species detected.</span>
                </div>
              `;
            }

            // Create card UI element
            const col = document.createElement("div");
            col.className = "col-md-6 col-12 fade-in mb-3";
            col.innerHTML = `
              <div class="card border border-light shadow-sm rounded-3 overflow-hidden h-100">
                <div class="card-body p-3 bg-light d-flex flex-column justify-content-between">
                  <div>
                    <div class="d-flex justify-content-between align-items-center mb-2">
                      <span class="fw-bold text-dark mb-0 small">
                        <i class="bi bi-speaker me-1 text-primary"></i> Separated Channel ${channelId}
                      </span>
                      <span class="badge bg-secondary text-white font-monospace" style="font-size: 0.65rem;">
                        Peak: ${maxVal.toFixed(3)} | Gain: ${gain.toFixed(1)}x
                      </span>
                    </div>
                    <audio src="${objectURL}" controls class="w-100 mb-2" style="height: 32px;"></audio>
                    ${predictionsHtml}
                  </div>
                  <div class="text-end mt-3">
                    <a href="${objectURL}" download="separated_channel_${channelId}.wav" class="btn btn-xs btn-outline-secondary py-1 px-2 font-monospace" style="font-size: 0.7rem;">
                      <i class="bi bi-download me-1"></i>Download WAV
                    </a>
                  </div>
                </div>
              </div>
            `;
            separatedChannelsContainer.appendChild(col);
          }
        });
      }
    }
  }

  function checkIfScanFinished() {
    if (isTimelineFeedFinished && activePipelineARequests === 0 && activePipelineBRequests === 0) {
      completeScan();
    }
  }

  /* ==========================================================================
     5. RENDER DETECTIONS TIMELINE
     ========================================================================== */

  function renderDetections() {
    if (!uploadDetectionsList) return;
    
    if (latestDetections.length === 0) {
      uploadDetectionsList.innerHTML = `
        <div class="col-12 text-center text-muted py-5 card border-0 shadow-sm rounded-3">
          <div class="card-body">
            <i class="bi bi-soundwave fs-1 d-block mb-3 opacity-25"></i>
            <p>Upload a file and start scanning to extract species detections.</p>
          </div>
        </div>
      `;
      return;
    }

    uploadDetectionsList.innerHTML = "";

    latestDetections.forEach((p) => {
      const confPct = (p.confidence * 100).toFixed(0);
      const commonName = p.commonNameI18n || p.commonName || `Class ${p.index}`;
      const scientificName = p.scientificName || "";
      const isVerified = verifiedDetections.has(scientificName);

      const cardPendingClass = isVerified ? "" : "card-pending";

      const badgeHtml = isVerified
        ? `<span class="badge bg-primary bg-opacity-10 text-primary border border-primary border-opacity-10 flex-shrink-0">${confPct}%</span>`
        : `<span class="badge bg-warning bg-opacity-10 text-warning border border-warning border-opacity-10 flex-shrink-0"><i class="bi bi-arrow-repeat spin-icon me-1"></i>${confPct}% (Analyzing)</span>`;

      const playBtnHtml = isVerified
        ? (currentlyPlayingSpecies === scientificName
            ? `<button class="btn btn-sm btn-outline-danger py-0 px-2 play-audio-btn" style="font-size: 0.75rem;" onclick="stopUploadIsolatedAudio()">
                 <i class="bi bi-stop-fill me-1"></i>Stop
               </button>`
            : `<button class="btn btn-sm btn-outline-primary py-0 px-2 play-audio-btn" style="font-size: 0.75rem;" onclick="playUploadIsolatedAudio('${scientificName.replace(/'/g, "\\'")}')" ${!verifiedDetections.get(scientificName)?.audioBuffer ? 'disabled' : ''}>
                 <i class="bi bi-play-fill me-1"></i>Play Isolated
               </button>`
          )
        : `<button class="btn btn-sm btn-outline-secondary py-0 px-2 play-audio-btn" style="font-size: 0.75rem;" disabled>
             <i class="bi bi-hourglass me-1"></i>Analyzing
           </button>`;

      const downloadBtnHtml = isVerified && verifiedDetections.get(scientificName)?.audioBuffer
        ? `<button class="btn btn-sm btn-outline-secondary py-0 px-2 ms-1" style="font-size: 0.75rem;" onclick="downloadUploadIsolatedAudio('${scientificName.replace(/'/g, "\\'")}')" title="Download isolated audio channel">
             <i class="bi bi-download"></i>
           </button>`
        : '';

      const wikiLang = (navigator.language || "en").split("-")[0];
      const wikiUrl = `https://${wikiLang}.wikipedia.org/wiki/${encodeURIComponent(scientificName)}`;
      const ebirdUrl = `https://www.google.com/search?q=site:ebird.org/species/+${encodeURIComponent(scientificName)}`;

      const cardCol = document.createElement("div");
      cardCol.className = "col-md-6 col-lg-12 fade-in";
      cardCol.innerHTML = `
        <div class="card border-0 shadow-sm overflow-hidden ${cardPendingClass}">
          <div class="d-flex h-100">
            <div class="flex-shrink-0 position-relative" style="width: 90px; background-color: #f8f9fa;">
              <img src="../img/dummy.webp" 
                   data-scientific-name="${scientificName}"
                   alt="${commonName}"
                   loading="lazy"
                   style="width: 100%; height: 100%; object-fit: cover;"
                   onerror="this.onerror=null; this.src='../img/dummy.webp';">
            </div>
            <div class="card-body py-2 px-3 flex-grow-1 d-flex flex-column justify-content-between">
              <div>
                <div class="d-flex justify-content-between align-items-start mb-1">
                  <h6 class="card-title mb-0 fw-bold text-primary text-truncate me-2" style="min-width: 0; font-size: 0.95rem;" title="${commonName}">${commonName}</h6>
                  ${badgeHtml}
                </div>
                ${scientificName ? `
                  <div class="text-muted fst-italic small mb-1 text-truncate" style="font-size: 0.8rem;">${scientificName}</div>
                  <div class="d-flex align-items-center gap-2 mb-2">
                    <a href="${wikiUrl}" target="_blank" rel="noopener" class="species-link" title="Wikipedia">
                      <i class="bi bi-wikipedia"></i> Wikipedia
                    </a>
                    <span class="species-links-divider">|</span>
                    <a href="${ebirdUrl}" target="_blank" rel="noopener" data-ebird-scientific="${scientificName}" class="species-link" title="eBird">
                      <i class="bi bi-box-arrow-up-right"></i> eBird
                    </a>
                  </div>
                ` : ""}
              </div>
              <div>
                <div class="d-flex justify-content-between align-items-center border-top pt-2 mt-1">
                  <span class="small text-muted" style="font-size: 0.75rem;">
                    Consensus engine verified track
                  </span>
                  <div class="play-btn-container">
                    ${playBtnHtml}
                    ${downloadBtnHtml}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      `;
      uploadDetectionsList.appendChild(cardCol);

      const imgEl = cardCol.querySelector('img[data-scientific-name]');
      if (imgEl) {
        loadSpeciesImage(scientificName, imgEl, '../img/dummy.webp');
      }
      const ebirdLinkEl = cardCol.querySelector(`a[data-ebird-scientific="${scientificName}"]`);
      if (ebirdLinkEl) {
        loadEbirdLink(scientificName, ebirdLinkEl);
      }
    });
  }

  /* ==========================================================================
     6. ORIGINAL PLAYBACK & ISOLATED PLAYBACK
     ========================================================================== */

  function startPreview() {
    if (!audioBuffer48k || isPlayingPreview) return;
    
    stopUploadIsolatedAudio();

    try {
      if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
      }
      if (audioContext.state === "suspended") {
        audioContext.resume();
      }

      // Convert audioBuffer48k back to standard AudioBuffer object
      const previewBuffer = audioContext.createBuffer(1, audioBuffer48k.length, SAMPLE_RATE);
      previewBuffer.getChannelData(0).set(audioBuffer48k);

      previewSource = audioContext.createBufferSource();
      previewSource.buffer = previewBuffer;
      previewSource.connect(audioContext.destination);

      previewSource.onended = () => {
        if (isPlayingPreview) {
          stopPreview();
        }
      };

      previewStartTime = audioContext.currentTime;
      previewSource.start(0, previewOffset);
      isPlayingPreview = true;
      playbackBtn.innerHTML = '<i class="bi bi-stop-fill me-1"></i>Stop Playback';

      // Keep updating playhead visually while playing
      updatePreviewPlayhead();
    } catch (e) {
      console.error(e);
      logConsole("System", `Playback failed: ${e.message}`, "error");
    }
  }

  function stopPreview() {
    if (previewSource) {
      try { previewSource.stop(); } catch (_) {}
      previewSource = null;
    }
    isPlayingPreview = false;
    if (playbackBtn) {
      playbackBtn.innerHTML = '<i class="bi bi-play-fill me-1"></i>Listen to File';
    }
    previewOffset = 0;
    scanPlayhead.style.left = "0%";
  }

  function updatePreviewPlayhead() {
    if (!isPlayingPreview || !audioContext) return;
    
    const elapsed = audioContext.currentTime - previewStartTime + previewOffset;
    if (elapsed > scanDurationSec) {
      scanPlayhead.style.left = "100%";
      return;
    }

    const pct = (elapsed / scanDurationSec) * 100;
    scanPlayhead.style.left = `${pct}%`;

    requestAnimationFrame(updatePreviewPlayhead);
  }

  // Isolated Playback
  async function playUploadIsolatedAudio(scientificName) {
    console.log("[playUploadIsolatedAudio] Request to play:", scientificName);
    if (currentlyPlayingSpecies === scientificName) {
      console.log("[playUploadIsolatedAudio] Stopping currently playing:", scientificName);
      stopUploadIsolatedAudio();
      return;
    }

    stopPreview();

    const verified = verifiedDetections.get(scientificName);
    console.log("[playUploadIsolatedAudio] verified entry from Map:", verified);
    if (!verified) {
      console.warn("[playUploadIsolatedAudio] No verified entry found for:", scientificName);
      return;
    }
    const samples = getBestIsolatedBufferForSpecies(scientificName);
    if (!samples) {
      console.warn("[playUploadIsolatedAudio] No audioBuffer found for:", scientificName);
      return;
    }
    console.log("[playUploadIsolatedAudio] retrieved samples length:", samples.length);

    try {
      stopUploadIsolatedAudio();

      if (!playbackAudioContext) {
        try {
          playbackAudioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
        } catch (e) {
          console.warn("[playUploadIsolatedAudio] Failed to create AudioContext with sampleRate, falling back to default constructor:", e);
          playbackAudioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
      }
      if (playbackAudioContext.state === "suspended") {
        await playbackAudioContext.resume();
      }

      // Peak normalize the track so it's clearly audible
      let maxVal = 0;
      for (let i = 0; i < samples.length; i++) {
        const abs = Math.abs(samples[i]);
        if (abs > maxVal) maxVal = abs;
      }

      const normalized = new Float32Array(samples.length);
      if (maxVal > 0.0001) {
        const gain = 0.8 / maxVal;
        const clampedGain = Math.min(20.0, gain); // up to 20x gain boost
        console.log(`[playUploadIsolatedAudio] Normalizing audio. Peak: ${maxVal.toFixed(4)}, Applied Gain: ${clampedGain.toFixed(2)}x`);
        for (let i = 0; i < samples.length; i++) {
          normalized[i] = samples[i] * clampedGain;
        }
      } else {
        console.log("[playUploadIsolatedAudio] Track is near-silent. Playing raw samples.");
        normalized.set(samples);
      }

      const audioBuf = playbackAudioContext.createBuffer(1, normalized.length, playbackAudioContext.sampleRate);
      audioBuf.getChannelData(0).set(normalized);

      const source = playbackAudioContext.createBufferSource();
      source.buffer = audioBuf;
      source.connect(playbackAudioContext.destination);
      
      source.onended = () => {
        if (activeAudioSource === source) {
          currentlyPlayingSpecies = null;
          activeAudioSource = null;
          renderDetections();
        }
      };

      source.start();
      activeAudioSource = source;
      currentlyPlayingSpecies = scientificName;
      renderDetections();
    } catch (err) {
      console.error("Failed to play isolated audio:", err);
    }
  }

  function stopUploadIsolatedAudio() {
    if (activeAudioSource) {
      try { activeAudioSource.stop(); } catch(e) {}
      activeAudioSource = null;
    }
    currentlyPlayingSpecies = null;
    renderDetections();
  }

  /* ==========================================================================
     7. SYSTEM LOGGING & FORMATTERS
     ========================================================================== */

  function logConsole(module, text, type = "info") {
    if (!debugLogConsole) return;
    const timeStr = new Date().toTimeString().split(' ')[0];
    let colorClass = "text-white";
    if (type === "success") colorClass = "text-success";
    if (type === "warning") colorClass = "text-warning";
    if (type === "error") colorClass = "text-danger";

    const div = document.createElement("div");
    div.innerHTML = `<span class="text-secondary">[${timeStr}]</span> <span class="${colorClass}">[${module}]</span> ${text}`;
    debugLogConsole.prepend(div);
  }

  function formatTime(sec) {
    const mins = Math.floor(sec / 60);
    const secs = Math.floor(sec % 60);
    const ms = Math.floor((sec % 1) * 10);
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}.${ms}`;
  }

  function calculateRMS(samples) {
    let sum = 0;
    const len = samples.length;
    for (let i = 0; i < len; i++) {
      sum += samples[i] * samples[i];
    }
    return Math.sqrt(sum / len);
  }

  function calculateSpectralCentroid(samples, sampleRate) {
    let crossings = 0;
    const len = samples.length;
    if (len <= 1) return 0;
    for (let i = 1; i < len; i++) {
      if ((samples[i] >= 0 && samples[i - 1] < 0) || (samples[i] < 0 && samples[i - 1] >= 0)) {
        crossings++;
      }
    }
    const duration = len / sampleRate;
    return crossings / (2 * duration);
  }

  function bindRange(id, initialValue, onChange, format, storageKey) {
    const input = document.getElementById(id);
    const label = document.querySelector(`[id='${id.replace("Range", "Value")}']`);
    if (!input) return;
    if (typeof initialValue === "number") {
      input.value = initialValue;
    }
    const setLabel = (val) => {
      if (label) label.textContent = format ? format(val) : val;
    };
    setLabel(parseFloat(input.value));
    input.addEventListener("input", () => {
      const val = parseFloat(input.value);
      onChange(val, input);
      setLabel(val);
      if (storageKey) localStorage.setItem(storageKey, val.toString());
    });
  }

  function initSettingsControls() {
    // AAD Gate Toggle
    const aadGateToggle = document.getElementById("aadGateToggle");
    if (aadGateToggle) {
      aadGateToggle.checked = aadEnabled;
      const updateContainers = () => {
        const container1 = document.getElementById("aadRmsContainer");
        const container2 = document.getElementById("aadCentroidContainer");
        if (container1) container1.classList.toggle("d-none", !aadEnabled);
        if (container2) container2.classList.toggle("d-none", !aadEnabled);
      };
      aadGateToggle.addEventListener("change", () => {
        aadEnabled = aadGateToggle.checked;
        localStorage.setItem("bn_aad_enabled", aadEnabled.toString());
        updateContainers();
      });
      updateContainers();
    }

    // Early Exit Toggle
    const earlyExitToggle = document.getElementById("earlyExitToggle");
    if (earlyExitToggle) {
      earlyExitToggle.checked = earlyExitEnabled;
      const updateContainer = () => {
        const container = document.getElementById("earlyExitConfidenceContainer");
        if (container) container.classList.toggle("d-none", !earlyExitEnabled);
      };
      earlyExitToggle.addEventListener("change", () => {
        earlyExitEnabled = earlyExitToggle.checked;
        localStorage.setItem("bn_early_exit_enabled", earlyExitEnabled.toString());
        updateContainer();
      });
      updateContainer();
    }

    // WebGPU Toggle
    const webgpuToggle = document.getElementById("webgpuToggle");
    if (webgpuToggle) {
      webgpuToggle.checked = webgpuEnabled;
      webgpuToggle.addEventListener("change", () => {
        webgpuEnabled = webgpuToggle.checked;
        localStorage.setItem("bn_webgpu_enabled", webgpuEnabled.toString());
      });
    }

    // Separator Precision Select
    const separatorPrecisionSelect = document.getElementById("separatorPrecisionSelect");
    if (separatorPrecisionSelect) {
      separatorPrecisionSelect.value = separatorPrecision;
      separatorPrecisionSelect.addEventListener("change", () => {
        separatorPrecision = separatorPrecisionSelect.value;
        localStorage.setItem("bn_separator_precision", separatorPrecision);
      });
    }

    // Advanced Range Sliders
    bindRange("aadRmsRange", aadRmsThreshold, (v) => {
      aadRmsThreshold = v;
    }, (v) => v.toFixed(3), "bn_aad_rms_threshold");

    bindRange("aadCentroidRange", aadCentroidMin, (v) => {
      aadCentroidMin = v;
    }, (v) => `${Math.round(v)} Hz`, "bn_aad_centroid_min");

    bindRange("earlyExitConfidenceRange", earlyExitConfidence * 100, (v) => {
      earlyExitConfidence = v / 100;
    }, (v) => `${Math.round(v)}%`, "bn_early_exit_confidence");

    bindRange("pipelineBWindowRange", pipelineBWindow, (v) => {
      pipelineBWindow = v;
    }, (v) => `${v.toFixed(1)}s`, "bn_pipeline_b_window");

    bindRange("pipelineBStrideRange", pipelineBStride, (v) => {
      pipelineBStride = v;
    }, (v) => `${v.toFixed(1)}s`, "bn_pipeline_b_stride");
  }

  // Initialize
  initSettingsControls();

  function bufferToWav(buffer, sampleRate) {
    const bufferLength = buffer.length;
    const wavHeader = new ArrayBuffer(44);
    const view = new DataView(wavHeader);

    function writeString(view, offset, string) {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    }

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + bufferLength * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, bufferLength * 2, true);

    const pcmBuffer = new Int16Array(bufferLength);
    for (let i = 0; i < bufferLength; i++) {
      const s = Math.max(-1, Math.min(1, buffer[i]));
      pcmBuffer[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }

    return new Blob([wavHeader, pcmBuffer], { type: 'audio/wav' });
  }

  function concatenateFloat32Arrays(arrays) {
    let totalLength = 0;
    for (const arr of arrays) {
      totalLength += arr.length;
    }
    const result = new Float32Array(totalLength);
    let offset = 0;
    for (const arr of arrays) {
      result.set(arr, offset);
      offset += arr.length;
    }
    return result;
  }

  function getBestIsolatedBufferForSpecies(scientificName) {
    if (isPipelineAOnly) {
      const verified = verifiedDetections.get(scientificName);
      return verified ? verified.audioBuffer : null;
    }

    let bestChannelId = -1;
    let maxConf = -Infinity;
    separatedChannelPredictions.forEach((predsMap, channelId) => {
      const pred = predsMap.get(scientificName);
      if (pred && pred.confidence > maxConf) {
        maxConf = pred.confidence;
        bestChannelId = channelId;
      }
    });

    if (bestChannelId === -1) {
      console.warn(`[getBestIsolatedBufferForSpecies] No channel found with predictions for: ${scientificName}. Falling back to default verified audioBuffer.`);
      const verified = verifiedDetections.get(scientificName);
      return verified ? verified.audioBuffer : null;
    }

    const arrays = separatedChannelBuffers.get(bestChannelId);
    if (!arrays || arrays.length === 0) {
      console.warn(`[getBestIsolatedBufferForSpecies] No buffers found for best channel: ${bestChannelId}`);
      return null;
    }

    console.log(`[getBestIsolatedBufferForSpecies] Concatenating ${arrays.length} segments for channel ${bestChannelId} (max confidence: ${(maxConf * 100).toFixed(1)}%)`);
    return concatenateFloat32Arrays(arrays);
  }

  function downloadUploadIsolatedAudio(scientificName) {
    const verified = verifiedDetections.get(scientificName);
    if (!verified) {
      console.warn("[downloadUploadIsolatedAudio] No verified entry found for", scientificName);
      return;
    }
    
    const samples = getBestIsolatedBufferForSpecies(scientificName);
    if (!samples) {
      console.warn("[downloadUploadIsolatedAudio] No audio buffer found for", scientificName);
      return;
    }
    
    let maxVal = 0;
    for (let i = 0; i < samples.length; i++) {
      const abs = Math.abs(samples[i]);
      if (abs > maxVal) maxVal = abs;
    }

    const normalized = new Float32Array(samples.length);
    if (maxVal > 0.0001) {
      const gain = 0.8 / maxVal;
      const clampedGain = Math.min(20.0, gain);
      for (let i = 0; i < samples.length; i++) {
        normalized[i] = samples[i] * clampedGain;
      }
    } else {
      normalized.set(samples);
    }

    const sampleRate = playbackAudioContext ? playbackAudioContext.sampleRate : 48000;
    const wavBlob = bufferToWav(normalized, sampleRate);
    const url = URL.createObjectURL(wavBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${scientificName.replace(/\s+/g, '_')}_isolated.wav`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // Bind functions to window context for onclick handlers
  window.playUploadIsolatedAudio = playUploadIsolatedAudio;
  window.stopUploadIsolatedAudio = stopUploadIsolatedAudio;
  window.downloadUploadIsolatedAudio = downloadUploadIsolatedAudio;

  // Cache to avoid querying Wikipedia multiple times for the same species in the same session
  const speciesImageCache = new Map();

  /**
   * Dynamically fetches a species image from Wikipedia API using its scientific name.
   * If found, sets the src of the target image element.
   * If it fails, falls back to the default dummy image.
   */
  async function loadSpeciesImage(scientificName, imgElement, fallbackPath = 'img/dummy.webp') {
    if (!scientificName) {
      imgElement.src = fallbackPath;
      return;
    }

    // Check cache first
    if (speciesImageCache.has(scientificName)) {
      const cachedUrl = speciesImageCache.get(scientificName);
      imgElement.src = cachedUrl || fallbackPath;
      return;
    }

    try {
      // Wikipedia API call to get page image by title (supporting redirects, e.g. scientific name to common name)
      const url = `https://en.wikipedia.org/w/api.php?action=query&format=json&origin=*&prop=pageimages&titles=${encodeURIComponent(scientificName)}&pithumbsize=250&redirects=1&formatversion=2`;
      
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      const page = data.query?.pages?.[0];
      
      if (page && page.thumbnail && page.thumbnail.source) {
        const imgUrl = page.thumbnail.source;
        speciesImageCache.set(scientificName, imgUrl);
        imgElement.src = imgUrl;
      } else {
        // Try search if direct page title query failed to find page image
        const searchUrl = `https://en.wikipedia.org/w/api.php?action=query&format=json&origin=*&generator=search&gsrsearch=${encodeURIComponent(scientificName)}&gsrlimit=1&prop=pageimages&pithumbsize=250&formatversion=2`;
        const searchResponse = await fetch(searchUrl);
        if (searchResponse.ok) {
          const searchData = await searchResponse.json();
          const searchPage = searchData.query?.pages?.[0];
          if (searchPage && searchPage.thumbnail && searchPage.thumbnail.source) {
            const imgUrl = searchPage.thumbnail.source;
            speciesImageCache.set(scientificName, imgUrl);
            imgElement.src = imgUrl;
            return;
          }
        }
        
        speciesImageCache.set(scientificName, null);
        imgElement.src = fallbackPath;
      }
    } catch (error) {
      console.warn(`Failed to fetch Wikipedia image for ${scientificName}:`, error);
      imgElement.src = fallbackPath;
    }
  }

  // Cache to avoid querying Wikidata multiple times for the same eBird species code
  const ebirdCodeCache = new Map();

  /**
   * Dynamically resolves the eBird species code via Wikidata SPARQL.
   * Upgrades the link from the Google fallback to the direct ebird.org species profile once resolved.
   */
  async function loadEbirdLink(scientificName, anchorElement) {
    if (!scientificName) return;

    // Check cache first
    if (ebbirdCodeCache.has(scientificName)) {
      const cachedCode = ebirdCodeCache.get(scientificName);
      if (cachedCode) {
        anchorElement.href = `https://ebird.org/species/${cachedCode}`;
      }
      return;
    }

    try {
      const endpoint = "https://query.wikidata.org/sparql";
      const query = `SELECT ?ebirdCode WHERE { ?item wdt:P225 "${scientificName}". ?item wdt:P3425 ?ebirdCode. } LIMIT 1`;
      const url = `${endpoint}?query=${encodeURIComponent(query)}&format=json`;
      
      const response = await fetch(url, {
        headers: {
          "Accept": "application/sparql-results+json"
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        const code = data.results?.bindings?.[0]?.ebirdCode?.value;
        if (code) {
          ebirdCodeCache.set(scientificName, code);
          anchorElement.href = `https://ebird.org/species/${code}`;
        } else {
          ebirdCodeCache.set(scientificName, null);
        }
      }
    } catch (error) {
      console.warn(`Failed to fetch eBird code for ${scientificName} via Wikidata:`, error);
    }
  }

})();
