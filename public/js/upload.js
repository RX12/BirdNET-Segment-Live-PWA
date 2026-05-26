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
  
  let isScanning = false;
  let scanIntervalId = null;
  let scanTimeSec = 0;           // Simulated time in seconds
  let scanDurationSec = 0;
  let simulatedEpochStart = 1000000000000; // Base epoch for simulation

  // Consensus state maps
  let verifiedDetections = new Map(); // scientificName -> { confidence, audioBuffer, timestamp }
  let pendingDetections = new Map();  // scientificName -> detectedAt (ms)
  let latestDetections = [];          // Displayed array of detections

  // Isolated Playback state
  let playbackAudioContext = null;
  let activeAudioSource = null;
  let currentlyPlayingSpecies = null;
  
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
  const scanStatusText = document.getElementById("scanStatusText");
  const scanProgressBar = document.getElementById("scanProgressBar");
  const scanProgressText = document.getElementById("scanProgressText");
  const uploadDetectionsList = document.getElementById("uploadDetectionsList");
  const debugLogConsole = document.getElementById("debugLogConsole");

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
        startScan();
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
    stopIsolatedAudio();

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

  async function startScan() {
    if (!audioBuffer48k || isScanning) return;
    
    stopPreview();
    stopIsolatedAudio();

    // Reset maps
    verifiedDetections.clear();
    pendingDetections.clear();
    latestDetections = [];
    renderDetections();

    isScanning = true;
    scanBtn.textContent = "Stop Scan";
    scanBtn.className = "btn btn-sm btn-danger";
    scanStatusText.textContent = "Spawning Web Workers...";
    logConsole("System", "Starting offline consensus scan...", "info");

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

    // Spawn Workers
    const tfPath = prefix + "js/tfjs-4.14.0.min.js";
    const root = prefix + "models";

    let sepParam = selectedSeparatorModel;
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

    const params = new URLSearchParams({ 
      tf: tfPath, 
      root, 
      lang: currentLabelLang, 
      prefix,
      precision: separatorPrecision,
      runOnGPU: webgpuEnabled ? "true" : "false",
      outputSampleRate: modelSR.toString()
    });
    if (selectedSeparatorModel === "dsp") {
      params.set("separator", "dsp");
    } else if (selectedSeparatorModel === "custom") {
      params.set("separator", "custom");
    } else {
      params.set("separator", selectedSeparatorModel);
      params.set("modelPath", modelPath);
    }

    logConsole("System", "Loading Pipeline A (Live model) and Pipeline B (Secondary classifier)...", "info");

    liveWorker = new Worker(prefix + "js/live-worker.js?" + params.toString());
    segmentationWorker = new Worker(prefix + "js/segmentation-worker.js?" + params.toString());

    if (selectedSeparatorModel === "custom" && customModelBytes) {
      segmentationWorker.postMessage({
        type: "SET_MODEL_BYTES",
        payload: {
          modelBytes: customModelBytes.slice(0)
        }
      }, [customModelBytes.slice(0)]);
    }

    // Setup Worker Events
    liveWorker.onmessage = (e) => {
      const data = e.data || {};
      if (data.message === "pooled" && Array.isArray(data.pooled)) {
        // Register detections in pendingDetections mapping to the scanner timeline
        const scanTimestamp = simulatedEpochStart + scanTimeSec * 1000;
        let maxConf = 0;
        data.pooled.forEach(p => {
          if (p.confidence > maxConf) maxConf = p.confidence;
          if (p.confidence >= detectionThreshold && p.scientificName) {
            if (!verifiedDetections.has(p.scientificName) && !pendingDetections.has(p.scientificName)) {
              pendingDetections.set(p.scientificName, scanTimestamp);
              logConsole("Pipeline A", `Draft detected: ${p.commonName} (${(p.confidence * 100).toFixed(0)}%) at ${formatTime(scanTimeSec)}`, "warning");
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
      } else if (data.message === "loaded") {
        logConsole("Pipeline A", "Live Worker loaded and warmed up successfully.", "success");
      } else if (data.message === "worker_error") {
        logConsole("Pipeline A", `Error: ${data.error}`, "error");
      }
    };

    segmentationWorker.onmessage = (e) => {
      const { type, payload } = e.data || {};
      
      if (type === "PIPELINE_STATUS") {
        logConsole("Pipeline B", `Status update: ${payload.message}`, "warning");
      } else if (type === "SEGMENT_RESULT") {
        const { segmentId, timestamp, results, error } = payload || {};
        if (error) {
          logConsole("Pipeline B", `Error in segment ${segmentId}: ${error}`, "error");
          return;
        }

        const elapsedSec = (timestamp - simulatedEpochStart) / 1000;
        logConsole("Pipeline B", `Completed segment separation for timestamp ${formatTime(elapsedSec)}`, "info");

        if (results && results.length > 0) {
          let updated = false;
          const verifiedThisSegment = new Set();

          results.forEach(res => {
            if (res.predictions && res.predictions.length > 0) {
              res.predictions.forEach(pred => {
                if (pred.scientificName && pred.confidence >= detectionThreshold) {
                  logConsole("Consensus", `VERIFIED: ${pred.commonName} (${(pred.confidence * 100).toFixed(0)}%) on isolated channel ${res.channelId} at ${formatTime(elapsedSec)}`, "success");
                  
                  verifiedDetections.set(pred.scientificName, {
                    confidence: pred.confidence,
                    audioBuffer: res.audioBuffer, // Float32Array isolated channel audio
                    timestamp: timestamp
                  });
                  verifiedThisSegment.add(pred.scientificName);
                  pendingDetections.delete(pred.scientificName);
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
      }
    };

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
        completeScan();
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
          liveWorker.postMessage({
            message: "predict",
            pcmAudio: copy,
            overlapSec: 1.5,
            sensitivity: 1.0
          }, [copy.buffer]);
        }
      }

      // Pipeline B (Segmented): run dynamically based on window and stride
      if (scanTimeSec >= nextPipelineBTime - 0.001) {
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
          }

          maxLiveConfidenceInInterval = 0; // reset for next interval

          if (!bypassB) {
            // Transfer copy to worker
            const copy = new Float32Array(slice);
            const simulatedTimestamp = simulatedEpochStart + scanTimeSec * 1000;
            
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

  function stopScan() {
    if (scanIntervalId) {
      clearInterval(scanIntervalId);
      scanIntervalId = null;
    }
    isScanning = false;
    if (scanBtn) {
      scanBtn.textContent = "Start Scan";
      scanBtn.className = "btn btn-sm btn-success";
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
    // Reconcile remaining unverified elements
    pendingDetections.forEach((detectedAt, sciName) => {
      logConsole("Consensus", `CLEANUP: Removing unverified species: ${sciName}`, "error");
      latestDetections = latestDetections.filter(d => d.scientificName !== sciName);
    });
    pendingDetections.clear();
    renderDetections();
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
      const imgUrl = `https://birdnet.cornell.edu/api2/bird/${encodeURIComponent(scientificName)}.webp`;

      const cardPendingClass = isVerified ? "" : "card-pending";

      const badgeHtml = isVerified
        ? `<span class="badge bg-primary bg-opacity-10 text-primary border border-primary border-opacity-10 flex-shrink-0">${confPct}%</span>`
        : `<span class="badge bg-warning bg-opacity-10 text-warning border border-warning border-opacity-10 flex-shrink-0"><i class="bi bi-arrow-repeat spin-icon me-1"></i>${confPct}% (Analyzing)</span>`;

      const playBtnHtml = isVerified
        ? (currentlyPlayingSpecies === scientificName
            ? `<button class="btn btn-sm btn-outline-danger py-0 px-2 play-audio-btn" style="font-size: 0.75rem;" onclick="stopIsolatedAudio()">
                 <i class="bi bi-stop-fill me-1"></i>Stop
               </button>`
            : `<button class="btn btn-sm btn-outline-primary py-0 px-2 play-audio-btn" style="font-size: 0.75rem;" onclick="playIsolatedAudio('${scientificName.replace(/'/g, "\\'")}')">
                 <i class="bi bi-play-fill me-1"></i>Play Isolated
               </button>`
          )
        : `<button class="btn btn-sm btn-outline-secondary py-0 px-2 play-audio-btn" style="font-size: 0.75rem;" disabled>
             <i class="bi bi-hourglass me-1"></i>Analyzing
           </button>`;

      const cardCol = document.createElement("div");
      cardCol.className = "col-md-6 col-lg-12 fade-in";
      cardCol.innerHTML = `
        <div class="card border-0 shadow-sm overflow-hidden ${cardPendingClass}">
          <div class="d-flex h-100">
            <div class="flex-shrink-0 position-relative" style="width: 90px; background-color: #f8f9fa;">
              <img src="${imgUrl}" 
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
                ${scientificName ? `<div class="text-muted fst-italic small mb-2 text-truncate" style="font-size: 0.8rem;">${scientificName}</div>` : ""}
              </div>
              <div>
                <div class="d-flex justify-content-between align-items-center border-top pt-2 mt-1">
                  <span class="small text-muted" style="font-size: 0.75rem;">
                    Consensus engine verified track
                  </span>
                  <div class="play-btn-container">
                    ${playBtnHtml}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      `;
      uploadDetectionsList.appendChild(cardCol);
    });
  }

  /* ==========================================================================
     6. ORIGINAL PLAYBACK & ISOLATED PLAYBACK
     ========================================================================== */

  function startPreview() {
    if (!audioBuffer48k || isPlayingPreview) return;
    
    stopIsolatedAudio();

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
  function playIsolatedAudio(scientificName) {
    if (currentlyPlayingSpecies === scientificName) {
      stopIsolatedAudio();
      return;
    }

    stopPreview();

    const verified = verifiedDetections.get(scientificName);
    if (!verified || !verified.audioBuffer) return;

    try {
      stopIsolatedAudio();

      if (!playbackAudioContext) {
        playbackAudioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
      }
      if (playbackAudioContext.state === "suspended") {
        playbackAudioContext.resume();
      }

      const audioBuf = playbackAudioContext.createBuffer(1, verified.audioBuffer.length, SAMPLE_RATE);
      audioBuf.getChannelData(0).set(verified.audioBuffer);

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

  function stopIsolatedAudio() {
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

  // Bind functions to window context for onclick handlers
  window.playIsolatedAudio = playIsolatedAudio;
  window.stopIsolatedAudio = stopIsolatedAudio;

})();
