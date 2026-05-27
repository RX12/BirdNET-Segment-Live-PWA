/**
 * BirdNET Live - Main Application Script
 * Handles audio capture, TensorFlow.js worker communication,
 * spectrogram visualization, and UI rendering.
 */

/* ==========================================================================
   1. CONFIGURATION & CONSTANTS
   ========================================================================== */

// Audio settings
const SAMPLE_RATE = 48000;
const WINDOW_SECONDS = 3;
const WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS;

// Inference settings
const TEMPORAL_POOL_WINDOW = 5;    // Number of recent predictions to pool
const USE_TEMPORAL_POOL = true;    // Enable log-mean-exp pooling

// Spectrogram settings
const SPECTRO_FFT_SIZE = 2048;
const SPECTRO_DEFAULT_DURATION_SEC = 20;
const SPECTRO_DEFAULT_GAIN = 1.5;
const SPECTRO_SMOOTHING = 0.0; // 0.0 = crisp details, 1.0 = very smooth
const SPECTRO_MIN_FREQ_DEFAULT = 0;
const SPECTRO_MAX_FREQ_DEFAULT = 12000;

// Supported Languages
const LANG_LABELS = {
  en_us: "English (US)", en_uk: "English (UK)", de: "Deutsch", fr: "Français",
  es: "Español", it: "Italiano", nl: "Nederlands", pt: "Português",
  fi: "Suomi", sv: "Svenska", no: "Norsk", da: "Dansk", pl: "Polski",
  ru: "Русский", uk: "Українська", cs: "Čeština", sk: "Slovenčina",
  sl: "Slovenski", hu: "Magyar", ro: "Română", tr: "Türkçe",
  ar: "العربية", ja: "日本語", ko: "한국어", th: "ไทย", zh: "中文",
  af: "Afrikaans"
};
const SUPPORTED_LABEL_LANGS = Object.keys(LANG_LABELS);

// UI Translation State
let currentUiLang = "en";
let translations = {};

/* ==========================================================================
   2. GLOBAL STATE
   ========================================================================== */

// Audio & Worker State
let isListening = false;
let workerReady = false;
let liveWorker = null;
let segmentationWorker = null;
let audioRouter = null;
let audioContext;
let workletNode;
let gainNode;
let highPassFilterNode;
let currentStream;

// Inference State
let lastInferenceStart = 0;
let lastInferenceMs = null;
let recentInferenceSets = []; // Buffer for temporal pooling
let latestDetections = [];
let liveHistoryDetections = []; // Persistent chronological history of detections in live session
let pendingRejections = new Map(); // Keep track of deferred rejection timeout IDs
let verifiedDetections = new Map();
let pendingDetections = new Map();
let playbackAudioContext = null;
let activeAudioSource = null;
let currentlyPlayingSpecies = null;

// Spectrogram State
let spectroCanvas, spectroCtx;
let spectroAxisCanvas, spectroAxisCtx; // Overlay for axis
let spectroAnimationId = null;
let analyser;
let dataArray; // Float32 array for dB values
let bufferLength;
let spectroColumnSeconds = 0;
let lastSpectroColumnTime = 0;

// Geolocation State
let geolocation = null;
let geoWatchId = null;

// Caching
let lastSpeciesList = null; // Cache for explore page filtering

/* ==========================================================================
   3. UTILITIES & STORAGE
   ========================================================================== */

const store = {
  get: (k, def) => localStorage.getItem(k) ?? def,
  getFloat: (k, def) => { const v = localStorage.getItem(k); return v === null ? def : parseFloat(v); },
  getBool: (k, def) => { const v = localStorage.getItem(k); return v === null ? def : v === "true"; },
  set: (k, v) => localStorage.setItem(k, v)
};

/**
 * Maps browser locale string (e.g., "en-US") to BirdNET label code (e.g., "en_us").
 */
function mapBrowserLangToLabelLang(locale) {
  if (!locale) return "en_us";
  const l = locale.toLowerCase();
  if (SUPPORTED_LABEL_LANGS.includes(l)) return l;
  
  const base = l.split(/[-_]/)[0];
  switch (base) {
    case "en": return l.includes("gb") || l.includes("uk") ? "en_uk" : "en_us";
    case "de": return "de";
    case "fr": return "fr";
    case "es": return "es";
    case "it": return "it";
    case "nl": return "nl";
    case "pt": return "pt";
    case "fi": return "fi";
    case "sv": return "sv";
    case "no": return "no";
    case "da": return "da";
    case "pl": return "pl";
    case "ru": return "ru";
    case "uk": return "uk";
    case "cs": return "cs";
    case "sk": return "sk";
    case "sl": return "sl";
    case "hu": return "hu";
    case "ro": return "ro";
    case "tr": return "tr";
    case "ar": return "ar";
    case "ja": return "ja";
    case "ko": return "ko";
    case "th": return "th";
    case "zh": return "zh";
    case "af": return "af";
    default: return "en_us";
  }
}

/**
 * Loads UI translations for the specified language.
 */
async function loadTranslations(lang) {
  try {
    const prefix = window.PATH_PREFIX || "/";
    const response = await fetch(`${prefix}locales/${lang}.json`);
    if (!response.ok) throw new Error(`Failed to load ${lang} translations`);
    translations = await response.json();
    currentUiLang = lang;
    store.set("bn_ui_lang", lang);
    updateUIText();
    
    // Re-render dynamic lists to apply new translations
    renderDetections();
    if (document.getElementById("exploreList")) renderExploreList();

    // Update selector if it exists
    const selector = document.getElementById("uiLangSelect");
    if (selector) selector.value = lang;
    
  } catch (e) {
    console.error("Translation load error:", e);
    // Fallback to English if not already English
    if (lang !== "en") loadTranslations("en");
  }
}

/**
 * Updates all elements with data-i18n attribute.
 */
function updateUIText() {
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.getAttribute("data-i18n");
    if (translations[key]) {
      el.innerHTML = translations[key];
    }
  });
  
  // Update status text if not currently recording/processing
  // (Dynamic status updates will use the new translation map)
  if (!isListening && !workerReady) {
    updateStatus("status_init");
  }
}

/**
 * Helper to get a translated string.
 */
function t(key, ...args) {
  let str = translations[key];
  if (!str) return "";
  args.forEach((arg, i) => {
    str = str.replace(`{${i}}`, arg);
  });
  return str;
}

/**
 * Updates the main status text with a translated string.
 */
function updateStatus(key, ...args) {
  const el = statusEl();
  if (el) {
    el.textContent = t(key, ...args);
    // Only set data-i18n if no args, to avoid overwriting dynamic text with static template
    if (args.length === 0) {
      el.setAttribute("data-i18n", key);
    } else {
      el.removeAttribute("data-i18n");
    }
  }
}

/* ==========================================================================
   4. USER SETTINGS (LOADED FROM STORAGE)
   ========================================================================== */

// Spectrogram
let spectroMinFreq = store.getFloat("bn_spec_min_freq", SPECTRO_MIN_FREQ_DEFAULT);
let spectroMaxFreq = store.getFloat("bn_spec_max_freq", SPECTRO_MAX_FREQ_DEFAULT);
let spectroMinDb = store.getFloat("bn_spec_min_db", -120);
let spectroMaxDb = store.getFloat("bn_spec_max_db", -40);
let spectroDurationSec = store.getFloat("bn_spec_duration", SPECTRO_DEFAULT_DURATION_SEC);
let spectroGain = store.getFloat("bn_spec_gain", SPECTRO_DEFAULT_GAIN);
let spectroAxisTicks = store.getFloat("bn_spec_axis_ticks", 9); 
let colormapName = store.get("bn_colormap", "viridis");
let colormapFn = d3.interpolateViridis; // Updated in init

// Model & Detection
let currentLabelLang = store.get("bn_lang", mapBrowserLangToLabelLang(navigator.language));
let geoEnabled = store.getBool("bn_geo_enabled", true);
let detectionThreshold = store.getFloat("bn_threshold", 0.15);
if (detectionThreshold > 1.0) detectionThreshold = 0.15; // Sanity check
let inputGain = store.getFloat("bn_input_gain", 1.0);
let sensitivity = store.getFloat("bn_sensitivity", 1.0); // New: Sensitivity
let inferenceInterval = store.getFloat("bn_inference_interval", 500);
let rumbleFilterFreq = store.getFloat("bn_rumble_freq", 200);
let geoThreshold = store.getFloat("bn_geo_threshold", 0.05);

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

// Separator Model State
let separatorModelCatalog = []; // Loaded from models.json
let selectedSeparatorModel = store.get("bn_separator_model", "bird_mixit_4source");
let customModelBytes = null;
let customModelName = store.get("bn_custom_model_name", "");

/* ==========================================================================
   5. DOM ACCESSORS
   ========================================================================== */

const statusEl          = () => document.getElementById("statusText");
const recordButtonEl    = () => document.getElementById("recordButton");
const recordLabelTextEl = () => document.querySelector(".record-label-text");
const detectionsList    = () => document.getElementById("detectionsList");
const geoStatusEl       = () => document.getElementById("geoStatusText");
const geoCoordsEl       = () => document.getElementById("geoCoordsText");
const settingsToggleEl  = () => document.getElementById("settingsToggle");
const settingsDrawerEl  = () => document.getElementById("settingsDrawer");
const settingsOverlayEl = () => document.getElementById("settingsOverlay");

/* ==========================================================================
   6. INITIALIZATION (BOOT)
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  // Check WebGPU availability for Pipeline B
  if (!navigator.gpu) {
    console.warn("[App] WebGPU is not supported. Pipeline B will run on CPU WebAssembly fallback.");
    const warningEl = document.getElementById("webgpuWarning");
    if (warningEl) warningEl.classList.remove("d-none");
  }

  updateColormap(colormapName);

  // Initialize Language
  const savedLang = store.get("bn_ui_lang");
  const browserLang = navigator.language.split("-")[0];
  const initialLang = savedLang || (["de", "en"].includes(browserLang) ? browserLang : "en");
  loadTranslations(initialLang);
  
  const isLive = !!document.getElementById("recordButton");
  const isExplore = !!document.getElementById("exploreList");

  // Only run if we are on Live or Explore pages
  if (!isLive && !isExplore) return;

  setupSettingsToggle();
  initUIControls();

  // Load separator models catalog
  const prefix = (window.PATH_PREFIX || "/");
  fetch(prefix + "models/models.json")
    .then(r => r.json())
    .then(catalog => {
      separatorModelCatalog = catalog;
      initWorker();
      populateSeparatorDropdowns();
    })
    .catch(err => {
      console.warn("[App] Failed to load separator models catalog models.json, using fallback presets.", err);
      separatorModelCatalog = [
        { "id": "bird_mixit_4source", "name": "Bird-MixIT 4-Source (ONNX)", "path": "models/bird_mixit_4source.onnx" }
      ];
      initWorker();
      populateSeparatorDropdowns();
    });

  if (isLive) {
    setupRecordButton();
    initSpectrogramCanvas(); // Initialize canvas size immediately to prevent layout shift
  }

  // Handle Geolocation Initialization
  if (geoEnabled) {
    getGeolocation();
  } else {
    updateGeoDisplay("status_geo_disabled", null);
    if (isExplore) {
      // Show hint instead of full list if geo is off
      const container = document.getElementById("exploreList");
      if (container) {
        container.innerHTML = `
          <div class="col-12 text-center py-5 text-muted">
            <i class="bi bi-geo-alt-slash fs-1 d-block mb-3 opacity-25"></i>
            <p data-i18n="msg_explore_geo_disabled">${t("msg_explore_geo_disabled")}</p>
          </div>
        `;
      }
    }
  }
});

/* ==========================================================================
   7. WORKER & MODEL LOGIC
   ========================================================================== */

function initWorker(langOverride) {
  if (liveWorker) {
    try { liveWorker.terminate(); } catch (_) {}
    liveWorker = null;
    workerReady = false;
  }
  if (segmentationWorker) {
    try { segmentationWorker.terminate(); } catch (_) {}
    segmentationWorker = null;
  }
  if (audioRouter) {
    audioRouter.stop();
    audioRouter = null;
  }
  
  // Disable record button while reloading
  const btn = recordButtonEl();
  if (btn) btn.disabled = true;
  
  const prefix = (window.PATH_PREFIX || "/");
  const lang   = langOverride || currentLabelLang || (navigator.language || "en-US");

  // Determine separator parameters
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

  const status = statusEl();
  if (status) updateStatus("status_loading_percent", 0);
  
  liveWorker = new Worker(prefix + "js/live-worker.js");
  segmentationWorker = new Worker(prefix + "js/segmentation-worker.js");

  // Send initialization parameters via message passing
  liveWorker.postMessage({
    message: 'init',
    lang: lang
  });

  segmentationWorker.postMessage({
    type: 'INIT',
    payload: {
      separator: selectedSeparatorModel,
      modelPath: modelPath,
      runOnGPU: webgpuEnabled,
      outputSampleRate: modelSR,
      lang: lang,
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

  liveWorker.onmessage = (event) => {
    const data = event.data || {};
    switch (data.message) {
      case "load_model":
      case "warmup":
      case "load_geomodel":
      case "load_labels":
        if (typeof data.progress === "number") {
          updateStatus("status_loading_percent", data.progress);
        }
        break;
        
      case "labels_loaded":
        updateStatus("status_ready");
        requestSpeciesList();
        break;

      case "loaded":
        workerReady = true;
        updateStatus("status_ready");
        
        // Enable record button
        const btn = recordButtonEl();
        if (btn) btn.disabled = false;

        if (geolocation) sendAreaScores();
        // If on explore page, request list immediately after load
        if (document.getElementById("exploreList")) requestSpeciesList();
        break;

      case "pooled":
        // Handle inference results
        if (Array.isArray(data.pooled)) {
          recentInferenceSets.push(data.pooled);
          if (recentInferenceSets.length > TEMPORAL_POOL_WINDOW) {
            recentInferenceSets.shift();
          }
        }
        const toRender = USE_TEMPORAL_POOL
          ? computeTemporalPooledDetections(recentInferenceSets)
          : data.pooled;

        if (Array.isArray(toRender)) {
          let maxConf = 0;
          const now = Date.now();
          toRender.forEach(p => {
            if (p.confidence > maxConf) {
              maxConf = p.confidence;
            }
            if (p.confidence >= detectionThreshold && p.scientificName) {
              // Cancel any pending rejection timeout
              if (pendingRejections.has(p.scientificName)) {
                clearTimeout(pendingRejections.get(p.scientificName));
                pendingRejections.delete(p.scientificName);
              }

              if (!verifiedDetections.has(p.scientificName)) {
                if (!pendingDetections.has(p.scientificName)) {
                  pendingDetections.set(p.scientificName, now);
                }

                // Add or update liveHistoryDetections
                const existing = liveHistoryDetections.find(d => d.scientificName === p.scientificName);
                if (!existing) {
                  liveHistoryDetections.unshift({
                    scientificName: p.scientificName,
                    commonName: p.commonName,
                    commonNameI18n: p.commonNameI18n || p.commonName,
                    confidence: p.confidence,
                    status: "analyzing",
                    timestamp: now,
                    lastSeen: now
                  });
                } else {
                  existing.lastSeen = now;
                  if (existing.status === "analyzing") {
                    existing.confidence = Math.max(existing.confidence, p.confidence);
                  }
                }
              } else {
                // Already verified, update lastSeen
                const existing = liveHistoryDetections.find(d => d.scientificName === p.scientificName);
                if (existing) {
                  existing.lastSeen = now;
                }
              }
            }
          });
          if (audioRouter) {
            audioRouter.reportLiveConfidence(maxConf);
          }
        }

        renderDetections();
        
        if (isListening && lastInferenceStart) {
          lastInferenceMs = Math.round(performance.now() - lastInferenceStart);
          updateStatus("status_listening_inference", lastInferenceMs);
        }
        break;

      case "area-scores":
        // Geo priors updated, refresh explore list if visible
        if (document.getElementById("exploreList")) requestSpeciesList();
        break;

      case "species_list":
        renderExploreList(data.list);
        break;
    }
  };

  liveWorker.onerror = (err) => {
    console.error("Live Worker error", err);
    updateStatus("status_worker_error");
  };

  segmentationWorker.onmessage = (event) => {
    const { type, payload } = event.data || {};

    // Release Pipeline B backpressure on any result/status message
    if (audioRouter && (type === "SEGMENT_RESULT" || type === "PIPELINE_STATUS")) {
      audioRouter.markSegmentComplete();
    }

    if (type === "PIPELINE_STATUS") {
      // WebGPU downgrade notification from segmentation worker
      const { status, message } = payload || {};
      console.log(`[App] Pipeline B status: ${status} — ${message}`);
      if (status === "webgpu_fallback" || status === "error") {
        const warningEl = document.getElementById("webgpuWarning");
        if (warningEl) warningEl.classList.remove("d-none");
      }
      return;
    }

    if (type === "SEGMENT_RESULT") {
      const { segmentId, timestamp, results, error } = payload || {};
      if (error) {
        console.error(`[App] Error in Pipeline B for segment ${segmentId}:`, error);
        return;
      }
      console.log(`[App] Pipeline B result for segment ${segmentId}:`, results);
      if (results && results.length > 0) {
        let updated = false;
        const verifiedThisSegment = new Set();

        results.forEach(res => {
          console.log(`[Consensus] Channel ${res.channelId} predictions:`, res.predictions);
          if (res.predictions && res.predictions.length > 0) {
            res.predictions.forEach(pred => {
              if (pred.scientificName && pred.confidence >= detectionThreshold) {
                console.log(`[Consensus] Verifying species: ${pred.scientificName} with confidence ${pred.confidence}`);
                console.log(`[Consensus] Storing verified detection: ${pred.scientificName}, audioBuffer:`, res.audioBuffer);
                verifiedDetections.set(pred.scientificName, {
                  confidence: pred.confidence,
                  audioBuffer: res.audioBuffer, // Float32Array isolated channel audio
                  timestamp: timestamp
                });
                verifiedThisSegment.add(pred.scientificName);
                pendingDetections.delete(pred.scientificName);

                // Cancel any pending rejection timeout
                if (pendingRejections.has(pred.scientificName)) {
                  clearTimeout(pendingRejections.get(pred.scientificName));
                  pendingRejections.delete(pred.scientificName);
                }

                // Append verified species to liveHistoryDetections so it appears in the UI
                const existing = liveHistoryDetections.find(d => d.scientificName === pred.scientificName);
                if (!existing) {
                  liveHistoryDetections.unshift({
                    scientificName: pred.scientificName,
                    commonName: pred.commonName,
                    commonNameI18n: pred.commonNameI18n || pred.commonName,
                    confidence: pred.confidence,
                    status: "verified",
                    timestamp: timestamp,
                    lastSeen: timestamp
                  });
                } else {
                  existing.status = "verified";
                  existing.confidence = Math.max(existing.confidence, pred.confidence);
                  existing.timestamp = timestamp;
                }
                updated = true;
              }
            });
          }
        });

        // Timeline reconciliation: unmount unverified detections (with a 2-second grace period)
        const segmentStart = timestamp - (pipelineBWindow * 1000);
        const segmentEnd = timestamp;

        pendingDetections.forEach((detectedAt, sciName) => {
          if (detectedAt >= segmentStart && detectedAt <= segmentEnd) {
            if (!verifiedThisSegment.has(sciName)) {
              if (!pendingRejections.has(sciName)) {
                console.log(`[Consensus] Rejecting false positive after grace period: ${sciName}`);
                const timeoutId = setTimeout(() => {
                  pendingRejections.delete(sciName);
                  pendingDetections.delete(sciName);
                  // Remove from liveHistoryDetections if still analyzing
                  const idx = liveHistoryDetections.findIndex(d => d.scientificName === sciName);
                  if (idx !== -1 && liveHistoryDetections[idx].status === "analyzing") {
                    liveHistoryDetections.splice(idx, 1);
                    renderDetections();
                  }
                }, 2000); // 2-second grace period
                pendingRejections.set(sciName, timeoutId);
              }
            }
          }
        });

        if (updated) {
          renderDetections();
        }
      }
    }
  };

  segmentationWorker.onerror = (err) => {
    console.error("Segmentation Worker error", err);
  };
}

function requestSpeciesList() {
  if (liveWorker) {
    liveWorker.postMessage({ message: "get_species_list" });
  }
}

/* ==========================================================================
   8. AUDIO ENGINE & INFERENCE LOOP
   ========================================================================== */

function setupRecordButton() {
  const btn = recordButtonEl();
  if (!btn) return;
  
  // Initially disable until model loads
  btn.disabled = true;
  
  btn.addEventListener("click", async () => {
    if (!isListening) {
      await startListening();
    } else {
      stopListening();
    }
  });
}

async function startListening() {
  if (!workerReady) {
    updateStatus("status_loading");
    return;
  }
  try {
    isListening = true;
    
    // UI Updates
    const button = recordButtonEl();
    if (button) button.classList.add("recording");
    const label = recordLabelTextEl();
    if (label) {
      label.textContent = t("btn_stop");
      label.setAttribute("data-i18n", "btn_stop");
    }
    const spinner = document.getElementById("listeningIndicator");
    if (spinner) spinner.classList.remove("d-none");
    
    updateStatus("status_requesting_mic");
    await requestWakeLock();

    // MOBILE FIX: Create and resume AudioContext synchronously inside the
    // user-gesture call stack, BEFORE the async getUserMedia() consent dialog.
    // iOS Safari breaks the gesture chain after getUserMedia resolves, which
    // causes AudioContext to remain permanently suspended.
    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
    if (audioContext.state === "suspended") {
      await audioContext.resume();
    }

    currentStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: SAMPLE_RATE,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false
      }
    });

    await setupAudioGraphFromStream(currentStream, audioContext);
    updateStatus("status_listening");
  } catch (e) {
    console.error(e);
    updateStatus("status_mic_failed");
    stopListening(); // Cleanup UI state
  }
}

function stopListening() {
  isListening = false;
  releaseWakeLock();
  
  // UI Updates
  const button = recordButtonEl();
  if (button) button.classList.remove("recording");
  const label = recordLabelTextEl();
  if (label) {
    label.textContent = t("btn_start");
    label.setAttribute("data-i18n", "btn_start");
  }
  const spinner = document.getElementById("listeningIndicator");
  if (spinner) spinner.classList.add("d-none");

  updateStatus("status_stopped");

  // Reset State
  lastInferenceStart = 0;
  lastInferenceMs = null;
  verifiedDetections.clear();
  pendingDetections.clear();
  liveHistoryDetections = [];
  pendingRejections.forEach(tId => clearTimeout(tId));
  pendingRejections.clear();
  stopIsolatedAudio();

  // Cleanup Audio
  if (currentStream) {
    currentStream.getTracks().forEach(t => t.stop());
    currentStream = null;
  }
  if (workletNode) {
    workletNode.port.onmessage = null;
    workletNode.disconnect();
    workletNode = null;
  }
  if (gainNode) {
    gainNode.disconnect();
    gainNode = null;
  }
  if (highPassFilterNode) {
    highPassFilterNode.disconnect();
    highPassFilterNode = null;
  }
  if (audioRouter) {
    audioRouter.stop();
    audioRouter = null;
  }
  if (audioContext) {
    stopSpectrogram();
    audioContext.close();
    audioContext = null;
  }
}

async function setupAudioGraphFromStream(stream, ctx) {
  // Use the pre-created AudioContext from the gesture handler.
  // This ensures iOS Safari treats it as user-gesture-initiated.
  audioContext = ctx;

  // Double-check resume (belt-and-suspenders for edge cases)
  if (audioContext.state === "suspended") {
    await audioContext.resume();
  }
  const source = audioContext.createMediaStreamSource(stream);
  
  // Create Gain Node (Hardware-like gain)
  gainNode = audioContext.createGain();
  gainNode.gain.value = inputGain;
  
  // Create High-Pass Filter (Rumble Filter)
  highPassFilterNode = audioContext.createBiquadFilter();
  highPassFilterNode.type = "highpass";
  highPassFilterNode.frequency.value = rumbleFilterFreq;
  highPassFilterNode.Q.value = 0.707; // Butterworth

  // Connect Mic -> Gain -> HighPass
  source.connect(gainNode);
  gainNode.connect(highPassFilterNode);

  // Start visualizer (Connect HighPass -> Spectrogram)
  startSpectrogram(highPassFilterNode);

  if (!audioContext.audioWorklet) {
    updateStatus("status_browser_old");
    return;
  }

  // Setup AudioRouter (replacing manual RingBuffer & inference loop)
  audioRouter = new AudioRouter(liveWorker, segmentationWorker, {
    sampleRate: SAMPLE_RATE,
    getSensitivity: () => sensitivity,
    getGeoContext: () => {
      return geolocation ? {
        latitude: geolocation.lat,
        longitude: geolocation.lon
      } : {};
    },
    onInferenceStart: () => {
      lastInferenceStart = performance.now();
    },
    onEarlyExit: (pcm) => {
      // Verify all pending drafts in the current segment window using the raw audio slice
      const now = Date.now();
      const windowMs = pipelineBWindow * 1000;
      const segmentStart = now - windowMs;
      const segmentEnd = now;
      let updated = false;

      pendingDetections.forEach((detectedAt, sciName) => {
        if (detectedAt >= segmentStart && detectedAt <= segmentEnd) {
          console.log(`[Consensus] Verifying via Early Exit Fallback: ${sciName}`);
          
          // Cancel any pending rejection timeout
          if (pendingRejections.has(sciName)) {
            clearTimeout(pendingRejections.get(sciName));
            pendingRejections.delete(sciName);
          }

          // We try to find common name in history or default it
          let cName = sciName;
          let cNameI18n = sciName;
          const histMatch = liveHistoryDetections.find(d => d.scientificName === sciName);
          if (histMatch) {
            cName = histMatch.commonName;
            cNameI18n = histMatch.commonNameI18n;
          }

          const conf = histMatch ? histMatch.confidence : earlyExitConfidence;
          
          verifiedDetections.set(sciName, {
            confidence: conf,
            audioBuffer: new Float32Array(pcm),
            timestamp: now
          });

          // Update history entry to verified
          if (histMatch) {
            histMatch.status = "verified";
            histMatch.confidence = Math.max(histMatch.confidence, conf);
            histMatch.timestamp = now;
          } else {
            liveHistoryDetections.unshift({
              scientificName: sciName,
              commonName: cName,
              commonNameI18n: cNameI18n,
              confidence: conf,
              status: "verified",
              timestamp: now,
              lastSeen: now
            });
          }

          pendingDetections.delete(sciName);
          updated = true;
        }
      });

      if (updated) {
        renderDetections();
      }
    },
    getBConfig: () => ({
      windowSize: pipelineBWindow,
      stride: pipelineBStride,
      gateEnabled: aadEnabled,
      rmsThreshold: aadRmsThreshold,
      centroidMin: aadCentroidMin,
      earlyExitEnabled: earlyExitEnabled,
      earlyExitConfidence: earlyExitConfidence
    })
  });

  // Use AudioWorklet for raw audio access (Replaces ScriptProcessor)
  try {
    const prefix = (window.PATH_PREFIX || "/");
    await audioContext.audioWorklet.addModule(prefix + "js/audio-processor.js");

    workletNode = new AudioWorkletNode(audioContext, "audio-processor");

    // Handle audio data from the worklet
    workletNode.port.onmessage = (event) => {
      if (audioRouter) {
        audioRouter.inputData(event.data);
      }
    };

    // Connect Gain -> Worklet -> Destination
    highPassFilterNode.connect(workletNode);
    workletNode.connect(audioContext.destination);

  } catch (err) {
    console.error("Failed to load AudioWorklet", err);
    updateStatus("status_audio_failed");
    return;
  }

  audioRouter.start();
}

/* ==========================================================================
   9. SPECTROGRAM VISUALIZATION
   ========================================================================== */

function initSpectrogramCanvas() {
  if (spectroCanvas) return;
  spectroCanvas = document.getElementById("liveSpectrogram");
  if (!spectroCanvas) return;

  // Create Axis Overlay if it doesn't exist
  if (!spectroAxisCanvas) {
    const parent = spectroCanvas.parentElement;
    if (parent) {
      parent.style.position = "relative"; // Ensure positioning context
      spectroAxisCanvas = document.createElement("canvas");
      spectroAxisCanvas.className = "spectro-axis-overlay";
      spectroAxisCanvas.style.position = "absolute";
      spectroAxisCanvas.style.top = "0";
      spectroAxisCanvas.style.left = "0";
      spectroAxisCanvas.style.pointerEvents = "none"; // Let clicks pass through
      spectroAxisCanvas.style.zIndex = "10"; // Above spectrogram
      parent.appendChild(spectroAxisCanvas);
      spectroAxisCtx = spectroAxisCanvas.getContext("2d");
    }
  }
  
  resizeSpectrogramCanvas();
  window.addEventListener("resize", resizeSpectrogramCanvas);
}

function resizeSpectrogramCanvas() {
  if (!spectroCanvas) return;
  const cssW = spectroCanvas.clientWidth || 600;
  const cssH = spectroCanvas.clientHeight || 220;

  // Preserve existing content if possible
  let snapshot = null;
  if (spectroCtx) {
    try {
      snapshot = spectroCtx.getImageData(0, 0, spectroCanvas.width, spectroCanvas.height);
    } catch (_) {}
  }

  spectroCanvas.width = cssW;
  spectroCanvas.height = cssH;

  // Optimize for frequent readback (resizing)
  spectroCtx = spectroCanvas.getContext("2d", { willReadFrequently: true });
  spectroCtx.fillStyle = "#000";
  spectroCtx.fillRect(0, 0, cssW, cssH);

  if (snapshot) spectroCtx.putImageData(snapshot, 0, 0);

  // Resize Axis Overlay (High DPI support)
  if (spectroAxisCanvas) {
    const dpr = window.devicePixelRatio || 1;
    // Set physical size based on DPR
    spectroAxisCanvas.width = Math.floor(cssW * dpr);
    spectroAxisCanvas.height = Math.floor(cssH * dpr);
    
    // Set CSS size to match layout
    spectroAxisCanvas.style.width = `${cssW}px`;
    spectroAxisCanvas.style.height = `${cssH}px`;
    
    // Scale context so drawing operations use CSS pixels
    spectroAxisCtx.scale(dpr, dpr);
    
    drawSpectrogramAxis();
  }

  spectroColumnSeconds = cssW > 0 ? spectroDurationSec / cssW : 0.05;
  lastSpectroColumnTime = audioContext ? audioContext.currentTime : 0;
}

function drawSpectrogramAxis() {
  if (!spectroAxisCtx || !spectroAxisCanvas) return;
  const ctx = spectroAxisCtx;
  // Use logical CSS dimensions for drawing
  const w = spectroAxisCanvas.clientWidth;
  const h = spectroAxisCanvas.clientHeight;

  ctx.clearRect(0, 0, w, h);
  
  // Background strip for legibility
  ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
  ctx.fillRect(0, 0, 40, h);

  ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
  ctx.font = "10px system-ui, -apple-system, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";

  const numSteps = spectroAxisTicks;
  const range = spectroMaxFreq - spectroMinFreq;

  for (let i = 0; i <= numSteps; i++) {
    const ratio = i / numSteps;
    const freq = spectroMinFreq + (range * ratio);
    // Canvas Y is inverted (0 is top/high freq)
    let y = h - (ratio * h);
    
    // Aesthetic tweak: inset edge ticks so they aren't on the absolute pixel edge
    if (i === 0) y -= 5;          // Move bottom tick up
    if (i === numSteps) y += 5;   // Move top tick down
    
    // Adjust text position to avoid clipping at edges
    let textY = y;
    if (i === 0) textY -= 5;
    if (i === numSteps) textY += 5;

    ctx.fillText(`${(freq/1000).toFixed(1)}k`, 32, textY);
    
    // Tick mark
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.fillRect(34, y, 6, 1);
    ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
  }
}

function startSpectrogram(source) {
  initSpectrogramCanvas();
  if (!spectroCanvas) return;

  analyser = audioContext.createAnalyser();
  analyser.fftSize = SPECTRO_FFT_SIZE;
  analyser.smoothingTimeConstant = SPECTRO_SMOOTHING;
  source.connect(analyser);

  bufferLength = analyser.frequencyBinCount;
  dataArray = new Float32Array(bufferLength); // Use Float32 for dB

  lastSpectroColumnTime = audioContext.currentTime;
  if (!spectroColumnSeconds) {
    const w = spectroCanvas.width || 600;
    spectroColumnSeconds = spectroDurationSec / w;
  }
  if (!spectroAnimationId) {
    spectroAnimationId = requestAnimationFrame(drawSpectrogram);
  }
}

function stopSpectrogram() {
  if (spectroAnimationId) {
    cancelAnimationFrame(spectroAnimationId);
    spectroAnimationId = null;
  }
  if (analyser) {
    try { analyser.disconnect(); } catch (_) {}
    analyser = null;
  }
}

function updateColormap(name) {
  switch (name) {
    case "inferno": colormapFn = d3.interpolateInferno; break;
    case "plasma": colormapFn = d3.interpolatePlasma; break;
    case "viridis": colormapFn = d3.interpolateViridis; break;
    case "turbo": colormapFn = d3.interpolateTurbo; break;
    case "cubehelix": colormapFn = d3.interpolateCubehelixDefault; break;
    default: colormapFn = d3.interpolateMagma; break;
  }
}

function drawSpectrogram() {
  spectroAnimationId = requestAnimationFrame(drawSpectrogram);
  if (!analyser || !audioContext) return;

  analyser.getFloatFrequencyData(dataArray);

  const w = spectroCanvas.width;
  const h = spectroCanvas.height;
  if (!w || !h) return;

  // Calculate scrolling
  if (!spectroColumnSeconds) {
    spectroColumnSeconds = spectroDurationSec / Math.max(1, w);
  }
  const audioNow = audioContext.currentTime;
  let columnsNeeded = Math.floor((audioNow - lastSpectroColumnTime) / spectroColumnSeconds);
  if (columnsNeeded <= 0) return;
  columnsNeeded = Math.min(columnsNeeded, w - 1);
  lastSpectroColumnTime += columnsNeeded * spectroColumnSeconds;

  // Shift canvas left
  spectroCtx.drawImage(
    spectroCanvas,
    columnsNeeded, 0, w - columnsNeeded, h,
    0, 0, w - columnsNeeded, h
  );

  // Frequency bin mapping
  const nyquist = SAMPLE_RATE / 2;
  const startBin = Math.max(0, Math.floor((spectroMinFreq / nyquist) * bufferLength));
  const endBin = Math.min(bufferLength - 1, Math.floor((spectroMaxFreq / nyquist) * bufferLength));
  const binCount = endBin - startBin + 1;

  // Draw new columns
  for (let c = 0; c < columnsNeeded; c++) {
    const x = w - columnsNeeded + c;
    
    // Clear column
    spectroCtx.fillStyle = "#000";
    spectroCtx.fillRect(x, 0, 1, h);

    // Draw frequency bins
    for (let i = startBin; i <= endBin; i++) {
      const db = dataArray[i];
      
      // Normalize dB to 0..1 range
      let norm = (db - spectroMinDb) / (spectroMaxDb - spectroMinDb);
      norm = Math.max(0, Math.min(1, norm));
      
      // Optional: slight gamma for contrast
      norm = Math.pow(norm, 0.8);

      // Map to Y pixels (flip Y so low freq is at bottom)
      // Calculate exact pixel boundaries for this bin to prevent gaps
      const relIndex = i - startBin;
      
      // yBottom is the lower edge of the bin (lower frequency, higher Y pixel value)
      const yBottom = h * (1 - relIndex / binCount);
      // yTop is the upper edge of the bin (higher frequency, lower Y pixel value)
      const yTop = h * (1 - (relIndex + 1) / binCount);
      
      // Snap to integer pixels to avoid sub-pixel rendering gaps
      const yDraw = Math.floor(yTop);
      const hDraw = Math.ceil(yBottom) - yDraw;

      spectroCtx.fillStyle = colormapFn(norm);
      spectroCtx.fillRect(x, yDraw, 1, hDraw);
    }
  }
}

/* ==========================================================================
   10. UI & RENDERING
   ========================================================================== */

function initUIControls() {
  // Geolocation Toggle
  const geoToggle = document.getElementById("geoToggle");
  if (geoToggle) {
    geoToggle.checked = geoEnabled;
    geoToggle.addEventListener("change", () => {
      geoEnabled = geoToggle.checked;
      store.set("bn_geo_enabled", geoEnabled);

      if (geoEnabled) {
        updateGeoDisplay("status_geo_requesting", null);
        getGeolocation();
      } else {
        geolocation = null;
        if (geoWatchId !== null) {
          navigator.geolocation.clearWatch(geoWatchId);
          geoWatchId = null;
        }
        updateGeoDisplay("status_geo_disabled", null);
        
        // Handle Explore page state change
        if (document.getElementById("exploreList")) {
           const container = document.getElementById("exploreList");
           container.innerHTML = `
            <div class="col-12 text-center py-5 text-muted">
              <i class="bi bi-geo-alt-slash fs-1 d-block mb-3 opacity-25"></i>
              <p data-i18n="msg_explore_geo_disabled">${t("msg_explore_geo_disabled")}</p>
            </div>
          `;
        } else {
          renderDetections();
        }
      }
    });
  }

  // Settings Sliders
  bindRange("geoThresholdRange", geoThreshold * 100, (v) => {
    geoThreshold = v / 100;
    store.set("bn_geo_threshold", geoThreshold);
    if (document.getElementById("exploreList")) {
      renderExploreList(null); // Re-render cached list
    }
  }, (v) => `${Math.round(v)}%`);

  bindRange("durationRange", spectroDurationSec, (v) => {
    spectroDurationSec = v;
    if (spectroCanvas && spectroCanvas.width > 0) {
      spectroColumnSeconds = spectroDurationSec / spectroCanvas.width;
    }
  }, (v) => `${v}s`, "bn_spec_duration");

  bindRange("gainRange", spectroGain, (v) => {
    spectroGain = v;
  }, (v) => `${v.toFixed(1)}×`, "bn_spec_gain");

  bindRange("thresholdRange", detectionThreshold * 100, (v) => {
    detectionThreshold = v / 100;
    store.set("bn_threshold", detectionThreshold);
    renderDetections();
  }, (v) => `${Math.round(v)}%`);

  bindRange("inputGainRange", inputGain, (v) => {
    inputGain = v;
    // Apply gain immediately if listening
    if (gainNode) gainNode.gain.value = v;
  }, (v) => `${v.toFixed(1)}×`, "bn_input_gain");

  bindRange("rumbleFilterRange", rumbleFilterFreq, (v) => {
    rumbleFilterFreq = v;
    if (highPassFilterNode) highPassFilterNode.frequency.value = v;
  }, (v) => `${Math.round(v)} Hz`, "bn_rumble_freq");

  bindRange("sensitivityRange", sensitivity, (v) => {
    sensitivity = v;
  }, (v) => v.toFixed(1), "bn_sensitivity");

  bindRange("inferenceIntervalRange", inferenceInterval, (v) => {
    inferenceInterval = v;
  }, (v) => `${Math.round(v)} ms`, "bn_inference_interval");

  bindRange("minFreqRange", spectroMinFreq, (v) => {
    spectroMinFreq = Math.min(v, spectroMaxFreq - 100);
    drawSpectrogramAxis();
  }, (v) => `${Math.round(v)} Hz`, "bn_spec_min_freq");

  bindRange("maxFreqRange", spectroMaxFreq, (v) => {
    spectroMaxFreq = Math.max(v, spectroMinFreq + 100);
    drawSpectrogramAxis();
  }, (v) => `${Math.round(v)} Hz`, "bn_spec_max_freq");

  bindRange("axisTicksRange", spectroAxisTicks, (v) => {
    spectroAxisTicks = v;
    drawSpectrogramAxis();
  }, (v) => `${v}`, "bn_spec_axis_ticks");

  bindRange("minDbRange", spectroMinDb, (v) => {
    spectroMinDb = Math.min(v, spectroMaxDb - 10);
  }, (v) => `${v} dB`, "bn_spec_min_db");

  bindRange("maxDbRange", spectroMaxDb, (v) => {
    spectroMaxDb = Math.max(v, spectroMinDb + 10);
  }, (v) => `${v} dB`, "bn_spec_max_db");

  // Dropdowns
  const colormapSelect = document.getElementById("colormapSelect");
  if (colormapSelect) {
    colormapSelect.value = colormapName;
    colormapSelect.addEventListener("change", () => {
      colormapName = colormapSelect.value;
      store.set("bn_colormap", colormapName);
      updateColormap(colormapName);
    });
  }

  const uiLangSelect = document.getElementById("uiLangSelect");
  if (uiLangSelect) {
    uiLangSelect.value = currentUiLang;
    uiLangSelect.addEventListener("change", () => {
      loadTranslations(uiLangSelect.value);
    });
  }

  const langSelect = document.getElementById("labelLangSelect");
  if (langSelect) {
    langSelect.innerHTML = SUPPORTED_LABEL_LANGS
      .map(code => {
        const label = LANG_LABELS[code] || code;
        const sel = code === currentLabelLang ? " selected" : "";
        return `<option value="${code}"${sel}>${label}</option>`;
      })
      .join("");
    langSelect.addEventListener("change", () => {
      currentLabelLang = langSelect.value;
      store.set("bn_lang", currentLabelLang);
      latestDetections = [];
      liveHistoryDetections = [];
      pendingRejections.forEach(tId => clearTimeout(tId));
      pendingRejections.clear();
      verifiedDetections.clear();
      renderDetections();
      
      if (liveWorker && workerReady) {
        updateStatus("status_reloading_model");
        liveWorker.postMessage({ message: 'load_labels', lang: currentLabelLang });
      } else {
        initWorker(currentLabelLang);
      }
    });
  }

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
      store.set("bn_aad_enabled", aadEnabled);
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
      store.set("bn_early_exit_enabled", earlyExitEnabled);
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
      store.set("bn_webgpu_enabled", webgpuEnabled);
      initWorker();
    });
  }

  // Separator Precision Select
  const separatorPrecisionSelect = document.getElementById("separatorPrecisionSelect");
  if (separatorPrecisionSelect) {
    separatorPrecisionSelect.value = separatorPrecision;
    separatorPrecisionSelect.addEventListener("change", () => {
      separatorPrecision = separatorPrecisionSelect.value;
      store.set("bn_separator_precision", separatorPrecision);
      initWorker();
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
    if (audioRouter) audioRouter.updatePipelineBInterval();
  }, (v) => `${v.toFixed(1)}s`, "bn_pipeline_b_window");

  bindRange("pipelineBStrideRange", pipelineBStride, (v) => {
    pipelineBStride = v;
    if (audioRouter) audioRouter.updatePipelineBInterval();
  }, (v) => `${v.toFixed(1)}s`, "bn_pipeline_b_stride");
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
    if (storageKey) store.set(storageKey, val);
  });
}

function setupSettingsToggle() {
  const toggle = settingsToggleEl();
  const drawer = settingsDrawerEl();
  const overlay = settingsOverlayEl();
  const closeBtn = document.getElementById("settingsClose");
  const closeBtnBottom = document.getElementById("settingsCloseBottom");

  if (!toggle || !drawer) return;

  const setState = (open) => {
    // Manage focus to avoid "aria-hidden" violation
    if (!open) {
      // If closing and focus is inside, move it back to toggle
      if (drawer.contains(document.activeElement)) {
        toggle.focus();
      }
    }

    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    drawer.setAttribute("aria-hidden", open ? "false" : "true");
    drawer.classList.toggle("open", open);
    document.body.classList.toggle("drawer-open", open);
    if (overlay) {
      overlay.classList.toggle("active", open);
      overlay.setAttribute("aria-hidden", open ? "false" : "true");
    }

    if (open) {
      // If opening, move focus to close button
      if (closeBtn) closeBtn.focus();
    }
  };

  setState(false);

  toggle.addEventListener("click", () => {
    const open = !drawer.classList.contains("open");
    setState(open);
  });

  if (overlay) overlay.addEventListener("click", () => setState(false));
  if (closeBtn) closeBtn.addEventListener("click", () => setState(false));
  if (closeBtnBottom) closeBtnBottom.addEventListener("click", () => setState(false));
  
  document.addEventListener("keydown", (evt) => {
    if (evt.key === "Escape") setState(false);
  });
}

/**
 * Renders the list of detected species (Live View).
 * Uses DOM diffing to prevent flickering of images.
 */
function renderDetections() {
  const container = detectionsList();
  if (!container) return;
  
  const useGeoFilter = geoEnabled && !!geolocation;
  const all = liveHistoryDetections || [];

  // Filter by Geo (if enabled) and Confidence
  const afterGeo = useGeoFilter
    ? all.filter(p => typeof p.geoscore === "number" ? p.geoscore >= 0.05 : true)
    : all;
  const afterAudio = afterGeo.filter(p => p.confidence >= detectionThreshold);
  
  // Sort chronologically descending so newest verified/analyzing species are on top
  const top = afterAudio.sort((a, b) => b.timestamp - a.timestamp);

  // Empty State
  if (!top.length) {
    container.innerHTML = `
      <div class="col-12 text-center text-muted py-5">
        <i class="bi bi-soundwave fs-1 d-block mb-3 opacity-25"></i>
        <p>${t("msg_no_detections", Math.round(detectionThreshold * 100))}</p>
        ${useGeoFilter ? `<small>${t("msg_geo_active")}</small>` : ""}
      </div>
    `;
    return;
  }

  // Clear empty state message if present
  if (container.querySelector(".text-center.text-muted")) {
    container.innerHTML = "";
  }

  // Diffing Strategy: Map existing cards by species key
  const existingCards = new Map();
  Array.from(container.children).forEach(child => {
    const key = child.dataset.species;
    if (key) existingCards.set(key, child);
  });

  const newKeys = new Set();

  top.forEach((p, index) => {
    const confPct = (p.confidence * 100).toFixed(1);
    const geoInfo = useGeoFilter && typeof p.geoscore === "number"
      ? t("lbl_geo_score", (p.geoscore * 100).toFixed(1))
      : "";
    const commonName = p.commonNameI18n || p.commonName || `Class ${p.index}`;
    const scientificName = p.scientificName || "";
    const key = scientificName || `idx-${p.index}`;

    newKeys.add(key);
    let cardCol = existingCards.get(key);

    const isVerified = (p.status === "verified");

    const playBtnHtml = isVerified
      ? (currentlyPlayingSpecies === scientificName
          ? `<button class="btn btn-sm btn-outline-danger py-0 px-2 play-audio-btn" style="font-size: 0.75rem;" onclick="stopIsolatedAudio()">
               <i class="bi bi-stop-fill me-1"></i>Stop
             </button>`
          : `<button class="btn btn-sm btn-outline-primary py-0 px-2 play-audio-btn" style="font-size: 0.75rem;" onclick="playIsolatedAudio('${scientificName.replace(/'/g, "\\'")}')" ${!verifiedDetections.get(scientificName)?.audioBuffer ? 'disabled' : ''}>
               <i class="bi bi-play-fill me-1"></i>Play Isolated
             </button>`
        )
      : `<button class="btn btn-sm btn-outline-secondary py-0 px-2 play-audio-btn" style="font-size: 0.75rem;" disabled>
           <i class="bi bi-hourglass-split spin-icon me-1"></i>Analyzing...
         </button>`;

    const downloadBtnHtml = isVerified && verifiedDetections.get(scientificName)?.audioBuffer
      ? `<button class="btn btn-sm btn-outline-secondary py-0 px-2 ms-1" style="font-size: 0.75rem;" onclick="downloadIsolatedAudio('${scientificName.replace(/'/g, "\\'")}')" title="Download isolated audio channel">
           <i class="bi bi-download"></i>
         </button>`
      : '';

    if (cardCol) {
      // UPDATE existing card (text only)
      const badge = cardCol.querySelector(".badge");
      if (badge) {
        if (isVerified) {
          badge.innerHTML = `${confPct}%`;
          badge.className = "badge bg-primary bg-opacity-10 text-primary border border-primary border-opacity-10 flex-shrink-0";
        } else {
          badge.innerHTML = `<i class="bi bi-arrow-repeat spin-icon me-1"></i>${confPct}% (Analyzing)`;
          badge.className = "badge bg-warning bg-opacity-10 text-warning border border-warning border-opacity-10 flex-shrink-0";
        }
      }
      
      const geoDiv = cardCol.querySelector(".geo-info");
      if (geoDiv) {
        if (geoInfo) geoDiv.innerHTML = `<i class="bi bi-geo-alt me-1"></i>${geoInfo}`;
        else geoDiv.innerHTML = "";
      }

      const card = cardCol.querySelector(".card");
      if (card) {
        if (isVerified) {
          card.classList.remove("card-pending");
        } else {
          if (!card.classList.contains("card-pending")) {
            card.classList.add("card-pending");
          }
        }
      }

      const playBtnContainer = cardCol.querySelector(".play-btn-container");
      if (playBtnContainer) {
        playBtnContainer.innerHTML = playBtnHtml + downloadBtnHtml;
      }

      container.appendChild(cardCol); // Re-order
    } else {
      // CREATE new card
      cardCol = document.createElement("div");
      cardCol.className = "col-md-6 col-lg-4 fade-in";
      cardCol.dataset.species = key;

      const badgeHtml = isVerified
        ? `<span class="badge bg-primary bg-opacity-10 text-primary border border-primary border-opacity-10 flex-shrink-0">${confPct}%</span>`
        : `<span class="badge bg-warning bg-opacity-10 text-warning border border-warning border-opacity-10 flex-shrink-0"><i class="bi bi-arrow-repeat spin-icon me-1"></i>${confPct}% (Analyzing)</span>`;

      const wikiLang = currentUiLang || "en";
      const wikiUrl = `https://${wikiLang}.wikipedia.org/wiki/${encodeURIComponent(scientificName)}`;
      const ebirdUrl = `https://www.google.com/search?q=site:ebird.org/species/+${encodeURIComponent(scientificName)}`;
      const cardPendingClass = isVerified ? "" : "card-pending";

      cardCol.innerHTML = `
        <div class="card h-100 border-0 shadow-sm overflow-hidden ${cardPendingClass}">
          <div class="d-flex h-100">
            <div class="flex-shrink-0 position-relative" style="width: 90px; background-color: #f8f9fa;">
              <img src="img/dummy.webp" 
                   data-scientific-name="${scientificName}"
                   alt="${commonName}"
                   loading="lazy"
                   style="width: 100%; height: 100%; object-fit: cover;"
                   onerror="this.onerror=null; this.src='img/dummy.webp';">
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
                  <span class="small text-muted geo-info" style="font-size: 0.75rem;">
                    ${geoInfo ? `<i class="bi bi-geo-alt me-1"></i>${geoInfo}` : ""}
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
      container.appendChild(cardCol);

      const imgEl = cardCol.querySelector('img[data-scientific-name]');
      if (imgEl) {
        loadSpeciesImage(scientificName, imgEl, 'img/dummy.webp');
      }
      const ebirdLinkEl = cardCol.querySelector(`a[data-ebird-scientific="${scientificName}"]`);
      if (ebirdLinkEl) {
        loadEbirdLink(scientificName, ebirdLinkEl);
      }
    }
  });

  // Remove old cards
  existingCards.forEach((node, key) => {
    if (!newKeys.has(key)) node.remove();
  });
}

/**
 * Renders the list of local species (Explore View).
 */
function renderExploreList(list) {
  if (list) lastSpeciesList = list;
  const sourceList = list || lastSpeciesList;

  const container = document.getElementById("exploreList");
  if (!container || !sourceList) return;

  if (!geolocation || !geoEnabled) {
    container.innerHTML = `
      <div class="col-12 text-center py-5 text-muted">
        <i class="bi bi-geo-alt-slash fs-1 d-block mb-3 opacity-25"></i>
        <p data-i18n="msg_explore_geo_disabled">${t("msg_explore_geo_disabled")}</p>
      </div>
    `;
    return;
  }

  const sorted = sourceList
    .filter(item => item.geoscore >= geoThreshold) 
    .sort((a, b) => b.geoscore - a.geoscore);

  container.innerHTML = "";
  
  if (sorted.length === 0) {
    container.innerHTML = `<div class="col-12 text-center text-muted py-5">${t("msg_explore_no_species", Math.round(geoThreshold * 100))}<br>${t("msg_explore_lower_threshold")}</div>`;
    return;
  }

  sorted.forEach(bird => {
    const scorePct = (bird.geoscore * 100).toFixed(1);
    const common = bird.commonNameI18n || bird.commonName;
    
    const wikiLang = currentUiLang || "en";
    const wikiUrl = `https://${wikiLang}.wikipedia.org/wiki/${encodeURIComponent(bird.scientificName)}`;
    const ebirdUrl = `https://www.google.com/search?q=site:ebird.org/species/+${encodeURIComponent(bird.scientificName)}`;

    const col = document.createElement("div");
    col.className = "col-md-6 col-lg-4";
    col.innerHTML = `
      <div class="card h-100 border-0 shadow-sm overflow-hidden">
        <div class="d-flex h-100">
          <div class="flex-shrink-0 position-relative" style="width: 90px; background-color: #f8f9fa;">
            <img src="img/dummy.webp" 
                 data-scientific-name="${bird.scientificName}"
                 alt="${common}"
                 loading="lazy"
                 style="width: 100%; height: 100%; object-fit: cover;"
                 onerror="this.onerror=null; this.src='img/dummy.webp';">
          </div>
          <div class="card-body py-2 px-3 flex-grow-1">
            <div class="d-flex justify-content-between align-items-start mb-1">
              <div class="overflow-hidden me-2">
                <h6 class="card-title mb-0 fw-bold text-dark text-truncate" style="font-size: 0.95rem;" title="${common}">${common}</h6>
                <div class="text-muted fst-italic small mt-1 mb-1 text-truncate" style="font-size: 0.8rem;">${bird.scientificName}</div>
                <div class="d-flex align-items-center gap-2 mb-2">
                  <a href="${wikiUrl}" target="_blank" rel="noopener" class="species-link" title="Wikipedia">
                    <i class="bi bi-wikipedia"></i> Wikipedia
                  </a>
                  <span class="species-links-divider">|</span>
                  <a href="${ebirdUrl}" target="_blank" rel="noopener" data-ebird-scientific="${bird.scientificName}" class="species-link" title="eBird">
                    <i class="bi bi-box-arrow-up-right"></i> eBird
                  </a>
                </div>
              </div>
              <span class="badge bg-light text-dark border flex-shrink-0">
                ${scorePct}%
              </span>
            </div>
            <div class="mt-3">
              <div class="progress" style="height: 4px;">
                <div class="progress-bar bg-success" role="progressbar" style="width: ${scorePct}%" aria-valuenow="${scorePct}" aria-valuemin="0" aria-valuemax="100"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
    container.appendChild(col);

    const imgEl = col.querySelector('img[data-scientific-name]');
    if (imgEl) {
      loadSpeciesImage(bird.scientificName, imgEl, 'img/dummy.webp');
    }
    const ebirdLinkEl = col.querySelector(`a[data-ebird-scientific="${bird.scientificName}"]`);
    if (ebirdLinkEl) {
      loadEbirdLink(bird.scientificName, ebirdLinkEl);
    }
  });
}

/* ==========================================================================
   11. GEOLOCATION
   ========================================================================== */

function updateGeoDisplay(key, coords) {
  const status = geoStatusEl();
  const coordsEl = geoCoordsEl();
  if (status) {
    status.textContent = t(key);
    status.setAttribute("data-i18n", key);
  }
  if (coordsEl) {
    coordsEl.textContent = coords
      ? `${coords.lat.toFixed(4)}, ${coords.lon.toFixed(4)} (±${Math.round(coords.accuracy)}m)`
      : "—";
  }
}

function getGeolocation() {
  if (!navigator.geolocation) {
    updateGeoDisplay("status_geo_unsupported", null);
    return;
  }
  if (geoWatchId !== null) {
    navigator.geolocation.clearWatch(geoWatchId);
    geoWatchId = null;
  }
  updateGeoDisplay("status_geo_requesting", null);

  geoWatchId = navigator.geolocation.watchPosition(
    (pos) => {
      geolocation = {
        lat: pos.coords.latitude,
        lon: pos.coords.longitude,
        accuracy: pos.coords.accuracy,
        timestamp: pos.timestamp
      };
      updateGeoDisplay("status_geo_acquired", geolocation);
      sendAreaScores();
      renderDetections(); // re-filter with geo prior active
    },
    (err) => {
      console.warn("Geolocation error", err);
      geolocation = null;
      updateGeoDisplay("status_geo_failed", null);
      renderDetections();
    },
    {
      enableHighAccuracy: true,
      maximumAge: 15000,
      timeout: 20000
    }
  );
}

function sendAreaScores() {
  if (!liveWorker || !geolocation) return;
  const now = new Date();
  const startYear = new Date(now.getFullYear(), 0, 1);
  const week = Math.min(
    52,
    Math.max(
      1,
      Math.floor((now - startYear) / (7 * 24 * 60 * 60 * 1000)) + 1
    )
  );
  const hour = now.getHours();
  liveWorker.postMessage({
    message: "area-scores",
    latitude: geolocation.lat,
    longitude: geolocation.lon,
    week,
    hour
  });
}

/* ==========================================================================
   12. SYSTEM UTILITIES
   ========================================================================== */

// Wake Lock (Keep screen on while recording)
let wakeLock = null;
let wakeLockRequested = false;

async function requestWakeLock() {
  if (!('wakeLock' in navigator)) return;
  try {
    wakeLock = await navigator.wakeLock.request('screen');
    wakeLockRequested = true;
    wakeLock.addEventListener('release', () => {
      wakeLock = null;
      wakeLockRequested = false;
    });
  } catch (e) {
    console.warn("Wake Lock request failed:", e);
  }
}

function releaseWakeLock() {
  if (wakeLock) {
    wakeLock.release().catch(()=>{});
    wakeLock = null;
    wakeLockRequested = false;
  }
}

// Lifecycle Management
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden" && isListening) {
    stopListening();
  }
});

window.addEventListener("pagehide", () => {
  if (isListening) {
    stopListening();
  }
});

/**
 * Temporal Pooling: Log-Mean-Exp over logits.
 * Smooths predictions over time to reduce noise.
 */
function computeTemporalPooledDetections(sets) {
  if (!sets || !sets.length) return [];
  if (sets.length === 1) return sets[0];

  const eps = 1e-8;
  const byIndex = new Map();
  
  // Group confidences by species index
  for (let s = 0; s < sets.length; s++) {
    for (const det of sets[s]) {
      const idx = det.index;
      if (!byIndex.has(idx)) {
        byIndex.set(idx, { samples: [], ref: det });
      }
      byIndex.get(idx).samples.push(det.confidence);
    }
  }

  const pooled = [];
  for (const [idx, entry] of byIndex.entries()) {
    const samples = entry.samples;
    // Convert confidences to logits
    const logits = samples.map(c => {
      const clipped = Math.min(1 - eps, Math.max(eps, c));
      return Math.log(clipped / (1 - clipped));
    });
    
    // Log-mean-exp pooling
    const maxLogit = Math.max(...logits);
    const sumExp = logits.reduce((acc, l) => acc + Math.exp(l - maxLogit), 0);
    const lme = maxLogit + Math.log(sumExp / logits.length);
    
    // Back to probability
    const pooledConf = 1 / (1 + Math.exp(-lme));

    pooled.push({
      ...entry.ref,
      confidence: pooledConf
    });
  }

  pooled.sort((a, b) => b.confidence - a.confidence);
  return pooled;
}

/* ==========================================================================
   13. AUDIO PLAYBACK UTILITIES
   ========================================================================== */

async function playIsolatedAudio(scientificName) {
  console.log("[playIsolatedAudio] Request to play:", scientificName);
  if (currentlyPlayingSpecies === scientificName) {
    console.log("[playIsolatedAudio] Stopping currently playing:", scientificName);
    stopIsolatedAudio();
    return;
  }

  const verified = verifiedDetections.get(scientificName);
  console.log("[playIsolatedAudio] verified entry from Map:", verified);
  if (!verified) {
    console.warn("[playIsolatedAudio] No verified entry found for:", scientificName);
    return;
  }
  if (!verified.audioBuffer) {
    console.warn("[playIsolatedAudio] verified entry has no audioBuffer for:", scientificName);
    return;
  }
  console.log("[playIsolatedAudio] verified.audioBuffer length:", verified.audioBuffer.length);

  try {
    stopIsolatedAudio();

    if (!playbackAudioContext) {
      try {
        playbackAudioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
      } catch (e) {
        console.warn("[playIsolatedAudio] Failed to create AudioContext with sampleRate, falling back to default constructor:", e);
        playbackAudioContext = new (window.AudioContext || window.webkitAudioContext)();
      }
    }
    if (playbackAudioContext.state === "suspended") {
      await playbackAudioContext.resume();
    }

    // Peak normalize the track so it's clearly audible
    const samples = verified.audioBuffer;
    let maxVal = 0;
    for (let i = 0; i < samples.length; i++) {
      const abs = Math.abs(samples[i]);
      if (abs > maxVal) maxVal = abs;
    }

    const normalized = new Float32Array(samples.length);
    if (maxVal > 0.0001) {
      const gain = 0.8 / maxVal;
      const clampedGain = Math.min(20.0, gain); // up to 20x gain boost
      console.log(`[playIsolatedAudio] Normalizing audio. Peak: ${maxVal.toFixed(4)}, Applied Gain: ${clampedGain.toFixed(2)}x`);
      for (let i = 0; i < samples.length; i++) {
        normalized[i] = samples[i] * clampedGain;
      }
    } else {
      console.log("[playIsolatedAudio] Track is near-silent. Playing raw samples.");
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

function stopIsolatedAudio() {
  if (activeAudioSource) {
    try { activeAudioSource.stop(); } catch(e) {}
    activeAudioSource = null;
  }
  currentlyPlayingSpecies = null;
  renderDetections();
}

function populateSeparatorDropdowns() {
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
      store.set("bn_separator_model", selectedSeparatorModel);
      customModelBytes = null;
      customOpt.textContent = "Load Custom Local ONNX Model...";
      initWorker();
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
        store.set("bn_custom_model_name", customModelName);
        customOpt.textContent = `Custom: ${customModelName}`;
        
        selectedSeparatorModel = "custom";
        store.set("bn_separator_model", "custom");
        select.value = "custom";

        console.log(`[App] Loading custom local model: ${file.name}...`);

        const reader = new FileReader();
        reader.onload = (e) => {
          customModelBytes = e.target.result;
          console.log("[App] Custom local model bytes loaded. Re-initializing workers...");
          initWorker();
        };
        reader.readAsArrayBuffer(file);
      } else {
        select.value = selectedSeparatorModel;
      }
    });
  }
}

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

function downloadIsolatedAudio(scientificName) {
  const verified = verifiedDetections.get(scientificName);
  if (!verified || !verified.audioBuffer) {
    console.warn("[downloadIsolatedAudio] No verified audio buffer found for", scientificName);
    return;
  }
  
  const samples = verified.audioBuffer;
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

// Bind functions to window context for onclick handlers (only on live dashboard page)
if (document.getElementById("liveSpectrogram")) {
  window.playIsolatedAudio = playIsolatedAudio;
  window.stopIsolatedAudio = stopIsolatedAudio;
  window.downloadIsolatedAudio = downloadIsolatedAudio;
}

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
  if (ebirdCodeCache.has(scientificName)) {
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
