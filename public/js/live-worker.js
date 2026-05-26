/**
 * BirdNET Live - Pipeline A (Live Worker)
 * Handles TensorFlow.js TFLite model loading, audio preprocessing,
 * real-time inference (3s window / 1s interval), and post-processing.
 */

/* ==========================================================================
   1. IMPORTS & CONFIGURATION
   ========================================================================== */

const params = new URL(self.location.href).searchParams;
const TF_PATH = params.get('tf') || 'js/tfjs-4.14.0.min.js';
const prefix = self.location.origin + (params.get('prefix') || '/');
const TFLITE_PATH = prefix + 'js/tf-tflite.min.js';
const WASM_PATH = prefix + 'tflite-wasm/';

importScripts(TF_PATH);
importScripts(TFLITE_PATH);

// Configure local WASM binaries path for offline execution
tflite.setWasmPath(WASM_PATH);

// Paths
const MODEL_PATH = prefix + 'models/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite';
const AREA_MODEL_PATH = prefix + 'models/birdnet/area-model/model.json';
const LABELS_DIR = prefix + 'models/birdnet/labels';

// Audio Constants
const SAMPLE_RATE = 48000;
const WINDOW_SAMPLES = 144000; // 3 seconds at 48kHz

/* ==========================================================================
   2. GLOBAL STATE
   ========================================================================== */

// Models
let birdModel = null;
let areaModel = null;

// Data
let birds = []; // Array of { scientificName, commonName, geoscore, ... }

// Inference Cache (for re-applying geo priors without re-running inference)
let lastPredictionList = null;
let lastMeans = null;
let lastHopSamples = null;
let lastNumFrames = 0;
let lastWindowSize = WINDOW_SAMPLES;

/* ==========================================================================
   3. INITIALIZATION
   ========================================================================== */

// Start initialization immediately
init();

async function init() {
  // Use CPU backend for WASM TFLite operations
  await tf.setBackend('cpu');

  // 1. Load Main TFLite Model
  try {
    postMessage({ message: 'load_model', progress: 10 });
    birdModel = await tflite.loadTFLiteModel(MODEL_PATH);
    postMessage({ message: 'load_model', progress: 70 });
  } catch (err) {
    console.error("[Live Worker] Failed to load TFLite model:", err);
    postMessage({ message: 'worker_error', error: err.message });
    return;
  }

  // 2. Warmup
  postMessage({ message: 'warmup', progress: 80 });
  tf.tidy(() => {
    const dummyInput = tf.zeros([1, WINDOW_SAMPLES], 'float32');
    birdModel.predict(dummyInput);
    dummyInput.dispose();
  });

  // 3. Load Geo Model (Optional)
  postMessage({ message: 'load_geomodel', progress: 90 });
  try {
    areaModel = await tf.loadGraphModel(AREA_MODEL_PATH);
  } catch (e) {
    console.warn("[Live Worker] Geo model failed to load", e);
  }

  // 4. Load Labels
  postMessage({ message: 'load_labels', progress: 95 });
  await loadLabels();

  postMessage({ message: 'loaded' });
}

async function loadLabels(langOverride) {
  const navigatorLang = params.get('lang');
  const supportedLanguages = [
    'af', 'da', 'en_us', 'fr', 'ja', 'no', 'ro', 'sl', 'tr', 'ar', 'de', 'es', 'hu',
    'ko', 'pl', 'ru', 'sv', 'uk', 'cs', 'en_uk', 'fi', 'it', 'nl', 'pt', 'sk', 'th', 'zh'
  ];
  
  // Determine language
  const lang = (() => {
    if (langOverride) return langOverride;
    const req = params.get('lang');
    if (req) return req;
    if (!navigatorLang) return 'en_us';
    const base = navigatorLang.split('-')[0];
    return supportedLanguages.find(l => l.startsWith(base)) || 'en_us';
  })();

  // Fetch default (English) and localized lists
  const birdsList = (await fetch(LABELS_DIR + '/en_us.txt').then(r => r.text())).split('\n');
  let birdsListI18n;
  try {
    birdsListI18n = (await fetch(`${LABELS_DIR}/${lang}.txt`).then(r => r.text())).split('\n');
  } catch {
    birdsListI18n = birdsList;
  }

  // Merge into objects
  const newBirds = birdsList.map((base, i) => {
    const i18nLine = birdsListI18n[i] || base;
    const [sciBase, comBase] = base.split('_');
    const [sciLoc, comLoc] = i18nLine.split('_');
    return {
      geoscore: 1, // Default probability
      scientificName: sciBase || base,
      commonName: comBase || base,
      commonNameI18n: comLoc || comBase || base
    };
  });

  // Preserve geoscores if we already had birds loaded
  if (birds.length === newBirds.length) {
    for (let i = 0; i < birds.length; i++) {
      newBirds[i].geoscore = birds[i].geoscore;
    }
  }

  birds = newBirds;
}

/* ==========================================================================
   4. MESSAGE HANDLING
   ========================================================================== */

onmessage = async ({ data }) => {
  switch (data.message) {
    case 'predict':
      await handlePredict(data);
      break;
    case 'area-scores':
      await handleAreaScores(data);
      break;
    case 'load_labels':
      await loadLabels(data.lang);
      postMessage({ message: 'labels_loaded', lang: data.lang });
      break;
    case 'get_species_list':
      postMessage({ 
        message: 'species_list', 
        list: birds.map((b, i) => ({
          index: i,
          scientificName: b.scientificName,
          commonName: b.commonName,
          commonNameI18n: b.commonNameI18n,
          geoscore: b.geoscore
        }))
      });
      break;
  }
};

/* ==========================================================================
   5. CORE LOGIC: PREDICTION
   ========================================================================== */

async function handlePredict(data) {
  if (!birdModel) return;

  // 1. Prepare Audio Window
  const overlapSecRaw = parseFloat(data.overlapSec ?? 1.5);
  const overlapSec = Math.min(2.5, Math.max(0.0, Math.round(overlapSecRaw * 2) / 2));
  const overlapSamples = Math.round(overlapSec * SAMPLE_RATE);
  const hopSamples = Math.max(1, WINDOW_SAMPLES - overlapSamples);

  const pcm = data.pcmAudio || new Float32Array(0);
  const total = pcm.length;

  // Frame the audio (sliding window)
  const numFrames = Math.max(1, Math.ceil(Math.max(0, total - WINDOW_SAMPLES) / hopSamples) + 1);
  const framed = new Float32Array(numFrames * WINDOW_SAMPLES);
  for (let f = 0; f < numFrames; f++) {
    const start = f * hopSamples;
    const srcEnd = Math.min(start + WINDOW_SAMPLES, total);
    framed.set(pcm.subarray(start, srcEnd), f * WINDOW_SAMPLES);
  }

  // 2. Run Inference
  let predictionList = [];
  
  // Run inference frame by frame on the TFLite model
  for (let f = 0; f < numFrames; f++) {
    const slice = framed.subarray(f * WINDOW_SAMPLES, (f + 1) * WINDOW_SAMPLES);
    const audioTensor = tf.tensor2d(slice, [1, WINDOW_SAMPLES], 'float32');
    const resTensor = birdModel.predict(audioTensor);
    const predictions = await resTensor.array();
    
    // push row 0 to predictionList
    predictionList.push(predictions[0]);
    
    audioTensor.dispose();
    resTensor.dispose();
  }

  // 3. Apply Sensitivity
  const sensitivity = parseFloat(data.sensitivity || 1.0);
  if (sensitivity !== 1.0) {
    predictionList = applySensitivity(predictionList, sensitivity);
  }

  // 4. Cache results (for geo updates)
  lastPredictionList = predictionList;
  lastHopSamples = hopSamples;
  lastNumFrames = numFrames;
  lastWindowSize = WINDOW_SAMPLES;

  // 5. Emit Segment Results
  emitSegments(predictionList, hopSamples, WINDOW_SAMPLES);

  // 6. Pool Results (Log-Mean-Exp) & Emit
  emitPooled(predictionList);
}

function emitSegments(predictionList, hopSamples, windowSize) {
  const segments = [];
  for (let f = 0; f < predictionList.length; f++) {
    const startSec = (f * hopSamples) / SAMPLE_RATE;
    const endSec = startSec + windowSize / SAMPLE_RATE;
    const preds = predictionList[f].map((conf, i) => ({
      index: i,
      confidence: conf,
      geoscore: birds[i].geoscore,
      scientificName: birds[i].scientificName,
      commonName: birds[i].commonName,
      commonNameI18n: birds[i].commonNameI18n
    }));
    segments.push({ start: startSec, end: endSec, preds });
  }
  postMessage({ message: 'segments', segments });
}

function emitPooled(predictionList) {
  const numClasses = predictionList[0]?.length || 0;
  const numFrames = predictionList.length;
  const ALPHA = 5.0; // Pooling factor
  
  const sumsExp = new Float64Array(numClasses);
  for (let f = 0; f < numFrames; f++) {
    const row = predictionList[f];
    for (let i = 0; i < numClasses; i++) {
      sumsExp[i] += Math.exp(ALPHA * row[i]);
    }
  }
  
  lastMeans = Array.from(sumsExp, s => Math.log(s / numFrames) / ALPHA);

  const pooled = lastMeans.map((m, i) => ({
    index: i,
    scientificName: birds[i].scientificName,
    commonName: birds[i].commonName,
    commonNameI18n: birds[i].commonNameI18n,
    confidence: m,
    geoscore: birds[i].geoscore
  }));
  
  postMessage({ message: 'pooled', pooled });
}

/**
 * Adjusts logits based on sensitivity slider.
 * Sensitivity > 1.0 boosts weak signals.
 */
function applySensitivity(list, sensitivity) {
  const bias = (sensitivity - 1.0) * 5.0; 
  return list.map(row => row.map(p => {
    const pp = Math.max(1e-7, Math.min(1 - 1e-7, p));
    const logit = Math.log(pp / (1 - pp));
    return 1 / (1 + Math.exp(-(logit + bias)));
  }));
}

/* ==========================================================================
   6. CORE LOGIC: GEOLOCATION
   ========================================================================== */

async function handleAreaScores(data) {
  if (!areaModel) return;

  // Calculate week of year
  tf.engine().startScope();
  const startOfYear = new Date(new Date().getFullYear(), 0, 1);
  startOfYear.setDate(startOfYear.getDate() + (1 - (startOfYear.getDay() % 7)));
  const week = Math.round((Date.now() - startOfYear.getTime()) / 604800000) + 1;

  // Predict occurrence probabilities
  const input = tf.tensor([[data.latitude, data.longitude, week]]);
  const areaScores = await areaModel.predict(input).data();
  tf.engine().endScope();

  // Update global state
  for (let i = 0; i < birds.length; i++) {
    birds[i].geoscore = areaScores[i];
  }
  
  postMessage({ message: 'area-scores' });

  // Re-emit cached results with new geo scores
  if (lastPredictionList && lastHopSamples != null) {
    emitSegments(lastPredictionList, lastHopSamples, lastWindowSize);
  }
  if (lastMeans) {
    const pooled = lastMeans.map((m, i) => ({
      index: i,
      scientificName: birds[i].scientificName,
      commonName: birds[i].commonName,
      commonNameI18n: birds[i].commonNameI18n,
      confidence: m,
      geoscore: birds[i].geoscore
    }));
    postMessage({ message: 'pooled', pooled });
  }
}