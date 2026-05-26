/**
 * BirdNET Live - Pipeline B (Segmentation Worker)
 * 
 * Manages environmental source separation (ONNX WebGPU) and
 * high-fidelity secondary species identification.
 */

/* ==========================================================================
   1. IMPORTS & CONFIGURATION
   ========================================================================== */

const params = new URL(self.location.href).searchParams;
const TF_PATH = params.get('tf') || 'js/tfjs-4.14.0.min.js';
const prefix = self.location.origin + (params.get('prefix') || '/');
const TFLITE_PATH = prefix + 'js/tf-tflite.min.js';
const WASM_PATH = prefix + 'tflite-wasm/';
const ORT_PATH = prefix + 'js/ort.min.js';
const ORT_WASM_PATH = prefix + 'onnx-wasm/';

importScripts(TF_PATH);
importScripts(TFLITE_PATH);
importScripts(ORT_PATH);

// Configure local WASM paths for offline execution
tflite.setWasmPath(WASM_PATH);
ort.env.wasm.wasmPaths = ORT_WASM_PATH;

// Paths
const MODEL_PATH = prefix + 'models/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite';
const LABELS_DIR = prefix + 'models/birdnet/labels';
const SEPARATOR_MODEL_PATH = prefix + 'models/biocppnet.onnx';

// Audio Constants
const SAMPLE_RATE = 48000;
const WINDOW_SAMPLES = 144000; // 3 seconds at 48kHz
const HOP_SAMPLES = 72000;      // 1.5 seconds overlap

/* ==========================================================================
   2. AUDIO SEPARATOR INTERFACE
   ========================================================================== */

class AudioSeparator {
    constructor() {}
    async loadModel() {
        throw new Error("Method loadModel() must be implemented.");
    }
    async separate(audioBuffer) {
        throw new Error("Method separate(audioBuffer) must be implemented.");
    }
}

class ONNXAudioSeparator extends AudioSeparator {
    constructor(modelPath) {
        super();
        this.modelPath = modelPath;
        this.session = null;
        this.hasFailedToLoad = false;
    }

    async loadModel() {
        try {
            console.log(`[ONNXAudioSeparator] Loading model from ${this.modelPath}...`);
            // Attempt to load using WebGPU first, falling back to WebAssembly
            this.session = await ort.InferenceSession.create(this.modelPath, {
                executionProviders: ['webgpu', 'wasm']
            });
            console.log(`[ONNXAudioSeparator] Model loaded successfully on provider: ${this.session.handler.provider || 'default'}`);
        } catch (err) {
            console.warn(`[ONNXAudioSeparator] Failed to load ONNX model. Dynamic DSP fallback will be used instead. Error:`, err);
            this.hasFailedToLoad = true;
            this.session = null;
        }
    }

    async separate(audioBuffer) {
        if (this.hasFailedToLoad || !this.session) {
            console.log(`[ONNXAudioSeparator] Utilizing DSP/mock separation fallback.`);
            return this.dspSeparate(audioBuffer);
        }

        try {
            const inputTensor = new ort.Tensor('float32', audioBuffer, [1, audioBuffer.length]);
            const feeds = { input: inputTensor }; // Replace with actual input node name
            const results = await this.session.run(feeds);
            
            const outputNames = Object.keys(results);
            const tracks = [];
            for (const name of outputNames) {
                const tensor = results[name];
                if (tensor && tensor.data instanceof Float32Array) {
                    tracks.push(tensor.data);
                }
            }

            if (tracks.length > 0) {
                return tracks;
            } else {
                throw new Error("No valid Float32Array tracks found in ONNX outputs.");
            }
        } catch (err) {
            console.error(`[ONNXAudioSeparator] ONNX run failed, permanently downgrading to DSP separation:`, err);

            // MOBILE FIX: Permanently downgrade on OOM / device-lost.
            // On mobile GPUs, once a WebGPU device is lost or runs out of memory,
            // the session is corrupted. Retrying every 4.5s wastes GPU cycles and
            // can cause thermal throttling. Mark as permanently failed.
            this.hasFailedToLoad = true;
            try { this.session.release(); } catch (_) {}
            this.session = null;

            // Notify main thread so the UI can reflect the downgrade
            self.postMessage({
                type: "PIPELINE_STATUS",
                payload: {
                    status: "webgpu_fallback",
                    message: "WebGPU execution failed. Pipeline B is running on CPU (DSP fallback).",
                    error: err.message
                }
            });

            return this.dspSeparate(audioBuffer);
        }
    }

    dspSeparate(audioBuffer) {
        const length = audioBuffer.length;
        const track1 = new Float32Array(length);
        const track2 = new Float32Array(length);

        // Zero-phase crossover filter split at 3.5kHz
        const cutoff = 3500;
        const Fs = 48000;
        const alpha = Math.exp(-2 * Math.PI * cutoff / Fs);

        let prevLow = 0;
        for (let i = 0; i < length; i++) {
            const x = audioBuffer[i];
            prevLow = alpha * prevLow + (1 - alpha) * x;
            track1[i] = prevLow;
            track2[i] = x - prevLow;
        }

        return [track1, track2];
    }
}

/* ==========================================================================
   3. GLOBAL STATE
   ========================================================================== */

let separatorModel = null;
let classificationModel = null;
let birds = [];
let isReady = false;

/* ==========================================================================
   4. INITIALIZATION
   ========================================================================== */

init();

async function init() {
    try {
        await tf.setBackend('cpu');

        // 1. Load Labels
        await loadLabels();

        // 2. Load Separator Model
        separatorModel = new ONNXAudioSeparator(SEPARATOR_MODEL_PATH);
        await separatorModel.loadModel();

        // 3. Load TFLite Classification Model
        console.log("[Segmentation Worker] Loading BirdNET FP32 model...");
        classificationModel = await tflite.loadTFLiteModel(MODEL_PATH);
        console.log("[Segmentation Worker] BirdNET FP32 model loaded successfully.");

        // Warmup classification model
        tf.tidy(() => {
            const dummyInput = tf.zeros([1, WINDOW_SAMPLES], 'float32');
            classificationModel.predict(dummyInput);
            dummyInput.dispose();
        });

        isReady = true;
        console.log("[Segmentation Worker] Initialization complete and ready.");
    } catch (err) {
        console.error("[Segmentation Worker] Initialization failed:", err);
    }
}

async function loadLabels() {
    const navigatorLang = params.get('lang');
    const supportedLanguages = [
        'af', 'da', 'en_us', 'fr', 'ja', 'no', 'ro', 'sl', 'tr', 'ar', 'de', 'es', 'hu',
        'ko', 'pl', 'ru', 'sv', 'uk', 'cs', 'en_uk', 'fi', 'it', 'nl', 'pt', 'sk', 'th', 'zh'
    ];
    
    const lang = (() => {
        const req = params.get('lang');
        if (req) return req;
        if (!navigatorLang) return 'en_us';
        const base = navigatorLang.split('-')[0];
        return supportedLanguages.find(l => l.startsWith(base)) || 'en_us';
    })();

    try {
        const birdsList = (await fetch(LABELS_DIR + '/en_us.txt').then(r => r.text())).split('\n');
        let birdsListI18n;
        try {
            birdsListI18n = (await fetch(`${LABELS_DIR}/${lang}.txt`).then(r => r.text())).split('\n');
        } catch {
            birdsListI18n = birdsList;
        }

        birds = birdsList.map((base, i) => {
            const i18nLine = birdsListI18n[i] || base;
            const [sciBase, comBase] = base.split('_');
            const [sciLoc, comLoc] = i18nLine.split('_');
            return {
                scientificName: sciBase || base,
                commonName: comBase || base,
                commonNameI18n: comLoc || comBase || base
            };
        });
    } catch (e) {
        console.error("[Segmentation Worker] Failed to load labels:", e);
    }
}

/* ==========================================================================
   5. MESSAGE HANDLING & PROCESS SEGMENT
   ========================================================================== */

self.onmessage = async (event) => {
    const { type, payload } = event.data || {};
    
    if (type === "PROCESS_SEGMENT") {
        const { segmentId, timestamp, sampleRate, audioBuffer, meta } = payload;
        
        if (!isReady) {
            console.warn(`[Segmentation Worker] Received segment ${segmentId} but worker is not ready yet.`);
            self.postMessage({
                type: "SEGMENT_RESULT",
                payload: {
                    segmentId: segmentId,
                    timestamp: timestamp,
                    results: [],
                    noiseDetected: false,
                    error: "Worker not initialized"
                }
            });
            return;
        }

        console.log(`[Segmentation Worker] Processing segment ${segmentId}. Samples: ${audioBuffer.length}`);
        
        try {
            // Step 1: Run source separation
            const isolatedTracks = await separatorModel.separate(audioBuffer);
            console.log(`[Segmentation Worker] Separation returned ${isolatedTracks.length} tracks.`);

            const results = [];
            
            // Step 2: Classify each separated track
            for (let tIdx = 0; tIdx < isolatedTracks.length; tIdx++) {
                const track = isolatedTracks[tIdx];
                const trackLen = track.length;

                // Frame the track audio into 3-second slices with 1.5s hop size
                const numFrames = Math.max(1, Math.ceil(Math.max(0, trackLen - WINDOW_SAMPLES) / HOP_SAMPLES) + 1);
                const framed = new Float32Array(numFrames * WINDOW_SAMPLES);
                for (let f = 0; f < numFrames; f++) {
                    const start = f * HOP_SAMPLES;
                    const srcEnd = Math.min(start + WINDOW_SAMPLES, trackLen);
                    framed.set(track.subarray(start, srcEnd), f * WINDOW_SAMPLES);
                }

                // Run inference frame by frame
                const predictionList = [];
                for (let f = 0; f < numFrames; f++) {
                    const slice = framed.subarray(f * WINDOW_SAMPLES, (f + 1) * WINDOW_SAMPLES);
                    const audioTensor = tf.tensor2d(slice, [1, WINDOW_SAMPLES], 'float32');
                    const resTensor = classificationModel.predict(audioTensor);
                    const predictions = await resTensor.array();
                    predictionList.push(predictions[0]);
                    audioTensor.dispose();
                    resTensor.dispose();
                }

                // Pool prediction results using Log-Mean-Exp
                const numClasses = predictionList[0]?.length || 0;
                const ALPHA = 5.0;
                const sumsExp = new Float64Array(numClasses);
                for (let f = 0; f < numFrames; f++) {
                    const row = predictionList[f];
                    for (let i = 0; i < numClasses; i++) {
                        sumsExp[i] += Math.exp(ALPHA * row[i]);
                    }
                }
                const pooledPredictions = Array.from(sumsExp, s => Math.log(s / numFrames) / ALPHA);

                // Map pooled predictions to sorted objects
                const formattedPredictions = pooledPredictions
                    .map((confidence, idx) => ({
                        speciesCode: birds[idx] ? birds[idx].scientificName : `class_${idx}`,
                        commonName: birds[idx] ? (birds[idx].commonNameI18n || birds[idx].commonName) : `Class ${idx}`,
                        scientificName: birds[idx] ? birds[idx].scientificName : `Class ${idx}`,
                        confidence: confidence
                    }))
                    .sort((a, b) => b.confidence - a.confidence)
                    .slice(0, 10); // Keep top 10

                results.push({
                    channelId: tIdx,
                    label: `separated_channel_${tIdx}`,
                    audioBuffer: track,
                    predictions: formattedPredictions
                });
            }

            // Transfer separated audio buffers back to main thread to save memory
            const transferList = results.map(r => r.audioBuffer.buffer);
            // Also transfer the original audioBuffer back if it wasn't destroyed
            if (audioBuffer && audioBuffer.buffer) {
                transferList.push(audioBuffer.buffer);
            }

            self.postMessage({
                type: "SEGMENT_RESULT",
                payload: {
                    segmentId: segmentId,
                    timestamp: timestamp,
                    results: results,
                    noiseDetected: false,
                    error: null
                }
            }, transferList);
            
        } catch (err) {
            console.error("[Segmentation Worker] Error processing segment:", err);
            self.postMessage({
                type: "SEGMENT_RESULT",
                payload: {
                    segmentId: segmentId,
                    timestamp: timestamp,
                    results: [],
                    noiseDetected: false,
                    error: err.message
                }
            });
        }
    }
};
