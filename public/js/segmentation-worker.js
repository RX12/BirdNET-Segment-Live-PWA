/**
 * BirdNET Live - Pipeline B (Segmentation Worker)
 * 
 * Manages environmental source separation (ONNX WebGPU) and
 * high-fidelity secondary species identification.
 */

/* ==========================================================================
   1. IMPORTS & CONFIGURATION
   ========================================================================== */

const prefix = self.location.origin + self.location.pathname.substring(0, self.location.pathname.lastIndexOf('/js/')) + '/';
const TF_PATH = prefix + 'js/tfjs-4.14.0.min.js';
const TFLITE_PATH = prefix + 'js/tf-tflite.min.js';
const WASM_PATH = prefix + 'tflite-wasm/';
const ORT_PATH = prefix + 'js/ort.min.js';
const ORT_WASM_PATH = prefix + 'onnx-wasm/';

let currentOutputSampleRate = 48000;
let currentPipelineBWindow = 9.0;

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
    constructor(modelPath, runOnGPU) {
        super();
        this.modelPath = modelPath;
        this.runOnGPU = runOnGPU !== false;
        this.session = null;
        this.hasFailedToLoad = false;
    }

    async loadModel() {
        try {
            console.log(`[ONNXAudioSeparator] Loading model from ${this.modelPath}...`);
            // Attempt to load using WebGPU first (if enabled), falling back to WebAssembly
            const providers = this.runOnGPU ? ['webgpu', 'wasm'] : ['wasm'];
            this.session = await ort.InferenceSession.create(this.modelPath, {
                executionProviders: providers
            });
            console.log(`[ONNXAudioSeparator] Model loaded successfully on provider: ${this.session.handler.provider || 'default'}`);
        } catch (err) {
            console.warn(`[ONNXAudioSeparator] Failed to load ONNX model. Dynamic DSP fallback will be used instead. Error:`, err);
            this.hasFailedToLoad = true;
            this.session = null;
        }
    }

    async loadModelFromBytes(modelBytes) {
        try {
            console.log(`[ONNXAudioSeparator] Loading model from custom bytes (${(modelBytes.byteLength / 1024 / 1024).toFixed(2)} MB)...`);
            this.session = await ort.InferenceSession.create(modelBytes, {
                executionProviders: ['webgpu', 'wasm']
            });
            console.log(`[ONNXAudioSeparator] Custom model loaded successfully on provider: ${this.session.handler.provider || 'default'}`);
        } catch (err) {
            console.error(`[ONNXAudioSeparator] Failed to load custom ONNX model from bytes:`, err);
            this.hasFailedToLoad = true;
            this.session = null;
            throw err;
        }
    }

    async separate(audioBuffer) {
        if (this.hasFailedToLoad || !this.session) {
            console.log(`[ONNXAudioSeparator] Utilizing DSP/mock separation fallback.`);
            return this.dspSeparate(audioBuffer);
        }

        try {
            const inputTensor = new ort.Tensor('float32', audioBuffer, [1, 1, audioBuffer.length]);
            const inputName = this.session.inputNames[0];
            const feeds = { [inputName]: inputTensor };
            const results = await this.session.run(feeds);
            
            const outputNames = Object.keys(results);
            const tracks = [];

            // If there is only a single output tensor, check if it contains multiple channels (e.g., Bird-MixIT shape [1, 4, samples])
            if (outputNames.length === 1) {
                const name = outputNames[0];
                const tensor = results[name];
                if (tensor && tensor.data instanceof Float32Array) {
                    const dims = tensor.dims;
                    if (dims && dims.length >= 2) {
                        const numChannels = dims[dims.length - 2];
                        const sampleLength = dims[dims.length - 1];
                        if (numChannels > 1 && numChannels < 10 && numChannels * sampleLength === tensor.data.length) {
                            console.log(`[ONNXAudioSeparator] Single output tensor detected with ${numChannels} channels of length ${sampleLength}. Splitting...`);
                            for (let c = 0; c < numChannels; c++) {
                                const start = c * sampleLength;
                                const end = start + sampleLength;
                                tracks.push(new Float32Array(tensor.data.subarray(start, end)));
                            }
                        }
                    }
                }
            }

            // Fallback to reading separate output nodes (if the single tensor split wasn't applicable)
            if (tracks.length === 0) {
                for (const name of outputNames) {
                    const tensor = results[name];
                    if (tensor && tensor.data instanceof Float32Array) {
                        tracks.push(tensor.data);
                    }
                }
            }

            // Check for silent/NaN outputs (silent failure of WebGPU execution provider)
            let maxOutputVal = 0;
            let containsNaN = false;
            for (const track of tracks) {
                for (let i = 0; i < track.length; i++) {
                    const val = track[i];
                    if (isNaN(val)) {
                        containsNaN = true;
                        break;
                    }
                    const abs = Math.abs(val);
                    if (abs > maxOutputVal) maxOutputVal = abs;
                }
                if (containsNaN) break;
            }

            if (containsNaN || maxOutputVal < 0.00001) {
                const reason = containsNaN ? "NaN values in output" : `output is silent (peak: ${maxOutputVal})`;
                console.warn(`[ONNXAudioSeparator] ONNX execution returned invalid output: ${reason}. Triggering fallback/reload...`);
                throw new Error(`ONNX execution returned invalid tensor data: ${reason}`);
            }

            if (tracks.length > 0) {
                return tracks;
            } else {
                throw new Error("No valid Float32Array tracks found in ONNX outputs.");
            }
        } catch (err) {
            console.error(`[ONNXAudioSeparator] ONNX run failed:`, err);

            // Release the broken session
            try { this.session.release(); } catch (_) {}
            this.session = null;

            // Attempt to reload the session with WASM-only execution before giving up
            if (this.modelPath && !this._wasmRetried) {
                this._wasmRetried = true;
                console.log(`[ONNXAudioSeparator] Attempting WASM-only session reload for model: ${this.modelPath}`);
                self.postMessage({
                    type: "PIPELINE_STATUS",
                    payload: {
                        status: "webgpu_fallback",
                        message: "WebGPU execution failed. Reloading model with WASM (CPU) backend...",
                        error: err.message
                    }
                });
                try {
                    this.session = await ort.InferenceSession.create(this.modelPath, {
                        executionProviders: ['wasm']
                    });
                    console.log(`[ONNXAudioSeparator] WASM-only session loaded successfully. Retrying separation...`);
                    // Retry with the new WASM session
                    return await this.separate(audioBuffer);
                } catch (reloadErr) {
                    console.error(`[ONNXAudioSeparator] WASM reload also failed. Permanently downgrading to DSP:`, reloadErr);
                    this.hasFailedToLoad = true;
                    this.session = null;
                    return this.dspSeparate(audioBuffer);
                }
            } else {
                // Already tried WASM reload or no model path — permanent DSP fallback
                console.error(`[ONNXAudioSeparator] Permanently downgrading to DSP separation.`);
                this.hasFailedToLoad = true;

                self.postMessage({
                    type: "PIPELINE_STATUS",
                    payload: {
                        status: "webgpu_fallback",
                        message: "ONNX execution failed on all backends. Pipeline B is running on CPU (DSP fallback).",
                        error: err.message
                    }
                });

                return this.dspSeparate(audioBuffer);
            }
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

// Initialization is triggered via 'INIT' message from the main thread

async function init(separator, modelPath, runOnGPU, outputSampleRate, lang, pipelineBWindow) {
    separator = separator || 'biocppnet';
    modelPath = modelPath || '';
    runOnGPU = runOnGPU !== false;
    if (outputSampleRate !== undefined) {
        currentOutputSampleRate = outputSampleRate;
    }
    if (pipelineBWindow !== undefined) {
        currentPipelineBWindow = pipelineBWindow;
    }
    try {
        await tf.setBackend('cpu');

        self.postMessage({
            type: "PIPELINE_STATUS",
            payload: {
                status: "loading",
                message: "Loading metadata labels..."
            }
        });

        // 1. Load Labels
        await loadLabels(lang);

        self.postMessage({
            type: "PIPELINE_STATUS",
            payload: {
                status: "loading",
                message: `Loading separator model (${separator})...`
            }
        });

        // 2. Load Separator Model
        if (separator === 'dsp') {
            console.log("[Segmentation Worker] Configured to use DSP crossover filter directly.");
            separatorModel = new ONNXAudioSeparator("");
            separatorModel.hasFailedToLoad = true;
        } else if (separator === 'custom') {
            console.log("[Segmentation Worker] Awaiting custom model bytes from main thread...");
            // Custom model loader will instantiate separatorModel and set isReady when bytes arrive
        } else {
            const separatorModelPath = modelPath || (prefix + 'models/biocppnet.onnx');
            separatorModel = new ONNXAudioSeparator(separatorModelPath, runOnGPU);
            await separatorModel.loadModel();
        }

        self.postMessage({
            type: "PIPELINE_STATUS",
            payload: {
                status: "loading",
                message: "Loading BirdNET classifier model..."
            }
        });

        // 3. Load TFLite Classification Model
        console.log("[Segmentation Worker] Loading BirdNET FP32 model...");
        classificationModel = await tflite.loadTFLiteModel(MODEL_PATH, { numThreads: 1 });
        console.log("[Segmentation Worker] BirdNET FP32 model loaded successfully.");

        self.postMessage({
            type: "PIPELINE_STATUS",
            payload: {
                status: "loading",
                message: "Warming up classifier model..."
            }
        });

        // Warmup classification model
        try {
            tf.tidy(() => {
                const dummyInput = tf.zeros([1, WINDOW_SAMPLES], 'float32');
                classificationModel.predict(dummyInput);
                dummyInput.dispose();
            });
        } catch (warmupErr) {
            console.error("[Segmentation Worker] Warmup failed:", warmupErr);
            throw warmupErr;
        }

        // If we are not waiting for custom model bytes, we are ready!
        if (separator !== 'custom') {
            isReady = true;
            console.log("[Segmentation Worker] Initialization complete and ready.");
            self.postMessage({
                type: "PIPELINE_STATUS",
                payload: {
                    status: "ready",
                    message: "Segmentation Worker loaded and warmed up successfully."
                }
            });
        }
    } catch (err) {
        console.error("[Segmentation Worker] Initialization failed:", err);
        self.postMessage({
            type: "PIPELINE_STATUS",
            payload: {
                status: "error",
                message: "Initialization failed: " + err.message
            }
        });
    }
}

async function loadLabels(langOverride) {
    const supportedLanguages = [
        'af', 'da', 'en_us', 'fr', 'ja', 'no', 'ro', 'sl', 'tr', 'ar', 'de', 'es', 'hu',
        'ko', 'pl', 'ru', 'sv', 'uk', 'cs', 'en_uk', 'fi', 'it', 'nl', 'pt', 'sk', 'th', 'zh'
    ];
    
    const lang = (() => {
        if (langOverride) return langOverride;
        return 'en_us';
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

function upsampleLinear(sourceBuffer, targetBuffer, sourceFs, targetFs) {
    const sourceLength = sourceBuffer.length;
    const targetLength = targetBuffer.length;
    if (sourceLength === 0 || targetLength === 0) return;
    
    const ratio = sourceLength / targetLength;
    for (let i = 0; i < targetLength; i++) {
        const srcIndex = i * ratio;
        const baseIndex = Math.floor(srcIndex);
        const nextIndex = Math.min(sourceLength - 1, baseIndex + 1);
        const safeBaseIndex = Math.min(sourceLength - 1, baseIndex);
        const fraction = srcIndex - baseIndex;
        
        targetBuffer[i] = sourceBuffer[safeBaseIndex] * (1 - fraction) + sourceBuffer[nextIndex] * fraction;
    }
}

/* ==========================================================================
   5. MESSAGE HANDLING & PROCESS SEGMENT
   ========================================================================== */

const segmentQueue = [];
let isProcessingSegment = false;

async function processNextSegment() {
    if (isProcessingSegment) return;
    if (segmentQueue.length === 0) return;

    isProcessingSegment = true;
    const { segmentId, timestamp, sampleRate, audioBuffer, meta } = segmentQueue.shift();

    try {
        await handleProcessSegment(segmentId, timestamp, sampleRate, audioBuffer, meta);
    } catch (err) {
        console.error("[Segmentation Worker] Critical error in segment processing pipeline:", err);
    } finally {
        isProcessingSegment = false;
        // Schedule next segment processing
        setTimeout(processNextSegment, 0);
    }
}

async function handleProcessSegment(segmentId, timestamp, sampleRate, audioBuffer, meta) {
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
        const isDSPFallback = separatorModel && separatorModel.hasFailedToLoad;
        const modelSR = isDSPFallback ? SAMPLE_RATE : (currentOutputSampleRate || SAMPLE_RATE);
        
        // Symmetrical padding/truncation to ensure constant input shape for WebGPU (fixes JSEP compilation crash)
        const targetLength = Math.round((currentPipelineBWindow || 9.0) * modelSR);
        let separationInput = audioBuffer;
        let originalSeparationLength = audioBuffer.length;
        
        // Resample input audio from SAMPLE_RATE (48000) to model native rate if needed
        if (modelSR !== SAMPLE_RATE) {
            const targetLen = Math.round(audioBuffer.length * (modelSR / SAMPLE_RATE));
            const resampled = new Float32Array(targetLen);
            upsampleLinear(audioBuffer, resampled, SAMPLE_RATE, modelSR);
            separationInput = resampled;
            originalSeparationLength = targetLen;
            console.log(`[Segmentation Worker] Resampled input from ${SAMPLE_RATE}Hz (${audioBuffer.length} samples) to ${modelSR}Hz (${targetLen} samples) before separation.`);
        } else {
            originalSeparationLength = audioBuffer.length;
        }

        // Pad or truncate separationInput to exactly targetLength to prevent dynamic WebGPU compiles
        if (separationInput.length !== targetLength) {
            const paddedInput = new Float32Array(targetLength);
            if (separationInput.length < targetLength) {
                paddedInput.set(separationInput);
                console.log(`[Segmentation Worker] Padded separation input from ${separationInput.length} to target ${targetLength} samples (constant shape for WebGPU).`);
            } else {
                paddedInput.set(separationInput.subarray(0, targetLength));
                console.log(`[Segmentation Worker] Truncated separation input from ${separationInput.length} to target ${targetLength} samples.`);
            }
            separationInput = paddedInput;
        }

        // Step 1: Run source separation
        const isolatedTracks = await separatorModel.separate(separationInput);
        console.log(`[Segmentation Worker] Separation returned ${isolatedTracks.length} tracks.`);

        const results = [];
        
        // Step 2: Classify each separated track
        for (let tIdx = 0; tIdx < isolatedTracks.length; tIdx++) {
            let track = isolatedTracks[tIdx];
            
            // Restore original resampled length by slicing or padding
            if (track.length !== originalSeparationLength) {
                if (originalSeparationLength < targetLength) {
                    track = track.subarray(0, originalSeparationLength);
                } else {
                    const restored = new Float32Array(originalSeparationLength);
                    restored.set(track);
                    track = restored;
                }
            }
            
            // Resample track from model native rate to 48000Hz if needed
            let processedTrack = track;
            if (modelSR !== SAMPLE_RATE) {
                const targetLen = Math.round(track.length * (SAMPLE_RATE / modelSR));
                const resampled = new Float32Array(targetLen);
                upsampleLinear(track, resampled, modelSR, SAMPLE_RATE);
                processedTrack = resampled;
                console.log(`[Segmentation Worker] Resampled track from ${modelSR}Hz (${track.length} samples) to ${SAMPLE_RATE}Hz (${targetLen} samples).`);
            }
            
            // Peak normalize the track to normal range [-0.8, 0.8] for the classifier
            let maxVal = 0;
            for (let i = 0; i < processedTrack.length; i++) {
                const abs = Math.abs(processedTrack[i]);
                if (abs > maxVal) maxVal = abs;
            }
            if (maxVal > 0.0001) {
                const gain = 0.8 / maxVal;
                const clampedGain = Math.min(20.0, gain); // up to 20x gain boost
                console.log(`[Segmentation Worker] Normalizing track ${tIdx} for classifier. Peak: ${maxVal.toFixed(4)}, Gain: ${clampedGain.toFixed(2)}x`);
                for (let i = 0; i < processedTrack.length; i++) {
                    processedTrack[i] = processedTrack[i] * clampedGain;
                }
            }
            
            const trackLen = processedTrack.length;

            // Frame the track audio into 3-second slices with 1.5s hop size
            const numFrames = Math.max(1, Math.ceil(Math.max(0, trackLen - WINDOW_SAMPLES) / HOP_SAMPLES) + 1);
            const framed = new Float32Array(numFrames * WINDOW_SAMPLES);
            for (let f = 0; f < numFrames; f++) {
                const start = f * HOP_SAMPLES;
                const srcEnd = Math.min(start + WINDOW_SAMPLES, trackLen);
                framed.set(processedTrack.subarray(start, srcEnd), f * WINDOW_SAMPLES);
            }

            // Run inference frame by frame
            const predictionList = [];
            for (let f = 0; f < numFrames; f++) {
                const slice = framed.subarray(f * WINDOW_SAMPLES, (f + 1) * WINDOW_SAMPLES);
                const audioTensor = tf.tensor2d(slice, [1, WINDOW_SAMPLES], 'float32');
                const resTensor = classificationModel.predict(audioTensor);
                const predictions = await resTensor.array();
                
                // Apply sigmoid to convert raw logits to probabilities
                const probabilities = predictions[0].map(val => 1 / (1 + Math.exp(-val)));
                predictionList.push(probabilities);
                audioTensor.dispose();
                resTensor.dispose();
            }

            // Pool prediction results using Max Pooling (robust for transient bird calls in isolated channels)
            const numClasses = predictionList[0]?.length || 0;
            const pooledPredictions = new Float32Array(numClasses);
            for (let i = 0; i < numClasses; i++) {
                let maxVal = -Infinity;
                for (let f = 0; f < numFrames; f++) {
                    if (predictionList[f] && predictionList[f][i] > maxVal) {
                        maxVal = predictionList[f][i];
                    }
                }
                pooledPredictions[i] = maxVal;
            }

            // Map pooled predictions to sorted objects
            const formattedPredictions = Array.from(pooledPredictions)
                .map((confidence, idx) => ({
                    speciesCode: birds[idx] ? birds[idx].scientificName : `class_${idx}`,
                    commonName: birds[idx] ? (birds[idx].commonNameI18n || birds[idx].commonName) : `Class ${idx}`,
                    scientificName: birds[idx] ? birds[idx].scientificName : `Class ${idx}`,
                    confidence: confidence
                }))
                .sort((a, b) => b.confidence - a.confidence)
                .slice(0, 10); // Keep top 10

            console.log(`[Segmentation Worker] Track ${tIdx} top predictions:`, 
                formattedPredictions.slice(0, 3).map(p => `${p.commonName} (${(p.confidence*100).toFixed(3)}%)`).join(', ')
            );

            results.push({
                channelId: tIdx,
                label: `separated_channel_${tIdx}`,
                audioBuffer: processedTrack,
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

self.onmessage = async (event) => {
    const { type, payload } = event.data || {};

    if (type === "INIT") {
        const { separator, modelPath, runOnGPU, outputSampleRate, lang, logServerUrl, pipelineBWindow } = payload || {};
        if (logServerUrl) {
            setupRemoteLogging(logServerUrl);
        }
        await init(separator, modelPath, runOnGPU, outputSampleRate, lang, pipelineBWindow);
        return;
    }

    if (type === "UPDATE_CONFIG") {
        const { outputSampleRate } = payload || {};
        if (outputSampleRate !== undefined) {
            currentOutputSampleRate = outputSampleRate;
            console.log(`[Segmentation Worker] Runtime config updated: currentOutputSampleRate = ${currentOutputSampleRate}`);
        }
        return;
    }

    if (type === "SET_MODEL_BYTES") {
        const { modelBytes } = payload;
        if (modelBytes) {
            try {
                separatorModel = new ONNXAudioSeparator("");
                await separatorModel.loadModelFromBytes(modelBytes);
                isReady = true;
                console.log("[Segmentation Worker] Custom local model bytes loaded successfully. Worker ready.");
                self.postMessage({
                    type: "PIPELINE_STATUS",
                    payload: {
                        status: "ready",
                        message: "Custom local model bytes loaded successfully. Worker ready."
                    }
                });
            } catch (err) {
                console.error("[Segmentation Worker] Failed to load custom local model bytes:", err);
                self.postMessage({
                    type: "PIPELINE_STATUS",
                    payload: {
                        status: "error",
                        message: "Failed to load custom local model bytes: " + err.message
                    }
                });
            }
        }
        return;
    }
    
    if (type === "PROCESS_SEGMENT") {
        segmentQueue.push(payload);
        processNextSegment();
        return;
    }
};

function setupRemoteLogging(url) {
  const originalLog = console.log;
  const originalWarn = console.warn;
  const originalError = console.error;

  console.log = (...args) => {
    originalLog.apply(console, args);
    fetch(`${url}/api/log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: 'Pipeline B (Worker)', type: 'info', message: args.map(String).join(' ') })
    }).catch(() => {});
  };
  console.warn = (...args) => {
    originalWarn.apply(console, args);
    fetch(`${url}/api/log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: 'Pipeline B (Worker)', type: 'warning', message: args.map(String).join(' ') })
    }).catch(() => {});
  };
  console.error = (...args) => {
    originalError.apply(console, args);
    fetch(`${url}/api/log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: 'Pipeline B (Worker)', type: 'error', message: args.map(String).join(' ') })
    }).catch(() => {});
  };
}
