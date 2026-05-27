/**
 * BirdNET Live - Audio Router & Ring Buffer
 * 
 * Manages the central rolling audio buffer in the main thread and handles
 * the dual-dispatch system to the Live (Pipeline A) and Segmentation (Pipeline B) workers.
 */

class RingBuffer {
    /**
     * @param {number} size - Buffer capacity in samples
     */
    constructor(size) {
        this.size = size;
        this.buffer = new Float32Array(size);
        this.writeIndex = 0;
        this.totalSamplesWritten = 0;
    }

    /**
     * Writes raw audio PCM samples into the circular buffer.
     * @param {Float32Array} data - Incoming audio buffer
     */
    write(data) {
        const len = data.length;
        for (let i = 0; i < len; i++) {
            this.buffer[this.writeIndex] = data[i];
            this.writeIndex = (this.writeIndex + 1) % this.size;
        }
        this.totalSamplesWritten += len;
    }

    /**
     * Reads the most recent samples from the circular buffer.
     * @param {number} samplesCount - Number of samples to read
     * @returns {Float32Array} - Extracted samples
     */
    readLast(samplesCount) {
        if (samplesCount > this.size) {
            throw new Error(`Requested samples count (${samplesCount}) exceeds RingBuffer capacity (${this.size})`);
        }
        
        const result = new Float32Array(samplesCount);
        let readIndex = (this.writeIndex - samplesCount + this.size) % this.size;
        
        for (let i = 0; i < samplesCount; i++) {
            // Apply gain clamping to prevent clipping artifacts in models
            result[i] = Math.max(-1.0, Math.min(1.0, this.buffer[readIndex]));
            readIndex = (readIndex + 1) % this.size;
        }
        return result;
    }

    /**
     * Returns the total number of samples written to this buffer since initialization.
     */
    getSamplesWritten() {
        return this.totalSamplesWritten;
    }
}

class AudioRouter {
    /**
     * @param {Worker} liveWorker - Worker instance for Pipeline A (Live predictions)
     * @param {Worker} segmentationWorker - Worker instance for Pipeline B (Hifi segmentation)
     * @param {Object} options - Configuration options and callbacks
     */
    constructor(liveWorker, segmentationWorker, options = {}) {
        this.liveWorker = liveWorker;
        this.segmentationWorker = segmentationWorker;
        this.sampleRate = options.sampleRate || 48000;
        
        // 15 seconds ring buffer to comfortably hold 9 seconds of audio with no read-write overlap
        this.ringBuffer = new RingBuffer(this.sampleRate * 15);
        
        this.intervalAId = null;
        this.intervalBId = null;
        this.segmentCounter = 0;
        this.isRouting = false;

        // MOBILE FIX: Backpressure flag for Pipeline B.
        // Prevents dispatching new segments while the worker is still processing
        // the previous one, avoiding memory pileup on constrained devices.
        this.segmentBusy = false;
        
        // Dynamic settings providers
        this.getSensitivity = options.getSensitivity || (() => 1.0);
        this.getGeoContext = options.getGeoContext || (() => ({}));
        this.onInferenceStart = options.onInferenceStart || (() => {});
        this.onEarlyExit = options.onEarlyExit || null;
        this.getBConfig = options.getBConfig || (() => ({
            windowSize: 9.0,
            stride: 4.5,
            gateEnabled: true,
            rmsThreshold: 0.01,
            centroidMin: 1000,
            earlyExitEnabled: true,
            earlyExitConfidence: 0.90
        }));
        this.maxConfidenceInInterval = 0;
    }

    /**
     * Push incoming PCM block into the circular buffer.
     * @param {Float32Array} pcmData - Block of float audio samples
     */
    inputData(pcmData) {
        this.ringBuffer.write(pcmData);
    }

    /**
     * Calculate Root Mean Square (RMS) of audio samples.
     */
    calculateRMS(samples) {
        let sum = 0;
        const len = samples.length;
        for (let i = 0; i < len; i++) {
            sum += samples[i] * samples[i];
        }
        return Math.sqrt(sum / len);
    }

    /**
     * Zero-crossing frequency surrogate of Spectral Centroid.
     */
    calculateSpectralCentroid(samples, sampleRate) {
        let crossings = 0;
        const len = samples.length;
        if (len <= 1) return 0;
        for (let i = 1; i < len; i++) {
            if ((samples[i] >= 0 && samples[i-1] < 0) || (samples[i] < 0 && samples[i-1] >= 0)) {
                crossings++;
            }
        }
        const duration = len / sampleRate;
        return crossings / (2 * duration);
    }

    /**
     * Set/update max confidence from live classifier to evaluate early exit.
     */
    reportLiveConfidence(confidence) {
        this.maxConfidenceInInterval = Math.max(this.maxConfidenceInInterval, confidence);
    }

    /**
     * Recreate Pipeline B setInterval loops dynamically.
     */
    updatePipelineBInterval() {
        if (!this.isRouting) return;

        if (this.intervalBId) {
            clearInterval(this.intervalBId);
            this.intervalBId = null;
        }

        const bConfig = this.getBConfig();
        const strideMs = bConfig.stride * 1000;
        const segmentedWindowSamples = this.sampleRate * bConfig.windowSize;

        console.log(`[AudioRouter] Recreating Pipeline B interval: every ${strideMs}ms with window size ${bConfig.windowSize}s (${segmentedWindowSamples} samples)`);

        this.intervalBId = setInterval(() => {
            if (!this.isRouting || !this.segmentationWorker) return;

            // MOBILE FIX: Skip dispatch if the worker is still processing the previous segment
            if (this.segmentBusy) {
                console.log("[AudioRouter] Pipeline B busy, skipping segment dispatch.");
                return;
            }

            // Require at least windowSize seconds of audio to start segmentation
            if (this.ringBuffer.getSamplesWritten() < segmentedWindowSamples) return;

            const pcm = this.ringBuffer.readLast(segmentedWindowSamples);

            // AAD Gate Check
            if (bConfig.gateEnabled) {
                const rms = this.calculateRMS(pcm);
                const centroid = this.calculateSpectralCentroid(pcm, this.sampleRate);
                
                if (rms < bConfig.rmsThreshold || centroid < bConfig.centroidMin) {
                    console.log(`[AudioRouter] Pipeline B: Bypassed via gate (rms: ${rms.toFixed(4)} < ${bConfig.rmsThreshold} OR centroid: ${Math.round(centroid)}Hz < ${bConfig.centroidMin}Hz)`);
                    return;
                }
            }

            // Early Exit Check
            if (bConfig.earlyExitEnabled && this.maxConfidenceInInterval >= bConfig.earlyExitConfidence) {
                console.log(`[AudioRouter] Pipeline B: Bypassed via early exit (max live confidence: ${(this.maxConfidenceInInterval * 100).toFixed(1)}% >= ${(bConfig.earlyExitConfidence * 100).toFixed(0)}%)`);
                this.maxConfidenceInInterval = 0; // reset for next interval
                if (this.onEarlyExit) {
                    this.onEarlyExit(pcm);
                }
                return;
            }
            this.maxConfidenceInInterval = 0; // reset for next interval

            const segmentId = `seg-${Date.now()}-${this.segmentCounter++}`;
            const geoCtx = this.getGeoContext();

            console.log(`[AudioRouter] Dispatching segment ${segmentId} to Pipeline B...`);
            this.segmentBusy = true;
            
            this.segmentationWorker.postMessage({
                type: "PROCESS_SEGMENT",
                payload: {
                    segmentId: segmentId,
                    timestamp: Date.now(),
                    sampleRate: this.sampleRate,
                    audioBuffer: pcm,
                    meta: geoCtx
                }
            }, [pcm.buffer]);

        }, strideMs);
    }

    /**
     * Start the dual-dispatch interval loops.
     */
    start() {
        if (this.isRouting) return;
        this.isRouting = true;
        this.segmentCounter = 0;
        this.maxConfidenceInInterval = 0;

        console.log("[AudioRouter] Starting dual-routing loops...");

        // Pipeline A (Live): Sends last 3 seconds of audio every 1 second
        const liveWindowSamples = this.sampleRate * 3;
        this.intervalAId = setInterval(() => {
            if (!this.isRouting || !this.liveWorker) return;
            
            // Require at least 3 seconds of audio to start inference
            if (this.ringBuffer.getSamplesWritten() < liveWindowSamples) return;

            const pcm = this.ringBuffer.readLast(liveWindowSamples);
            const geoCtx = this.getGeoContext();
            
            this.onInferenceStart();
            
            this.liveWorker.postMessage({
                message: "predict",
                pcmAudio: pcm,
                overlapSec: 1.5,
                sensitivity: this.getSensitivity(),
                ...geoCtx
            }, [pcm.buffer]);
            
        }, 1000);

        // Start Pipeline B dispatch loop
        this.updatePipelineBInterval();
    }

    /**
     * Stop and clear the dispatch loops.
     */
    stop() {
        this.isRouting = false;
        this.segmentBusy = false;
        if (this.intervalAId) {
            clearInterval(this.intervalAId);
            this.intervalAId = null;
        }
        if (this.intervalBId) {
            clearInterval(this.intervalBId);
            this.intervalBId = null;
        }
        console.log("[AudioRouter] Dual-routing loops stopped.");
    }

    /**
     * Signal that Pipeline B has finished processing its current segment.
     * Called from app.js when SEGMENT_RESULT is received.
     */
    markSegmentComplete() {
        this.segmentBusy = false;
    }
}

// Export for base.njk / app.js
window.AudioRouter = AudioRouter;
window.RingBuffer = RingBuffer;
