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
    }

    /**
     * Push incoming PCM block into the circular buffer.
     * @param {Float32Array} pcmData - Block of float audio samples
     */
    inputData(pcmData) {
        this.ringBuffer.write(pcmData);
    }

    /**
     * Start the dual-dispatch interval loops.
     */
    start() {
        if (this.isRouting) return;
        this.isRouting = true;
        this.segmentCounter = 0;

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

        // Pipeline B (Segmented): Sends last 9 seconds of audio every 4.5 seconds
        const segmentedWindowSamples = this.sampleRate * 9;
        this.intervalBId = setInterval(() => {
            if (!this.isRouting || !this.segmentationWorker) return;

            // MOBILE FIX: Skip dispatch if the worker is still processing
            // the previous segment (backpressure).
            if (this.segmentBusy) {
                console.log("[AudioRouter] Pipeline B busy, skipping segment dispatch.");
                return;
            }

            // Require at least 9 seconds of audio to start segmentation
            if (this.ringBuffer.getSamplesWritten() < segmentedWindowSamples) return;

            const pcm = this.ringBuffer.readLast(segmentedWindowSamples);
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

        }, 4500);
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
