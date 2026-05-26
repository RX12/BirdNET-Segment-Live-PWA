<div align="center"><img width="300" alt="BirdNET+ logo" src="public/img/birdnet-logo-circle.png"></div>

**BirdNET-Segment-Live-PWA** is an experimental fork of the official [birdnet-team/real-time-pwa](https://github.com/birdnet-team/real-time-pwa). It uses the high analysis power of the BirdNET algorithm directly to browser-based environments and introduces a **dual-pipeline audio segmentation engine**. This architecture isolates overlapping bird vocalizations and filters environmental noise in near real-time, drastically reducing false positives while providing isolated post-detection playback. 

This fork is an independent hobby project developed with the assistance of AI coding agents. Features, performance, and stability may vary; **please use it at your own risk!**

## **Key Features**

* **Dual-Pipeline Audio Engine**:  
  * **Pipeline A (Live Radar)**: Feeds continuous 3-second rolling-window chunks to a fast inference engine for real-time "Pending" UI feedback.  
  * **Pipeline B (Segmented & Verified)**: Periodically takes longer buffers (9 seconds) and executes high-fidelity source separation to isolate overlapping birds and noise.  
* **On-Device Source Separation**: Integrates a modular separation interface (e.g., Demucs v4 or BioCPPNet) to split mixed audio into isolated channels, re-analyzing each with BirdNET FP32 to resolve overlapping calls.  
* **Post-Detection Isolated Playback**: Listen to isolated channels for individual species directly on the UI card to verify songs without wind, traffic, or other background noise.  
* **WebGPU Hardware Acceleration**: Fully optimized for modern mobile silicon (Apple A17 Pro+, Google Tensor G4+) using onnxruntime-web/webgpu to run heavy separation networks inside a background Web Worker.  
* **Highly Precise FP32 Inference**: Runs the full-precision BirdNET\_GLOBAL\_6K\_V2.4\_Model\_FP32.tflite model locally on the browser client for ultimate taxonomic accuracy.  
* **External Integration Links**: Instant redirect links to Wikipedia, eBird, and iNaturalist integrated directly into the verified species cards.  
* **100% Offline & Private**: Once model assets are cached, all computing happens on-device. No telemetry, voice data, or GPS metrics ever leave the user's browser.

## **Technical Pipeline Architecture**

The application splits computational loads across multi-threaded Web Workers to secure a buttery smooth 60 FPS user interface:

                  \[ Microphone Input (WebAudio API) \]  
                                   │  
                         \[ Main Thread Router \]  
                                   │  
                      ┌────────────┴────────────┐  
         (Every 1s)   │                         │ (Every 4.5s)  
                      ▼                         ▼  
         \[ Pipeline A: Live Worker \]     \[ Pipeline B: Segmentation Worker \]  
          \- BirdNET FP32 Model            \- AudioSeparator (ONNX WebGPU)  
          \- Fast 3s Inference             \- Demixing / Stem Extraction  
                        │                 \- Re-infer each isolated channel  
                        │                                 │  
                        ▼                                 ▼  
              \[ Pending Species UI \] \<───► \[ Consensus Engine / Validator \]  
                                                          │  
                                                          ▼  
                                              \[ Verified Species UI \]  
                                              \[ Isolated Playback Button \]

## **Setup & Local Development**

### **Prerequisites**

* **Node.js** (v18 or higher recommended)  
* A browser supporting WebGPU (e.g., Chrome 121+, Safari 17.4+ on iOS, Edge)

### **Installation**

1. Clone fork:  
   git clone \[https://github.com/\](https://github.com/)\<your-github-username\>/BirdNET-Segment-Live-PWA.git  
   cd BirdNET-Segment-Live-PWA

2. Install dependencies:  
   npm install

3. Fire up the local development server:  
   npm run serve

4. Navigate to http://localhost:8080 in your browser.

**Note**: For mobile testing, serve via HTTPS or configure your mobile browser to treat your local IP address as a secure origin to bypass microphone permission blocks.

## **Deploying to GitHub Pages**

The application is fully static-hosted. You can build and deploy directly to GitHub Pages:

1. Configure the path settings in your bundler configuration.  
2. Run the build script:  
   npm run build

3. Push the distribution folder to your gh-pages branch. Once deployed, users can visit https://\<your-github-username\>.github.io/BirdNET-Segment-Live-PWA/ and install it directly on their home screen as an offline PWA.

## **⚖️ Licensing & Compliance**

This fork strictly complies with the original codebase licenses:

* **Fork-Specific Modifications**: © 2026 BirdNET-Segment-Live-PWA Contributors. Released under the [MIT License](https://opensource.org/licenses/MIT).  
* **Original UI & Audio Setup**: The foundational PWA framework is copyrighted by the BirdNET Team and licensed under the [MIT License](https://opensource.org/licenses/MIT).  
* **Deep Learning Weights**: The BirdNET neural network model weights are provided under the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

Please ensure you review and adhere to the license parameters of any custom segmentation or bioacoustic models packed or loaded into Pipeline B.

## **Citation**

If you use BirdNET for academic analysis or conservation research, please cite the official paper:

@article{kahl2021birdnet,  
  title={BirdNET: A deep learning solution for avian diversity monitoring},  
  author={Kahl, Stefan and Wood, Connor M and Eibl, Maximilian and Klinck, Holger},  
  journal={Ecological Informatics},  
  volume={61},  
  pages={101236},  
  year={2021},  
  publisher={Elsevier}  
}

## **Funding & Acknowledgments**

This project builds directly on the robust bioacoustic tools established by the **K. Lisa Yang Center for Conservation Bioacoustics** at the Cornell Lab of Ornithology. Their work is made possible by the generosity of K. Lisa Yang to advance conservation technologies that protect global ecosystems.

The development of the core BirdNET algorithm is supported by:

* The German Federal Ministry of Research, Technology and Space (FKZ 01|S22072)  
* The German Federal Ministry for the Environment, Climate Action, Nature Conservation and Nuclear Safety (FKZ 67KI31040E)  
* The German Federal Ministry of Economic Affairs and Energy (FKZ 16KN095550)  
* The Deutsche Bundesstiftung Umwelt (project 39263/01)  
* The European Social Fund
