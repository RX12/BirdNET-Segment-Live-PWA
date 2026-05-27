const http = require('http');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');

// Color constants
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  red: '\x1b[31m',
  gray: '\x1b[90m'
};

// Help menu
function showHelp() {
  console.log(`
${colors.bright}BirdNET PWA Programmatic Pipeline Test Runner${colors.reset}

Usage:
  node scripts/run-pipeline-test.js <audio-file-path> [options]

Options:
  --separator <model>    Separator model to use: 'dsp', 'bird_mixit_4source', 'biocppnet' (default: 'bird_mixit_4source')
  --precision <prec>     Model precision: 'fp32', 'fp16', 'int8' (default: 'fp32')
  --webgpu <true/false>  Enable/disable WebGPU acceleration (default: 'true')
  --aad <true/false>     Enable/disable Acoustic Activity Gate (default: 'true')
  --early-exit <t/f>     Enable/disable early exit (default: 'true')
  --threshold <float>    Confidence threshold (default: '0.15')
  --help                 Show this help menu
`);
  process.exit(0);
}

// Parse command line arguments
const args = process.argv.slice(2);
if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
  showHelp();
}

const audioFilePath = args[0];
if (!fs.existsSync(audioFilePath)) {
  console.error(`${colors.red}Error: Audio file not found at path: ${audioFilePath}${colors.reset}`);
  process.exit(1);
}

// Default query param values
let separator = 'bird_mixit_4source';
let precision = 'fp32';
let webgpu = 'true';
let aad = 'true';
let earlyExit = 'true';
let threshold = '0.15';

// Parse options
for (let i = 1; i < args.length; i++) {
  if (args[i] === '--separator' && args[i + 1]) {
    separator = args[i + 1];
    i++;
  } else if (args[i] === '--precision' && args[i + 1]) {
    precision = args[i + 1];
    i++;
  } else if (args[i] === '--webgpu' && args[i + 1]) {
    webgpu = args[i + 1];
    i++;
  } else if (args[i] === '--aad' && args[i + 1]) {
    aad = args[i + 1];
    i++;
  } else if (args[i] === '--early-exit' && args[i + 1]) {
    earlyExit = args[i + 1];
    i++;
  } else if (args[i] === '--threshold' && args[i + 1]) {
    threshold = args[i + 1];
    i++;
  }
}

const siteDir = path.join(__dirname, '../_site');
const testWavPath = path.join(siteDir, 'test.wav');

// Create _site if it doesn't exist
if (!fs.existsSync(siteDir)) {
  console.error(`${colors.red}Error: _site directory not found. Please run 'npm run build' first.${colors.reset}`);
  process.exit(1);
}

// Copy input audio to _site/test.wav
console.log(`${colors.gray}Copying input file to pipeline test directory...${colors.reset}`);
fs.copyFileSync(audioFilePath, testWavPath);

// MIME type map for static serving
const mimeTypes = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'application/javascript',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
  '.json': 'application/json',
  '.png': 'image/png',
  '.webp': 'image/webp',
  '.txt': 'text/plain',
  '.wav': 'audio/wav',
  '.mp3': 'audio/mpeg'
};

const port = 8085;

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${port}`);
  
  // API endpoints
  if (req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk.toString(); });
    req.on('end', () => {
      if (url.pathname === '/api/log') {
        try {
          const { source, type, message } = JSON.parse(body);
          let prefixColor = colors.gray;
          if (source.includes('Pipeline A')) prefixColor = colors.cyan;
          else if (source.includes('Pipeline B')) prefixColor = colors.magenta;
          else if (source === 'Main') prefixColor = colors.blue;
          else if (source.includes('Consensus')) prefixColor = colors.green;

          let typeColor = colors.reset;
          if (type === 'error') typeColor = colors.red;
          else if (type === 'warning') typeColor = colors.yellow;

          console.log(`${prefixColor}[${source}]${colors.reset} ${typeColor}${message}${colors.reset}`);
        } catch (e) {
          console.error('Error logging:', body);
        }
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'ok' }));
      }
      else if (url.pathname === '/api/save_separated') {
        try {
          const { species, channelId, sampleRate, audioData } = JSON.parse(body);
          const dir = path.join(__dirname, '../debug_output');
          if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir);
          }
          
          const filename = `${species.replace(/[^a-zA-Z0-9]/g, '_')}_channel_${channelId}.wav`;
          const filepath = path.join(dir, filename);
          
          // Encode to 16-bit PCM WAV
          const bufferLength = audioData.length;
          const header = Buffer.alloc(44);
          
          header.write('RIFF', 0);
          header.writeUInt32LE(36 + bufferLength * 2, 4);
          header.write('WAVE', 8);
          header.write('fmt ', 12);
          header.writeUInt32LE(16, 16);
          header.writeUInt16LE(1, 20); // raw PCM
          header.writeUInt16LE(1, 22); // mono
          header.writeUInt32LE(sampleRate, 24);
          header.writeUInt32LE(sampleRate * 2, 28);
          header.writeUInt16LE(2, 32);
          header.writeUInt16LE(16, 34);
          header.write('data', 36);
          header.writeUInt32LE(bufferLength * 2, 40);
          
          const pcmData = Buffer.alloc(bufferLength * 2);
          for (let i = 0; i < bufferLength; i++) {
            const s = Math.max(-1, Math.min(1, audioData[i]));
            const val = s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7FFF);
            pcmData.writeInt16LE(val, i * 2);
          }
          
          fs.writeFileSync(filepath, Buffer.concat([header, pcmData]));
          console.log(`${colors.green}[System] Saved isolated separation for ${species} to debug_output/${filename}${colors.reset}`);
        } catch (e) {
          console.error('Error saving separated audio:', e);
        }
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'ok' }));
      }
      else if (url.pathname === '/api/error') {
        try {
          const { error } = JSON.parse(body);
          console.error(`\n${colors.red}${colors.bright}[CRITICAL PIPELINE ERROR]${colors.reset} ${colors.red}${error}${colors.reset}\n`);
        } catch (e) {
          console.error('Error payload:', body);
        }
        cleanupAndExit(1);
      }
      else if (url.pathname === '/api/complete') {
        try {
          const { results } = JSON.parse(body);
          console.log(`\n${colors.bright}${colors.green}==================================================`);
          console.log(`              DETECTION SCAN COMPLETE`);
          console.log(`==================================================${colors.reset}`);
          
          if (!results || results.length === 0) {
            console.log(`No species detected above confidence threshold.`);
          } else {
            // Sort by confidence descending
            results.sort((a, b) => b.confidence - a.confidence);
            results.forEach(d => {
              const status = d.verified 
                ? `${colors.green}[VERIFIED]${colors.reset}` 
                : `${colors.yellow}[DRAFT]   ${colors.reset}`;
              const pct = (d.confidence * 100).toFixed(1) + '%';
              console.log(` ${status} ${colors.bright}${d.commonName.padEnd(25)}${colors.reset} (${d.scientificName.padEnd(30)}) - Confidence: ${colors.green}${pct}${colors.reset}`);
            });
          }
          console.log(`${colors.green}==================================================${colors.reset}\n`);
        } catch (e) {
          console.error('Complete payload parse error:', body);
        }
        cleanupAndExit(0);
      }
    });
    return;
  }

  // Serve static files from _site
  let filePath = path.join(siteDir, url.pathname);
  
  // Route /test-pipeline/ to test-pipeline/index.html
  if (url.pathname === '/test-pipeline' || url.pathname === '/test-pipeline/') {
    filePath = path.join(siteDir, 'test-pipeline/index.html');
  }

  // If path is directory, try index.html
  if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {
    filePath = path.join(filePath, 'index.html');
  }

  if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
    const ext = path.extname(filePath).toLowerCase();
    const contentType = mimeTypes[ext] || 'application/octet-stream';
    
    // Add WASM/WebGPU specific headers if necessary
    res.writeHead(200, {
      'Content-Type': contentType,
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp'
    });
    
    fs.createReadStream(filePath).pipe(res);
  } else {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('404 Not Found');
  }
});

function cleanupAndExit(code) {
  try {
    if (fs.existsSync(testWavPath)) {
      fs.unlinkSync(testWavPath);
    }
  } catch (e) {}
  process.exit(code);
}

server.listen(port, '127.0.0.1', () => {
  console.log(`${colors.green}Test runner server listening on http://127.0.0.1:${port}${colors.reset}`);
  
  // Format query params
  const params = new URLSearchParams({
    separator,
    precision,
    webgpu,
    aad,
    earlyExit,
    threshold
  });

  const runUrl = `http://127.0.0.1:${port}/test-pipeline/?${params.toString()}`;
  console.log(`\n${colors.bright}${colors.cyan}>>> ACTION REQUIRED <<<${colors.reset}`);
  console.log(`${colors.gray}Please click or open the following link in your browser to run the test:${colors.reset}`);
  console.log(`${colors.bright}${colors.blue}${runUrl}${colors.reset}\n`);
});
