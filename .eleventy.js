module.exports = function (eleventyConfig) {
  // Static (public)
  eleventyConfig.addPassthroughCopy({ "public": "/" });

  // Prevent Jekyll on GitHub Pages
  eleventyConfig.addPassthroughCopy({ "src/.nojekyll": ".nojekyll" });

  // Vendor assets
  eleventyConfig.addPassthroughCopy({
    "node_modules/bootstrap/dist/css/bootstrap.min.css": "vendor/bootstrap/bootstrap.min.css",
  });
  eleventyConfig.addPassthroughCopy({
    "node_modules/bootstrap/dist/js/bootstrap.bundle.min.js": "vendor/bootstrap/bootstrap.bundle.min.js",
  });
  eleventyConfig.addPassthroughCopy({
    "node_modules/bootstrap-icons/font/bootstrap-icons.css": "vendor/bootstrap-icons/bootstrap-icons.css",
  });
  eleventyConfig.addPassthroughCopy({
    "node_modules/bootstrap-icons/font/fonts": "vendor/bootstrap-icons/fonts",
  });
  eleventyConfig.addPassthroughCopy({
    "node_modules/d3/dist/d3.min.js": "vendor/d3/d3.min.js",
  });
  eleventyConfig.addPassthroughCopy({
    "node_modules/@tensorflow/tfjs/dist/tf.min.js": "js/tfjs-4.14.0.min.js",
  });
  eleventyConfig.addPassthroughCopy({
    "node_modules/onnxruntime-web/dist/ort.min.js": "js/ort.min.js",
  });
  eleventyConfig.addPassthroughCopy({
    "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm": "onnx-wasm/ort-wasm-simd-threaded.wasm",
    "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs": "onnx-wasm/ort-wasm-simd-threaded.mjs",
    "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm": "onnx-wasm/ort-wasm-simd-threaded.jsep.wasm",
    "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs": "onnx-wasm/ort-wasm-simd-threaded.jsep.mjs",
    "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jspi.wasm": "onnx-wasm/ort-wasm-simd-threaded.jspi.wasm",
    "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jspi.mjs": "onnx-wasm/ort-wasm-simd-threaded.jspi.mjs",
    "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm": "onnx-wasm/ort-wasm-simd-threaded.asyncify.wasm",
    "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs": "onnx-wasm/ort-wasm-simd-threaded.asyncify.mjs",
  });

  return {
    pathPrefix: "/BirdNET-Segment-Live-PWA/",
    dir: { input: "src", includes: "_includes", output: "_site" },
    markdownTemplateEngine: "njk",
    htmlTemplateEngine: "njk",
    templateFormats: ["md", "njk", "html"]
  };
};
