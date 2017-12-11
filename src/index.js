const path = require('path');
const fs = require('fs');

let m;
const p = new Promise(resolve => {
  m = require('../build/cv-wasm_node.js');
  const id = setInterval(() => {
    if (m._detectDiff) {
      resolve(m);
      clearInterval(id);
      return;
    }
  }, 10);
});

function detectDiff (img1, img2, config, cb) {
  return p.then(m => {
    return m.detectDiff(m, img1, img2, config);
  });
}

detectDiff.getBrowserJsPath = function() {
  return path.resolve(__dirname, '..', 'build', 'cv-wasm_browser.js');
};

detectDiff.getBrowserWasmPath = function() {
  return path.resolve(__dirname, '..', 'build', 'cv-wasm_browser.wasm');
};

module.exports = detectDiff;
