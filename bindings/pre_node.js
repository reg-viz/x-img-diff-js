var path = require('path');
Module.wasmBinaryFile = path.resolve(__dirname, 'cv-wasm_node.wasm');
Module.onRuntimeInitialized = function () {
  cv._init_(Module);
};
