const m = require('../build/cv-wasm_node.js');
const p = new Promise(resolve => {
  const id = setInterval(() => {
    if (m._detectDiff) {
      resolve(m);
      clearInterval(id);
      return;
    }
  }, 10);
});

module.exports = function (img1, img2, config, cb) {
  return p.then(m => {
    return m.detectDiff(m, img1, img2, config);
  });
};

