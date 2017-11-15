
function r2r(rect) {
  return {
    x: rect.x,
    y: rect.y,
    width: rect.width,
    height: rect.height,
  };
}

function convertRvector(rvec) {
  const ret = [];
  for (let i = 0; i < rvec.size(); i++) {
    ret.push(r2r(rvec.get(i)));
  }
  return ret;
}

function ditectDiff(cv, img1array, img2array, conf) {
  const img1Raw = cv.matFromArray(img1array, 24), img1 = new cv.Mat();
  cv.cvtColor(img1Raw, img1, cv.ColorConversionCodes.COLOR_RGBA2RGB.value, 0);

  const img2Raw = cv.matFromArray(img2array, 24), img2 = new cv.Mat();
  cv.cvtColor(img2Raw, img2, cv.ColorConversionCodes.COLOR_RGBA2RGB.value, 0);

  const config = new cv.DiffConfig();
  if (conf) {
    Object.keys(conf).forEach(k => {
      config[k] = conf[k];
    });
  }
  const r = new cv.DiffResult();
  const result = {
    matches: [],
    strayingRects: [
      convertRvector(r.strayingRects1),
      convertRvector(r.strayingRects2),
    ],
  };
  cv.detectDiff(img1, img2, r, config);
  for (let i = 0; i < r.matches.size(); i++) {
    const m = r.matches.get(i);
    const obj = [{
      center: r2r(m.center1),
      bounding: r2r(m.bounding1),
      diffMarkers: convertRvector(m.diffMarkers1),
    }, {
      center: r2r(m.center2),
      bounding: r2r(m.bounding2),
      diffMarkers: convertRvector(m.diffMarkers2),
    }];
    m.center1.delete();
    m.center2.delete();
    m.bounding1.delete();
    m.bounding2.delete();
    m.diffMarkers1.delete();
    m.diffMarkers2.delete();
    m.delete();
    result.matches.push(obj);
  }
  r.matches.delete();
  r.strayingRects1.delete();
  r.strayingRects2.delete();

  // postMessage({ type: 'res_match', result });

  [img1Raw, img2Raw, img1, img2, config, r].forEach(m => m.delete());
  return result;
}

function getEmModule() {
  const cvNode = require('../build/cv-wasm_node.js');
  return new Promise((resolve, reject) => {
    try {
      cvNode({
        _init_(Module) {
          console.log('init done!');
          setTimeout(() => resolve(Module), 0);
        },
      });
    } catch (e) {
      reject(e);
    }
  });
}

const cvNode = require('../build/cv-wasm_node.js');

module.exports = function (img1, img2, config, cb) {
  return new Promise((resolve) => {
    var x;
    cvNode({ _init_: (m) => {
      x = m;
    } });
    const id = setInterval(() => {
      if (!x.detectDiff) return;
      clearInterval(id);
      const result = ditectDiff(x, img1, img2, config);
      resolve(result);
    }, 10);
  });
};

