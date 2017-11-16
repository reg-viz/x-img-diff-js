"use strict";

(function () {

  function r2r(rect) {
    return {
      x: rect.x,
      y: rect.y,
      width: rect.width,
      height: rect.height
    };
  }

  function convertRvector(rvec) {
    var ret = [];
    for (var i = 0; i < rvec.size(); i++) {
      ret.push(r2r(rvec.get(i)));
    }
    return ret;
  }

  function detectDiff(cv, img1array, img2array, conf) {
    var img1Raw = cv.matFromArray(img1array, 24),
        img1 = new cv.Mat();
    cv.cvtColor(img1Raw, img1, cv.ColorConversionCodes.COLOR_RGBA2RGB.value, 0);

    var img2Raw = cv.matFromArray(img2array, 24),
        img2 = new cv.Mat();
    cv.cvtColor(img2Raw, img2, cv.ColorConversionCodes.COLOR_RGBA2RGB.value, 0);

    var config = new cv.DiffConfig();
    if (conf) {
      Object.keys(conf).forEach(function (k) {
        config[k] = conf[k];
      });
    }
    var r = new cv.DiffResult();
    cv._detectDiff(img1, img2, r, config);
    var result = {
      matches: [],
      strayingRects: [convertRvector(r.strayingRects1), convertRvector(r.strayingRects2)]
    };
    for (var i = 0; i < r.matches.size(); i++) {
      var m = r.matches.get(i);
      var obj = [{
        center: r2r(m.center1),
        bounding: r2r(m.bounding1),
        diffMarkers: convertRvector(m.diffMarkers1)
      }, {
        center: r2r(m.center2),
        bounding: r2r(m.bounding2),
        diffMarkers: convertRvector(m.diffMarkers2)
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

    [img1Raw, img2Raw, img1, img2, config, r].forEach(function (m) {
      return m.delete();
    });
    return result;
  }

  Module.detectDiff = detectDiff;
})();

