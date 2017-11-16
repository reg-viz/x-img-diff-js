const fs = require('fs');
const path = require('path');
const PNG = require('pngjs').PNG;
const assert = require('assert');

const detectDiff = require('../src');

function decodePng(filename) {
  return new Promise((resolve, reject) => {
    try {
      fs.createReadStream(filename).pipe(new PNG())
        .on("parsed", function() {
          const { width, height, data } = this;
          resolve({
            width,
            height,
            data: new Uint8Array(data),
          });
        })
        .on("error", function(err) {
          reject(err);
        })
      ;
    } catch (e) {
      reject(e);
    }
  });
}

function test() {
  return Promise.all([
    decodePng(path.resolve(__dirname, '../demo/img/actual.png')),
    decodePng(path.resolve(__dirname, '../demo/img/expected.png')),
  ])
  .then(([img1, img2]) => detectDiff(img1, img2, { }))
  .then((diffResult) => {
    assert(diffResult);
    console.log("diff result:", diffResult);
    console.log("the number of matching area:", diffResult.matches.length);
    console.log("img1's macthing area bounding rect:", diffResult.matches[0][0].bounding);
    console.log("ima2's matching area bounding rect:", diffResult.matches[0][1].bounding);
    console.log("diff marker rectangulars in img1's matching area", diffResult.matches[0][0].diffMarkers.length);
  })
  ;
}

test()
  .then(() => process.exit(0))
  .catch(() => process.exit(1));
