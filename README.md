# x-img-diff-js
[![CircleCI](https://circleci.com/gh/reg-viz/x-img-diff-js.svg?style=svg)](https://circleci.com/gh/reg-viz/x-img-diff-js)


Image comparison module(Highly experimental).

## Demonstration
See https://reg-viz.github.io/x-img-diff-js/

## Usage
### Node.js

```sh
npm install x-img-diff-js pngjs
```

```javascript
const fs = require('fs');
const path = require('path');
const PNG = require('pngjs').PNG;

const detectDiff = require('x-img-diff-js');

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
    console.log("diff result:", diffResult);
    console.log("mathied area:", diffResult.matches.length);
    console.log("img1's matched area bounding rect:", diffResult.matches[0][0].bounding);
    console.log("ima2's matched area bounding rect:", diffResult.matches[0][1].bounding);
    console.log("diff marker rectangulars in img1's matched area", diffResult.matches[0][0].diffMarkers.length);
  })
  ;
}

test()
  .then(() => process.exit(0))
  .catch(() => process.exit(1));
```

### Browser
*T.B.D.*

### API
*T.B.D.*


## How to build module

1. Clone this repo and change the current directory to it.

2. Get OpenCV source code

 ```
 git clone https://github.com/opencv/opencv.git
 cd opencv
 git checkout 3.1.0
 cd ..
 ```

3. Get x-img-diff source code

 ```
 git clone https://github.com/quramy/x-img-diff.git
 ```

4. Execute docker

```sh
$ docker-compose build
$ docker-compose run emcc
```

## Run module in your local machine

```
python -mhttp.server
open http://localhost:8000/index.html
```

## License
MIT.
