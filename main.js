'use strict';

function showSpinner(msg) {
  document.querySelector('.spin-wrap').classList.remove('hide');
  document.querySelector('.spin-message').innerHTML = msg || 'loading...';
}

function hideSpinner(msg) {
  document.querySelector('.spin-wrap').classList.add('hide');
  document.querySelector('body').classList.remove('loading');
}

function showSpinnerBtn() {
  document.querySelector('#btn').classList.add('loading');
}

function hideSpinnerBtn() {
  document.querySelector('#btn').classList.remove('loading');
}

function handleDragover(e) {
  e.stopPropagation();
  e.preventDefault();
  e.dataTransfer.dropEffect = 'copy'; // Explicitly show this is a copy.
}

function handleDropFile(canvasId, e) {
  e.stopPropagation();
  e.preventDefault();
  var files = e.dataTransfer.files; // FileList object.
  var file = files[0];
  var reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onload = function(){
    loadImage(reader.result, canvasId);
  }
}

function getInput(canvasId) {
  var canvas = document.querySelector('#' + canvasId);
  var ctx = canvas.getContext('2d');
  var { width, height, data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return { width, height, data };
}

function loadImage(path, canvasId) {
  const img = new Image();
  return new Promise(resolve => {
    img.onload = () => {
      const canvas = document.querySelector('#' + canvasId);
      const ctx = canvas.getContext('2d');
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      resolve();
    };
    img.src = path
  });
}

function showImage({ channels, data, width, height }, canvasId){
  const canvas = document.querySelector('#' + canvasId);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  canvas.width = width;
  canvas.height = height;
  const imgData = ctx.createImageData(width, height);
  for (let i = 0, j = 0; i < data.length; i += channels, j += 4) {
    imgData.data[j] = data[i];
    imgData.data[j + 1] = data[i + 1 % channels];
    imgData.data[j + 2] = data[i + 2 % channels];
    imgData.data[j + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
}

function rendarRects(rects, canvasId, style, width = 1) {
  const canvas = document.querySelector('#' + canvasId);
  const ctx = canvas.getContext("2d");
  ctx.strokeStyle = style;
  ctx.lineWidth = width;
  rects.forEach(r => {
    ctx.strokeRect(r.x, r.y, r.width, r.height);
  });
}

function renderResult(result, canvasId1, canvasId2) {
  result.matches.forEach(m => {
    rendarRects(m[0].diffMarkers, canvasId1, 'rgb(255, 0, 0)', 2);
    rendarRects(m[1].diffMarkers, canvasId2, 'rgb(255, 0, 0)', 2);
  });
  rendarRects(result.matches.map(m => m[0].bounding), canvasId1, 'rgb(0, 255, 255)')
  rendarRects(result.matches.map(m => m[1].bounding), canvasId2, 'rgb(0, 255, 255)')
  rendarRects(result.strayingRects[0], canvasId1, 'rgb(255, 0, 255)', 2)
  rendarRects(result.strayingRects[1], canvasId2, 'rgb(255, 0, 255)', 2)
}

function startCalc() {
  const img1 = getInput('img1'), img2 = getInput('img2');
  worker.postMessage({ type: 'req_match', img1, img2 }, [img1.data.buffer, img2.data.buffer]);
}

const worker = new Worker('worker.js');

Promise.all([
  loadImage('test/img/actual.png', 'img1'),
  loadImage('test/img/expected.png', 'img2'),
]).then(() => {
  worker.addEventListener('message', (ev) => {
    const meta = ev.data;
    switch (meta.type) {
      case 'init':
        showSpinner('Calculating...');
        startCalc();
        document.querySelector('.output').classList.remove('hide');
        break;
      case 'res_match':
        hideSpinner();
        hideSpinnerBtn();
        renderResult(ev.data.result, 'img1', 'img2');
      default:
    }
  });
});

document.querySelector('#btn').addEventListener('click', () => {
  showSpinnerBtn();
  startCalc();
});

for (let node of document.querySelectorAll('.input > div')){
  const canvasId = node.querySelector('canvas').id;
  node.addEventListener('dragover', handleDragover);
  node.addEventListener('drop', handleDropFile.bind(null, canvasId));
}
