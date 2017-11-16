importScripts('wasm-util.js', 'module.js', 'build/cv-wasm_browser.js');

addEventListener('message', (ev) => {
  const meta = ev.data;
  switch (meta.type) {
    case 'req_match':
      const { img1, img2 } = ev.data;
      const diffResult = Module.detectDiff(Module, img1, img2, {
        _debug: true,
      });
      postMessage({ type: 'res_match', result: diffResult });
      break;
    default:
  }
});

Module.onInit(cv => {
  postMessage({ type: 'init' });
});
