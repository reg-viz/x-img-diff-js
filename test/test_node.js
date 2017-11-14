const fs = require('fs');
const path = require('path');

function load() {
  fs.readFile(path.resolve(__dirname, '..', 'build', 'cv-wasm.wasm'), (err, buf) => {
    if (err) {
      console.error(err);
      return;
    }
    const arrayBuf = new Uint8Array(buf).buffer;

    WebAssembly.compile(arrayBuf).then(module => {
      const imports = {
        env: {
          memoryBase: 0,
          tableBase: 0,
          memory: new WebAssembly.Memory({
            initial: 256,
          }),
          table: new WebAssembly.Table({
            initial: 0,
            element: 'anyfunc',
          }),
        },
      }
      const instance = new WebAssembly.Instance(module, imports)
      console.log(instance.exports)
    });
  });
};

load();
