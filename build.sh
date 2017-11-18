#/bin/bash

if [ -f "build/cv-wasm_browser.js" \
  -a -f "build/cv-wasm_browser.wasm" \
  -a -f "build/cv-wasm_node.js" \
  -a -f "build/cv-wasm_node.wasm" ] ;then
  echo "Modules are already built, so skip building..."
  exit 0
fi

python make.py --wasm
