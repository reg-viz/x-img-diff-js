FROM trzeci/emscripten:sdk-tag-1.37.21-64bit

RUN emsdk install clang-e1.37.21-64bit
RUN emsdk activate clang-e1.37.21-64bit

# Patch binding header and .js
COPY ./patch_emscripten.diff /src/patch_emscripten.diff
RUN patch -p0 -d /emsdk_portable/sdk/ < /src/patch_emscripten.diff
