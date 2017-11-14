#!/bin/bash

mkdir -p dist
cp index.html dist
cp -rf build dist
./node_modules/.bin/gh-pages -d dist

