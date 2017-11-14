# x-img-diff-js

## How to Build

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

## Run demo

```
python -mhttp.server
open http://localhost:8000/index.html
```

## License
MIT.
