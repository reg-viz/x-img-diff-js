#!/usr/bin/python
import os, sys, re, json, shutil
import argparse
from subprocess import Popen, PIPE, STDOUT


#   parse the command-line options
parser = argparse.ArgumentParser()
parser.add_argument( "--wasm", action="store_true", help="Create a .wasm file (WebAssembly format, experimental) instead of asm.js format." )
clArguments = parser.parse_args()

# Startup
exec(open(os.path.expanduser('~/.emscripten'), 'r').read())

EMSCRIPTEN_ROOT = os.getenv('EMSCRIPTEN')
# try:
#     EMSCRIPTEN_ROOT
# except:
#     print "ERROR: Missing EMSCRIPTEN_ROOT (which should be equal to emscripten's root dir) in ~/.emscripten"
#     sys.exit(1)

#Popen('source ' + emenv)
sys.path.append(EMSCRIPTEN_ROOT)
import tools.shared as emscripten

# Settings
'''
          Settings.INLINING_LIMIT = 0
          Settings.DOUBLE_MODE = 0
          Settings.PRECISE_I64_MATH = 0
          Settings.CORRECT_SIGNS = 0
          Settings.CORRECT_OVERFLOWS = 0
          Settings.CORRECT_ROUNDINGS = 0
'''

# For Debug
#emcc_args = '-O0 -g0 --closure 0 --llvm-lto 1 -s NO_EXIT_RUNTIME=1 -s ASSERTIONS=0 -s AGGRESSIVE_VARIABLE_ELIMINATION=0 -s NO_DYNAMIC_EXECUTION=0 --memory-init-file 0 -s NO_FILESYSTEM=0'.split(' ')

# For Release
emcc_args = '-O3 --llvm-lto 1 -s NO_EXIT_RUNTIME=1 -s ASSERTIONS=0 -s AGGRESSIVE_VARIABLE_ELIMINATION=0 -s NO_DYNAMIC_EXECUTION=0 --memory-init-file 0 -s NO_FILESYSTEM=0'.split(' ')

print
print '--------------------------------------------------'
print 'Building opencv.js, build type:', emcc_args
print '--------------------------------------------------'
print


stage_counter = 0
def stage(text):
    global stage_counter
    stage_counter += 1
    text = 'Stage %d: %s' % (stage_counter, text)
    print
    print '=' * len(text)
    print text
    print '=' * len(text)
    print

# Main
try:
    this_dir = os.getcwd()
    os.chdir('opencv')
    if not os.path.exists('build'):
        os.makedirs('build')
    os.chdir('build')

    stage('OpenCV Configuration')
    configuration = ['cmake',
                     '-DCMAKE_BUILD_TYPE=RELEASE',
                     '-DBUILD_DOCS=OFF',
                     '-DBUILD_EXAMPLES=OFF',
                     '-DBUILD_PACKAGE=OFF',
                     '-DBUILD_WITH_DEBUG_INFO=OFF',
                     '-DBUILD_opencv_bioinspired=OFF',
                     '-DBUILD_opencv_calib3d=OFF',
                     '-DBUILD_opencv_cuda=OFF',
                     '-DBUILD_opencv_cudaarithm=OFF',
                     '-DBUILD_opencv_cudabgsegm=OFF',
                     '-DBUILD_opencv_cudacodec=OFF',
                     '-DBUILD_opencv_cudafeatures2d=OFF',
                     '-DBUILD_opencv_cudafilters=OFF',
                     '-DBUILD_opencv_cudaimgproc=OFF',
                     '-DBUILD_opencv_cudaoptflow=OFF',
                     '-DBUILD_opencv_cudastereo=OFF',
                     '-DBUILD_opencv_cudawarping=OFF',
                     '-DBUILD_opencv_gpu=OFF',
                     '-DBUILD_opencv_gpuarithm=OFF',
                     '-DBUILD_opencv_gpubgsegm=OFF',
                     '-DBUILD_opencv_gpucodec=OFF',
                     '-DBUILD_opencv_gpufeatures2d=OFF',
                     '-DBUILD_opencv_gpufilters=OFF',
                     '-DBUILD_opencv_gpuimgproc=OFF',
                     '-DBUILD_opencv_gpuoptflow=OFF',
                     '-DBUILD_opencv_gpustereo=OFF',
                     '-DBUILD_opencv_gpuwarping=OFF',
                     '-BUILD_opencv_hal=OFF',
                     '-DBUILD_opencv_highgui=OFF',
                     '-DBUILD_opencv_java=OFF',
                     '-DBUILD_opencv_legacy=OFF',
                     '-DBUILD_opencv_ml=ON',
                     '-DBUILD_opencv_nonfree=OFF',
                     '-DBUILD_opencv_optim=OFF',
                     '-DBUILD_opencv_photo=ON',
                     '-DBUILD_opencv_shape=ON',
                     '-DBUILD_opencv_objdetect=ON',
                     '-DBUILD_opencv_softcascade=OFF',
                     '-DBUILD_opencv_stitching=OFF',
                     '-DBUILD_opencv_superres=OFF',
                     '-DBUILD_opencv_ts=OFF',
                     '-DBUILD_opencv_videostab=OFF',
                     '-DENABLE_PRECOMPILED_HEADERS=OFF',
                     '-DWITH_1394=OFF',
                     '-DWITH_CUDA=OFF',
                     '-DWITH_CUFFT=OFF',
                     '-DWITH_EIGEN=OFF',
                     '-DWITH_FFMPEG=OFF',
                     '-DWITH_GIGEAPI=OFF',
                     '-DWITH_GSTREAMER=OFF',
                     '-DWITH_GTK=OFF',
                     '-DWITH_JASPER=OFF',
                     '-DWITH_JPEG=OFF',
                     '-DWITH_OPENCL=OFF',
                     '-DWITH_OPENCLAMDBLAS=OFF',
                     '-DWITH_OPENCLAMDFFT=OFF',
                     '-DWITH_OPENEXR=OFF',
                     '-DWITH_PNG=OFF',
                     '-DWITH_PVAPI=OFF',
                     '-DWITH_TIFF=OFF',
                     '-DWITH_LIBV4L=OFF',
                     '-DWITH_WEBP=OFF',
                     '-DWITH_PTHREADS_PF=OFF',
                     '-DBUILD_opencv_apps=OFF',
                     '-DBUILD_PERF_TESTS=OFF',
                     '-DBUILD_TESTS=OFF',
                     '-DBUILD_SHARED_LIBS=OFF',
                     '-DWITH_IPP=OFF',
                     '-DENABLE_SSE=OFF',
                     '-DENABLE_SSE2=OFF',
                     '-DENABLE_SSE3=OFF',
                     '-DENABLE_SSE41=OFF',
                     '-DENABLE_SSE42=OFF',
                     '-DENABLE_AVX=OFF',
                     '-DENABLE_AVX2=OFF',
                     '-DCMAKE_CXX_FLAGS=%s' % ' '.join(emcc_args),
                     '-DCMAKE_EXE_LINKER_FLAGS=%s' % ' '.join(emcc_args),
                     '-DCMAKE_CXX_FLAGS_DEBUG=%s' % ' '.join(emcc_args),
                     '-DCMAKE_CXX_FLAGS_RELWITHDEBINFO=%s' % ' '.join(emcc_args),
                     '-DCMAKE_C_FLAGS_RELWITHDEBINFO=%s' % ' '.join(emcc_args),
                     '-DCMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO=%s' % ' '.join(emcc_args),
                     '-DCMAKE_MODULE_LINKER_FLAGS_RELEASE=%s' % ' '.join(emcc_args),
                     '-DCMAKE_MODULE_LINKER_FLAGS_DEBUG=%s' % ' '.join(emcc_args),
                     '-DCMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO=%s' % ' '.join(emcc_args),
                     '-DCMAKE_SHARED_LINKER_FLAGS_RELEASE=%s' % ' '.join(emcc_args),
                     '-DCMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO=%s' % ' '.join(emcc_args),
                     '-DCMAKE_SHARED_LINKER_FLAGS_DEBUG=%s' % ' '.join(emcc_args),
                     '..']
    emscripten.Building.configure(configuration)


    stage('Making OpenCV')

    emcc_args += ('-s TOTAL_MEMORY=%d' % (128*1024*1024)).split(' ') # default 128MB.
    emcc_args += '-s ALLOW_MEMORY_GROWTH=1'.split(' ')  # resizable heap
    emcc_args += '-s EXPORT_NAME="cv"'.split(' ')

    emscripten.Building.make(['make', '-j4'])

    stage('Generating Bindings')
    INCLUDE_DIRS = [
             os.path.join('..', 'include'),
             os.path.join('..', 'modules', 'core', 'include'),
             os.path.join('..', 'modules', 'flann', 'include'),
             os.path.join('..', 'modules', 'ml', 'include'),
             os.path.join('..', 'modules', 'photo', 'include'),
             os.path.join('..', 'modules', 'shape', 'include'),
             os.path.join('..', 'modules', 'imgproc', 'include'),
             os.path.join('..', 'modules', 'calib3d', 'include'),
             os.path.join('..', 'modules', 'features2d', 'include'),
             os.path.join('..', 'modules', 'video', 'include'),
             os.path.join('..', 'modules', 'videoio', 'include'),
             os.path.join('..', 'modules', 'objdetect', 'include'),
             os.path.join('..', 'modules', 'imgcodecs', 'include'),
             os.path.join('..', 'modules', 'hal', 'include'),
             os.path.join('..', 'build'),
             os.path.join('..', '..', 'x-img-diff', 'src'),
    ]
    include_dir_args = ['-I'+item for item in INCLUDE_DIRS]
    emcc_binding_args = ['--bind']
    emcc_binding_args += include_dir_args

    emscripten.Building.emcc('../../bindings/bindings.cpp', emcc_binding_args, 'bindings.bc')
    emscripten.Building.emcc('../../x-img-diff/src/rectutil.cpp', emcc_binding_args, 'rectutil.bc')
    emscripten.Building.emcc('../../x-img-diff/src/hunter.cpp', emcc_binding_args, 'hunter.bc')
    assert os.path.exists('bindings.bc')

    stage('Building for Browser module')

    if clArguments.wasm:
        emcc_args += "-s WASM=1".split( " " )
        basename = "cv-wasm"
    else:
        basename = "cv"

    destBrowser = os.path.join('..', '..', 'build', basename + '_browser.js')
    destNode = os.path.join('..', '..', 'build', basename + '_node.js')

    input_files = [
                'bindings.bc',
                'hunter.bc',
                'rectutil.bc',
                os.path.join('lib','libopencv_core.a'),
                os.path.join('lib','libopencv_imgproc.a'),
                # os.path.join('lib','libopencv_imgcodecs.a'),

                # os.path.join('lib','libopencv_ml.a'),
                # os.path.join('lib','libopencv_flann.a'),
                # os.path.join('lib','libopencv_objdetect.a'),
                os.path.join('lib','libopencv_features2d.a') ,

                # os.path.join('lib','libopencv_shape.a'),
                # os.path.join('lib','libopencv_photo.a'),
                # os.path.join('lib','libopencv_video.a'),

                # external libraries
                # os.path.join('3rdparty', 'lib', 'liblibjpeg.a'),
                # os.path.join('3rdparty', 'lib', 'liblibpng.a'),
                # os.path.join('3rdparty', 'lib', 'libzlib.a'),
                #os.path.join('3rdparty', 'lib', 'liblibtiff.a'),
                #os.path.join('3rdparty', 'lib', 'liblibwebp.a'),
                ]

    emscripten.Building.link(input_files, 'ximgdiff.bc')
    #emcc_args += '--preload-file ../../test/data/'.split(' ') #For testing purposes
    emcc_args += ['--bind', '--post-js', '../../bindings/post.js']
    #emcc_args += ['--memoryprofiler']
    #emcc_args += ['--tracing']      #   ability to use custom memory profiler, with hooks Module.onMalloc(), .onFree() and .onRealloc()

    emscripten.Building.emcc('ximgdiff.bc', emcc_args, destBrowser)

    stage('Building for Node.js module')
    emcc_node_args = emcc_args
    emcc_node_args += '--pre-js ../../bindings/pre_node.js'.split(' ')
    emscripten.Building.emcc('ximgdiff.bc', emcc_node_args, destNode)

finally:
    os.chdir(this_dir)
