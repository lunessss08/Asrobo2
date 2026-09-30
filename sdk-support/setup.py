from setuptools import setup, Extension
import pybind11
setup(name='libmedia_codec', version='0.0.1', ext_modules=[Extension(
    'libmedia_codec', ['media_codec.cpp'],
    include_dirs=[pybind11.get_include(), '/opt/homebrew/opt/ffmpeg@4/include', '/opt/homebrew/opt/opus/include'],
    library_dirs=['/opt/homebrew/opt/ffmpeg@4/lib', '/opt/homebrew/opt/opus/lib'],
    libraries=['avcodec', 'avutil', 'swscale', 'opus'],
    language='c++', extra_compile_args=['-std=c++11'])])
