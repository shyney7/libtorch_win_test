<h1 align="center">
    <img src="https://raw.githubusercontent.com/shyney7/pytorch_cpp_test/master/images/pytorch_cpp.png" width="50%">
</h1>
<p align="center">
    C++ Implementation of PyTorch
    <br />
<img src="https://img.shields.io/travis/prabhuomkar/pytorch-cpp">
<img src="https://img.shields.io/github/license/prabhuomkar/pytorch-cpp">
<img src="https://img.shields.io/badge/libtorch-1.4-ee4c2c">
<img src="https://img.shields.io/badge/cmake-3.14-064f8d">
</p>
This is just a private playground repo of my first attempts with the Pytorch C++ API.

In this example the FANN learns the negation with 10 inputs of batch size 10 and the corresponding target outputs.

### Requirements

1. [C++](http://www.cplusplus.com/doc/tutorial/introduction/)
2. [CMake](https://cmake.org/download/)
3. [LibTorch v1.4.0](https://pytorch.org/cppdocs/installing.html)
3. [MSVC Buildtools](https://docs.microsoft.com/de-de/cpp/build/building-on-the-command-line?view=vs-2019)

### Guide
1. Download the Pytorch C++ API and extract it to your Project Folder
2. Make sure that the MSVC Buildtools for C++ are installed (MinGW doesn't work!)
3. run '''cmake -B build -DCAMKE_PREFIX_PATH=path/to/libtorch/share/cmake/Torch'''
4. Build the actual source file with '''cmake --build build --config Release'''
5. The binary file will be located under /build/Release/ run with '''./build/Release/binaryname.exe'''
