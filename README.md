<div align="center">
    <h1>MapClosures</h1>
    <a href="https://github.com/PRBonn/MapClosures/releases"><img src="https://img.shields.io/github/v/release/PRBonn/MapClosures?label=version" /></a>
    <a href="https://github.com/PRBonn/MapClosures/blob/main/LICENSE"><img src=https://img.shields.io/badge/license-MIT-green" /></a>
    <a href="https://github.com/PRBonn/MapClosures/blob/main/"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <br />
    <br />
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://github.com/PRBonn/MapClosures/blob/main/README.md#Install">Install</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href=https://www.ipb.uni-bonn.de/pdfs/gupta2024icra.pdf>Paper</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href=https://github.com/PRBonn/MapClosures/issues>Contact Us</a>
  <br />
  <br />

Effectively Detecting Loop Closures using Point Cloud Density Maps.

<p align="center">

![image](https://github.com/PRBonn/MapClosures/assets/28734882/18d5ee54-61a9-4d9f-87f2-8aba16de0f75)
</p>
</div>
<hr />

## Install

### Dependencies
- *Essentials*
    ```sh
    sudo apt-get install --no-install-recommends -y build-essential cmake pybind11-dev python3-dev python3-pip
    ```

- *Optionally Built* In this case you have two options:
  - **Option 1**: You can install them by the package manager in your operative system, e.g. in Ubuntu 22.04:
      ```sh
      sudo apt-get install libeigen3-dev libopencv-dev libtbb-dev
      ```
      this will of course make the build of **MapClosures** much faster.
  - **Option 2**: Let the build system handle them:
      ```sh
      cmake -B build -S cpp -DUSE_SYSTEM_TBB=OFF -DUSE_SYSTEM_OPENCV=OFF -DUSE_SYSTEM_EIGEN3=OFF
      ```
      this will be slower in terms of build time, but will enable you to have a different version of this libraries installed in your system without interfering with the build of **MapClosures**.

Once the dependencies are installed, the C++ library can be build by using standard cmake commands:
```sh
cmake -B build -S cpp
cmake --build build -j8
```

## Python Package
We provide a _python_ wrapper for **MapClosures** which can be easily installed by simply running:
```sh
make
```
### Usage
<details>
<summary>
The following command will provide details about how to use our pipeline:

```sh
map_closure_pipeline --help
```
</summary>

![CLI_usage](https://github.com/PRBonn/MapClosures/assets/28734882/6dc885d2-e0fc-4aa4-b5b0-be8a98ed6ff9)
</details>


## Citation

If you use this library for any academic work, please cite our original [paper](https://www.ipb.uni-bonn.de/pdfs/gupta2024icra.pdf).

```bibtex
@inproceedings{gupta2024icra,
    author     = {S. Gupta and T. Guadagnino and B. Mersch and I. Vizzo and C. Stachniss},
    title      = {{Effectively Detecting Loop Closures using Point Cloud Density Maps}},
    booktitle  = {IEEE International Conference on Robotics and Automation (ICRA)},
    year       = {2024},
    codeurl    = {https://github.com/PRBonn/MapClosures},
}
```
### Paper Results
As we decided to continue the development of **MapClosures** beyond the scope of the ICRA paper, we created a ``git tag`` so that researchers can consistently reproduce the results of the publication. To checkout at this tag, you can run the following:
```sh
git checkout ICRA2024
```
Our development aims to push the performances of **MapClosures** above the original results of the paper.

## Acknowledgement

This repository is heavily inspired by, and also depends on [KISS-ICP](https://github.com/PRBonn/kiss-icp)

## Contributors

<a href="https://github.com/PRBonn/MapClosures/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PRBonn/MapClosures" />
</a>
