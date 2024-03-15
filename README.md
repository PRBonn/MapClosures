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
    <a href=https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/gupta2024icra.pdf>Paper</a>
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
- Essentials
    ```sh
    sudo apt-get install --no-install-recommends -y build-essential ccache clang-format git cmake pybind11-dev
    ```
- Python
    ```sh
    sudo apt-get install --no-install-recommends -y python3 python3-numpy python3-pip
    pip3 install --upgrade pip
    pip3 install --upgrade numpy
    pip3 install kiss-icp
    ```
- OpenCV
    ```sh
    git clone --depth 1 https://github.com/opencv/opencv.git -b 4.x \
    cd opencv && mkdir build && cd build
    cmake .. && make -j$(nproc --all) && make install
    ```
### MapClosures
```sh
git clone https://github.com/PRBonn/MapClosures.git
cd MapClosures
make install
```

## Install (Docker)

```sh
git clone https://github.com/PRBonn/MapClosures.git
cd MapClosures

# Build
docker build . -t mapclosures:latest

# Run docker image with GUI support (dataset is volume mounted)
docker run -it \
    --privileged \
    -v $(pwd)/results/:/MapClosures/results \
    -v {DATASET_PATH}:/MapClosures/dataset \
    mapclosures:latest
```

## Usage
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

If you use this library for any academic work, please cite our original [paper](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/gupta2024icra.pdf).

```bibtex
@inproceedings{gupta2024icra,
    author     = {S. Gupta and T. Guadagnino and B. Mersch and I. Vizzo and C. Stachniss},
    title      = {{Effectively Detecting Loop Closures using Point Cloud Density Maps}},
    booktitle  = {IEEE International Conference on Robotics and Automation (ICRA)},
    year       = {2024},
    codeurl    = {https://github.com/PRBonn/MapClosures},
}
```

## Acknowledgement

This repository is heavily inspired by, and also depends on [KISS-ICP](https://github.com/PRBonn/kiss-icp)
