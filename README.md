# TRITON Inference server - Example Configuration and scripts

Personal collection of examples and scripts for working with [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)

## Pre-requisites

- NVIDIA GPU (Compute >= 6) and NVIDIA Driver >= 510
- Docker
- Docker compose
- Python 3.8

## Python backend

Python backend in Triton Inference Server allows us to write custom python functions. It is useful in scenarios where exporting model to formats such as torchscript, onnx, savedmodel etc is unavailable / fails / operators are unsupported outside native execution.

### TLDR
```bash
# Create execution environment
conda env create -f environment.yml
conda activate triton
conda install conda-pack
conda pack # Creates triton.tar.gz

# Model
cp triton.tar.gz models/

docker-compose up serving
```

Once the server is up (should take a while to load the model), you can now use the client

```bash
pip install tritonclient[http] Pillow numpy
python3 client.py <IMAGE> --model test --url localhost:8000
```


### Longer version

This example shows how to use Detectron2 Faster RCNN model using the python backend of Triton.

[Detectron2](https://github.com/facebookresearch/detectron2) is a Pytorch based DL framework for training Object detection and segmentation models with an easy-to-use API. ([Tutorial](https://detectron2.readthedocs.io/en/latest/tutorials/index.html))

To use models trained with Detectron2 in the Python backend of Triton Server, we have to bundle these dependencies, according to these [instructions](https://github.com/triton-inference-server/python_backend#using-custom-python-execution-environments)

TODO - Explainers for
1. conda pack
2. Config.pbtxt
3. model.py
4. client.py