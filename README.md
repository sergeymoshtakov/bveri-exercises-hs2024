# Exercises for bverI Deep Learning (Part 2) - HS 2024

This repository is used for the distribution of exercises for bverI Deep Learning (Part 2).


There are several ways to work on the assignments:

- Google Colab (easiest)
- pip  (local install)
- Docker (only linux/amd64 available)


## Google Colab

Use Google Colab by clicking on the links below.

**Note**: After running the notebook on a compute instance it may gget stuck on the "import" cell. In that case restart the kernel and run again. It should work then.


### Exercise 01 - PyTorch and Image Data

Click on the following badge to open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/i4Ds/bveri-exercises-hs2024/blob/main/notebooks/01_pytorch_and_images/pytorch_and_images.ipynb)



## Local

```
pip install .
```

## Docker

### 1. Install Docker on your computer

Depending on your operating system you have to install docker in different ways.  

You'll find detailed instructions here: https://docs.docker.com/get-docker


### 2. Pull the Docker image

```
# pull the image
docker login -u read_registry -p glpat-XaJ-SDxB6Qn7j_8USgAF cr.gitlab.fhnw.ch/ml/courses/bveri/hs2024/bveri-exercises-dev-hs2024
docker pull cr.gitlab.fhnw.ch/ml/courses/bveri/hs2024/bveri-exercises-dev-hs2024:latest
```

### 3. Fork this repository

Fork this repository by pressing the fork button on the upper right.

### 4. Clone your fork to your computer. 

Clone into your ml directory (`MY_ML_DIR`) using:

```
git clone MY_REPO_FORK_HTTPS_ADDRESS
```

### 5. Start a ml container on your machine

```
# Replace 'MY_ML_DIR' with your local code directory
docker run -it -d \
    -p 8881:8881 \
    -p 6007:6007 \
    --gpus=all \
    --shm-size 12G \
    -v MY_ML_DIR:/workspace/code \
    --name=bveri_hs24 \
    cr.gitlab.fhnw.ch/ml/courses/bveri/hs2024/bveri-exercises-dev-hs2024:latest
```

### 6. Check that your container is running

```
docker ps -a
```

### 7. Connect to your container through your browser

Enter `http://localhost:8881/lab` in your browser.

