# Feasible-Region-Iteration
Code of the paper "Synthesizing Control Barrier Functions With Feasible Region Iteration for Safe Reinforcement Learning" published on *IEEE Transactions on Automatic Control*.\
[paper](https://ieeexplore.ieee.org/document/10328440)

## Installation

```bash
# Create environemnt
mamba create -n fri python=3.9 numpy sympy jaxlib gym matplotlib scikit-learn dm-haiku numpyro optax
# tqdm tensorboard tensorboardX matplotlib scikit-learn black snakeviz ipykernel ipywidgets Cython imageio cffi fasteners
mamba activate fri
# One of: Install jax WITH CUDA
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# Or: Install jax WITHOUT CUDA
pip install --upgrade "jax[cpu]"
# Install package
pip install SumOfSquares
# pip install -r requirements.txt
pip install -e .

# Optional: Install safety-gym
pip install glfw
pip install mujoco-py==2.0.2.13 --no-cache-dir --no-binary :all: --no-build-isolation
pip install -e .  # in safety-gym
```
