#/bin/bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"

conda env create -f env.yml
conda activate mmm

pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
echo "Installing flash-attention, this can take up to 1h!"
MAX_JOBS=4 pip install flash-attn==2.8.3 --no-build-isolation
echo "The 'flash'-attention wait is over, installing last packages."
pip install triton==3.6.0 accelerate==1.11.0
# vLLM requires CUDA 13 runtime; expose Ollama's bundled libcudart.so.13.
export LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v13:${LD_LIBRARY_PATH:-}
pip install vllm
echo "Time to celebrate, it is done!"