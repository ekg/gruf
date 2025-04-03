# minGRU

PyTorch implementation of the minGRU architecture from the paper [Were RNNs All We Needed?](https://arxiv.org/abs/2410.01201v1), using the log-space numerically stable version.

## Features

- Efficient training with DeepSpeed integration via `learn.py`
- Text generation with efficient KV-caching via `generate.py`
- Support for CPU offloading and mixed precision training
- Support for configurable model sizes via dimension and depth parameters

## Environment Setup

### Using Micromamba (Recommended)

Micromamba is a fast, lightweight package manager compatible with conda environments:

1. Install Micromamba if you don't have it:
   ```bash
   # For Linux
   curl -L -O "https://micro.mamba.pm/api/micromamba/linux-64/latest"
   chmod +x micromamba
   ./micromamba shell init -s bash -p ~/micromamba
   source ~/.bashrc  # Or reopen your terminal
   ```

2. Create and activate the environment:
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/minGRU.git
   cd minGRU
   
   # Create environment from the yml file
   micromamba env create -f environment.yml
   micromamba activate cuda_ok
   ```

3. Verify the environment:
   ```bash
   # Check that PyTorch can access your GPU
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"
   ```

### Using Conda

You can also use regular conda if preferred:

```bash
conda env create -f environment.yml
conda activate cuda_ok
```

## Training

Train a model on your text data:

```bash
python learn.py --data your_text_file.txt --params 15m --gpus 0,1
```

For distributed training with DeepSpeed:

```bash
deepspeed learn.py --data your_text_file.txt --zero_stage 2 --params 19.9m
```

## Generation

Generate text with a trained model:

```bash
python generate.py --model checkpoints/best.pt --primer_text "Once upon a time" --temperature 0.8
```

### Common Generation Options

```bash
# Use lower temperature for more focused text
python generate.py --model checkpoints/best.pt --temperature 0.7 --top_k 0.95

# Generate longer text samples
python generate.py --model checkpoints/best.pt --generation_length 1024

# Use memory-efficient generation for large models
python generate.py --model checkpoints/best.pt --memory_efficient --no_kv_cache
```

## Citation

```bibtex
@inproceedings{Feng2024WereRA,
    title   = {Were RNNs All We Needed?},
    author  = {Leo Feng and Frederick Tung and Mohamed Osama Ahmed and Yoshua Bengio and Hossein Hajimirsadegh},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273025630}
}
```

Original implementation by the authors of the paper.
