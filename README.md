# minGRU

PyTorch implementation of the minGRU architecture from the paper [Were RNNs All We Needed?](https://arxiv.org/abs/2410.01201v1), using the log-space numerically stable version.

## Features

- Efficient training with DeepSpeed integration via `learn.py`
- Text generation with efficient KV-caching via `generate.py`
- Support for CPU offloading and mixed precision training
- Support for configurable model sizes via dimension and depth parameters

## Training

Train a model on your text data:

```bash
python learn.py --data your_text_file.txt --params 15m --gpus 0,1
```

For distributed training with DeepSpeed:

```bash
python learn.py --data your_text_file.txt --deepspeed --zero_stage 2 --params 19.9m --gpus 0,1,2,3
```

## Generation

Generate text with a trained model:

```bash
python generate.py --model checkpoints/best.pt --primer_text "Once upon a time" --temperature 0.8
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
