<h1 align="center">
  <b>cosSquareFormer-PyTorch</b><br>
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/python-3.7-blue.svg" /></a>
       <a href= "https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-1.9-FF0000.svg" /></a>
       <a href= "https://github.com/davidsvy/cosformer-pytorch/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/license-MIT-white.svg" /></a>
</p>

This implementation of cosSquareFormer is based on the PyTorch implementation of the cosFormer model by [davidsvy](https://github.com/davidsvy/cosformer-pytorch) (Thank you!!!).


Table of contents
===

<!--ts-->
  * [➤ Installation](#installation)
  * [➤ Usage](#usage)
  * [➤ Citations](#citations)
<!--te-->

Installation
===
```
$ git clone https://github.com/sayannag/cossquareformer-pytorch
$ cd cossquareformer-pytorch
$ pip install -r requirements.txt
```

Usage
===

```python
from models.cos_square_former import cosSquareFormer
import torch

model = cosSquareFormer(
    # Linear attention args:
    use_cos=True,         # Whether to use the cosine reweighting mechanism proposed in the paper.
    kernel='relu',        # Kernel that approximates softmax. Available options are 'relu' and 'elu'.
    denom_eps=1e-5,       # Added to the denominator of linear attention for numerical stability.
    # If use_cos=True & kernel='relu' the model is equivalent to https://arxiv.org/pdf/2109.04572.pdf
    # If use_cos=False & kernel='elu' the model is equivalent to https://arxiv.org/pdf/2006.16236.pdf
    # Vanilla transformer args:
    d_model=512,
    n_heads=8, 
    n_layers=6,
    n_emb=20000, 
    ffn_ratio=4, 
    rezero=True,          # If True, use the ReZero architecture from https://arxiv.org/pdf/2003.04887.pdf, else the Pre-LN architecture from https://arxiv.org/pdf/2002.04745.pdf
    ln_eps=1e-5, 
    bias=False, 
    dropout=0.2, 
    max_len=1024, 
    xavier=True
)

input_ids = torch.randint(0, 20000, [4, 100])
lengths = torch.randint(1, 100, [4])
attention_mask = torch.arange(100)[None, :] < lengths[:, None]

output = model(
    input_ids=input_ids,
    lengths=lengths,
    attention_mask=attention_mask,
)
```


Citations
===

```
Deciphering Environmental Air Pollution with Large Scale City Data. In IJCAI 2022
```

