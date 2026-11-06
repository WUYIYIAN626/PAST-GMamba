# PAST-GMamba

Official implementation of the manuscript

**PAST-GMamba: Persistence-Anchored Parallel Spatio-Temporal Graph Mamba for Multi-Site Photovoltaic Power Forecasting**

The code will be made public after the paper is published. 

## Architecture

![Overall architecture of PAST-GMamba](overall-architecture.png)

## Requirements

Python 3.10–3.12 and PyTorch 2.4 or later.

| File | Contents |
|---|---|
| `requirements.txt` | Runtime dependencies |
| `requirements-dev.txt` | Runtime dependencies and pytest |
| `requirements-mamba.txt` | Runtime dependencies, `mamba-ssm==2.2.5`, and `causal-conv1d==1.5.2` |

## Installation

```bash
conda create -n past-gmamba python=3.11 -y
conda activate past-gmamba
pip install -r requirements.txt
pip install -e .
