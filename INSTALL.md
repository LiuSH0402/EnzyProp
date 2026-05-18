# EnzyProp Environment Installation

Recommended environment name:

```bash
conda env create -f environment.yml
conda activate web_pre
```

If you prefer creating the environment manually:

```bash
conda create -n web_pre python=3.10 pip -y
conda activate web_pre
pip install -r requirements.txt
pip install -e .
```

Check the command:

```bash
enzyprop --help
```

Train command:

```bash
enzyprop train --config configs/topt_train.yaml
```

CPU inference is supported. The model still depends on PyTorch Geometric because
the trained checkpoints include the GNN channel.

Put trained EnzyProp model files in:

```text
models/
  topt_best_model.pt
  phopt_best_model.pt
  pi_best_model.pt
```

Put or cache HuggingFace base models in:

```text
hf_cache/
```
