# EnzyProp

EnzyProp is a command line package for enzyme property prediction. It predicts:

- `topt`: optimal temperature
- `phopt`: optimal pH
- `pi`: isoelectric point

The package takes a sequence table plus PDB structure files, calculates
three-class SASA regions when needed, and writes compact CSV prediction results.

## Folder Layout

Put trained model checkpoints here:

```text
models/
  topt_best_model.pt
  phopt_best_model.pt
  pi_best_model.pt
```

HuggingFace base models for ESM2 and ProtBert are cached here by default:

```text
hf_cache/
```

For batch prediction, a typical input folder looks like this:

```text
test/
  query.csv
  PDB/
    Q2HWU5.pdb
    D9I0I9.pdb
```

If `--sasa-dir` is not provided, SASA files are generated in a temporary folder
and deleted after prediction. If you want to keep/reuse SASA files, pass
`--sasa-dir path/to/sasa_three`.

## Input Table

The input table can be `.csv`, `.xlsx`, or `.xls`.

Minimum format:

```csv
id,sequence
Q2HWU5,WHKATVYQIYPKSFMDTNGDGIGDLKGITSKLDYLQKLGV...
```

Accepted ID columns:

```text
uniprotkb_ac, id, entry, protein_id, accession
```

Accepted sequence columns:

```text
sequence, seq, aa_seq
```

Optional target columns are allowed but not required:

```text
topt/temp/tm/temperature
phopt/ph/ph_value
pi
```

## Install

Recommended:

```bash
conda env create -f environment.yml
conda activate web_pre
```

Manual install:

```bash
conda create -n web_pre python=3.10 pip -y
conda activate web_pre
pip install -r requirements.txt
pip install -e .
```

Check:

```bash
enzyprop --help
```

## Predict One Target

If the input CSV is next to a `PDB/` folder:

```bash
enzyprop predict --target topt --input-csv F:\path\to\test\query.csv --out-dir outputs\topt --device cpu
```

Predict optimal pH:

```bash
enzyprop predict --target phopt --input-csv F:\path\to\test\query.csv --out-dir outputs\phopt --device cpu
```

Predict pI:

```bash
enzyprop predict --target pi --input-csv F:\path\to\test\query.csv --out-dir outputs\pi --device cpu
```

The old target names are still accepted as aliases:

```text
temp -> topt
ph   -> phopt
```

## Predict With One PDB File

Use this when the input table contains exactly one sequence row:

```bash
enzyprop predict --target topt --input-csv F:\path\to\test\one.csv --pdb-file F:\path\to\test\PDB\Q2HWU5.pdb --out-dir outputs\topt --device cpu
```

## Predict All Three Targets

Batch folder mode:

```bash
enzyprop predict-all --input-csv F:\path\to\test\query.csv --out-dir outputs\all --device cpu
```

Single PDB mode:

```bash
enzyprop predict-all --input-csv F:\path\to\test\one.csv --pdb-file F:\path\to\test\PDB\Q2HWU5.pdb --out-dir outputs\all --device cpu
```

Outputs:

```text
outputs/all/
  topt_result.csv
  phopt_result.csv
  pi_result.csv
  all_targets_predictions.csv
```

## Train A Model

Training is available through a config file:

```bash
enzyprop train --config configs/topt_train.yaml
```

Example configs are provided:

```text
configs/
  topt_train.yaml
  phopt_train.yaml
  pi_train.yaml
```

The training config keeps data paths and hyperparameters in one place. Important
fields are:

```yaml
target: topt
train_table: ../data/topt_train.csv
val_table: ../data/topt_val.csv
test_table: ../data/topt_test.csv
pdb_dir: ../data/PDB
out_dir: ../runs/topt

epochs: 40
batch_size: 4
loss_alpha_B: 0.15
lr_head: 1.0e-4
lr_backbone: 1.0e-5
```

`sasa_dir` is not required in the training config. EnzyProp calculates
three-class SASA files automatically and stores them under:

```text
out_dir/sasa_three/
```

Training outputs include:

```text
runs/topt/
  topt_best_model.pt
  train_log.csv
  val_predictions.csv
  test_predictions.csv
  metrics.json
  train_config_resolved.json
  sasa_three/
```

The best checkpoint can be copied into `models/` and used directly by
`enzyprop predict`.

## SASA Handling

By default, prediction automatically calculates missing SASA files and deletes
temporary SASA files after prediction.

Save SASA files for reuse:

```bash
enzyprop predict --target topt --input-csv F:\path\to\test\query.csv --pdb-dir F:\path\to\test\PDB --sasa-dir F:\path\to\test\sasa_three --out-dir outputs\topt --device cpu
```

Only generate SASA files:

```bash
enzyprop sasa --input-csv F:\path\to\test\query.csv --pdb-dir F:\path\to\test\PDB --sasa-dir F:\path\to\test\sasa_three
```

Skip SASA calculation and use existing files:

```bash
enzyprop predict --target topt --input-csv F:\path\to\test\query.csv --pdb-dir F:\path\to\test\PDB --sasa-dir F:\path\to\test\sasa_three --skip-sasa --out-dir outputs\topt --device cpu
```

## Output

Single-target files are:

```text
topt_result.csv
phopt_result.csv
pi_result.csv
```

For compatibility with earlier EnzyProp runs, `predict-all` can still read
existing legacy `*_relust.csv` files when merging outputs.

Each single-target CSV contains:

```text
id,pred_mu,pred_var,lower95,upper95,uncertainty
```

`pred_var` is the prediction variance. Internally, EnzyProp uses the predicted
standard deviation to assign `uncertainty`:

- `low`: lower uncertainty, model is more confident
- `medium`: medium uncertainty
- `high`: higher uncertainty, model is less confident

The merged `all_targets_predictions.csv` contains target-specific prediction,
variance, confidence interval, and uncertainty columns, for example:

```text
id,topt_pred,topt_variance,topt_lower95,topt_upper95,topt_uncertainty,...
```

## Defaults

If omitted, EnzyProp uses:

```text
models dir:  ./models
HF cache:    ./hf_cache
PDB dir:     input_csv_folder/PDB
SASA dir:    temporary folder deleted after prediction
```

CUDA is used when available. Use `--device cpu` to force CPU inference.

## Notes

CPU inference is supported, but the trained checkpoints include the GNN channel,
so PyTorch Geometric is required even on CPU servers.
