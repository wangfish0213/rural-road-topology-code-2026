# Urban-to-Rural Road Centerline Extraction — reproducibility package

Code and evaluation artefacts for the manuscript

> **Urban-to-Rural Road Centerline Extraction via Topology-Aware Learning and
> Graph-Node Semantic Link Correction**

The pipeline segments roads from high-resolution imagery, converts the mask into a
graph, and corrects that graph at two levels: an orientation field regressed jointly
with the mask suppresses skeleton spurs, and a learned discriminator links endpoints
that belong to *different* connected components, followed by a deterministic
size-aware pruning rule for isolated components of at most 64 pixels.

This repository is self-contained for **reproducing every table, every number and
every macro in the manuscript**. It does not redistribute the third-party imagery
(download instructions below) or the 1.6 GB of predicted masks (available from the
authors on request).

## Repository layout

```
tools/        data preparation, table and figure generation, auditing, verification
train/        models, training, inference, graph construction, graph correction
eval/         metrics and batch evaluation
manifests/    CSV indexes of the tiles used in each experiment
runs/         evaluation outputs (summary.csv, per_image.csv, config.json) per run
weights/      trained checkpoints for the core models (see below)
docs/         command-line reference for all 40 scripts
```

## Trained checkpoints

| File | Model | Size |
|---|---|---|
| `weights/e1_unet.pt` | U-Net baseline, cross-domain arm (E1) | 7.4 MB |
| `weights/e2_unet.pt` | U-Net baseline, same-domain reference (E2) | 22 MB |
| `weights/m1_ori_e1b.pt` | mask + orientation field (M1) | 7.5 MB |
| `weights/m2_link_net.pt`, `weights/m2_delete_net.pt` | graph correction discriminators (M2) | 93 KB each |

The DeepLabv3-MobileNetV3 checkpoints are 132 MB each and are therefore not
committed; they are available from the authors on request.

## Installation

Python 3.11 or newer.

```
pip install -r requirements.txt
```

A CUDA-capable GPU is optional: all experiments reported in the paper were run on a
single 2 GB laptop GPU (NVIDIA MX450) at the resolutions and subset sizes stated in
the manuscript. CPU execution is possible but slower.

## Data

The datasets are public; download them from their persistent identifiers and place
them as shown. Licences are those of the original providers.

| Directory | Dataset | Source |
|---|---|---|
| `datasets/WHU-RRoad/{train,test}` | WHU-RRoad V1, 27,770 pairs, 0.3 m | doi:10.57760/sciencedb.09181 |
| `datasets/WHU_Road` | WHU Road (1 m / 2 m), urban source domain | github.com/fightingMinty/WHU-road-dataset |
| `datasets/WHU_Building` | WHU Building, aerial 0.3 m | doi:10.21227/g0hv-0521 |
| `datasets/IRSAMap/IRSAMap_V2` | IRSAMap V2, 0.5 m, vector layers | doi:10.57760/sciencedb.j00240.00047 |

IRSAMap V2 distributes road centerlines as GeoJSON only. Rasterise them first:

```
python tools/irsamap_prep.py
```

Note that IRSAMap class rasters store class *ids* (building = 31, road area = 32);
they must be binarised with `> 0`, not with a 127 threshold.

## Reproducing the paper

### 1. Tables, numbers and macros (no GPU needed)

```
python tools/build_manuscript_tables.py     # writes manuscript/tables/*.tex, numbers.json
python tools/audit_manuscript.py            # provenance audit; must print RESULT: PASS
python tools/wordcount.py                   # word count against the journal limit
```

The `runs/` directory shipped here contains the evaluation outputs of every
experiment reported in the paper. Running the first command on a clean checkout
regenerates all six tables, `key_numbers.tex` and `numbers.json` **byte for byte
identical** to the versions used in the submitted manuscript.

### 2. Training, inference and evaluation

Every script documents its options in `docs/USAGE.md` or via `--help`. The typical
sequence for one experiment is:

```
# orientation field + mask (M1)
python train/train_ori.py --manifest manifests/whu_road_1m_train.csv \
    --val-manifest manifests/whu_rroad_test.csv --out runs/m1_ori

# sliding-window inference on 1024 x 1024 tiles
# (use runs/m1_ori/best.pt instead if you retrained rather than using the release)
python train/predict.py --ckpt weights/m1_ori_e1b.pt \
    --manifest manifests/whu_rroad_test.csv --out runs/m1_ori/pred

# per-tile metrics and summary.csv
python eval/eval_dir.py --gt datasets/WHU-RRoad/test \
    --pred runs/m1_ori/pred --out runs/m1_ori/eval --gsd 0.3
```

### 3. Graph correction (M2)

```
python train/m2_correction.py build  --pred runs/m1_ori/pred \
    --gt datasets/WHU-RRoad/test --out runs/m2_data
python train/m2_correction.py train  --data runs/m2_data --out runs/m2_model
python train/m2_correction.py apply  --pred runs/m1_ori/pred --model runs/m2_model \
    --out runs/m2_corrected/pred --out-dir-stats runs/m2_corrected/apply_stats.json
```

### 4. Line–area consistency experiment (E4)

```
python tools/irsamap_prep.py
python train/e4_consistency.py --anchor gt --lam-excl 0.1 --lam-dist 0.1 --limit-train 300
python train/e4_postprocess.py
python tools/e4_gain_criteria.py
python tools/summarize_e4.py
```

The gain criterion was fixed in code (`tools/e4_gain_criteria.py`) *before* the
experiments were run: a minimum improvement of 0.005 in APLS or building IoU, a
paired *t*-test with Bonferroni correction, and a coverage-matched control. None of
the 29 candidate working points passed, which is why the consistency prior is
reported as a preliminary experiment and not as a contribution.

## Verification

Outputs are never trusted on the strength of a success message. `tools/verify_outputs.py`
re-reads every generated artefact in a separate process and checks existence, size,
image decodability (non-uniform pixels) and non-empty CSV/JSON content:

```
python tools/verify_outputs.py --spec manuscript/verify_spec.json
```

## Compute and runtime

Reported experiments: 176x176 to 256x256 crops, 300-600 training tiles, 4-14 epochs.
Training a small U-Net on 300 tiles for 14 epochs takes roughly 47 minutes on the
2 GB GPU. Increasing the batch size did not reduce wall-clock time, and CUDA
automatic mixed precision produced NaN losses on this device, so it is disabled by
default.

## Licence

MIT — see `LICENSE`. Third-party datasets remain under their original licences
(IRSAMap V2 is CC BY-NC-SA 4.0); nothing in this repository relicenses them.

## Citation

If you use this code, please cite the manuscript above. A `CITATION.cff` is provided
for reference managers.
