# Data

Datasets are not committed to this repository. Download them locally as
described below.

## UMD Part Affordance Dataset

The primary dataset used by every probing experiment in this repository.
Approximately 6 GB of RGB-D imagery with per-pixel affordance labels for
17 tool categories, released by Myers *et al.* (ICRA 2015).

### Automated download

```bash
bash ../scripts/download_umd.sh
```

This places the raw archive at `data/umd/raw/` and unpacks the per-tool
images and `_label.mat` files under `data/umd/tools/`. The canonical
mirror is `https://obj.umiacs.umd.edu/part-affordance/part-affordance-dataset-tools.tar.gz`.

### Splits

The 500-sample stratified split used for the headline results lives at
`data/umd/splits_500/{train,val,test}.json` and is committed to the
repository. To regenerate from scratch:

```bash
python ../scripts/make_split.py
```

### Taxonomy

The native UMD taxonomy has seven affordance classes:
`grasp, cut, scoop, contain, pound, support, w-grasp`.
We collapse `grasp + w-grasp` and drop `pound`, yielding the 5-class +
background taxonomy used throughout: `grasp, cut, scoop, contain, support`.
See `configs/affordance_taxonomy.yaml` for the mapping.

### Gotcha

UMD `_label.mat` files store the per-pixel label under the key `gt_label`
(older releases used `gt`). The dataset loader at
[`src/eval/dataset_umd.py`](../src/eval/dataset_umd.py) handles both.

## Other datasets

The repository scaffolds support for additional affordance benchmarks but
does not currently include them:

- **AGD20K** (Luo *et al.*, CVPR 2022) — 20K images, 36 affordance classes.
  Available on Google Drive only.
- **IIT-AFF** — extended grasp affordance benchmark.
- **3D AffordanceNet** — ShapeNet-based 3D annotations.

Cross-dataset validation is listed as future work in
[`../findings.md`](../findings.md).
