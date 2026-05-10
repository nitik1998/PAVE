PY ?= python
DEVICE ?= cpu
N ?= 200
OUT := outputs

.PHONY: help env data split policy oracle demo probe-dinov3 probe-dinov2 probe-siglip2 \
        ground-qwen point-molmoe sam2-weights qual all clean

help:
	@echo "Targets:"
	@echo "  env             Print install commands (does not auto-install)"
	@echo "  data            Download UMD subset to data/umd/"
	@echo "  split           Build train/val/test split lists"
	@echo "  policy          Verify pretrained sb3/tqc-PandaPush-v3 (1 episode)"
	@echo "  oracle          Render oracle affordance overlay for PandaPush-v3"
	@echo "  demo            Record demo MP4 (TQC policy + oracle heatmap panel)"
	@echo "  probe-dinov3    M1 — DINOv3 + linear probe on UMD subset"
	@echo "  probe-dinov2    M1b — DINOv2 baseline"
	@echo "  probe-siglip2   M2 — SigLIP 2 + linear probe"
	@echo "  ground-qwen     M3 — Qwen2.5-VL grounding"
	@echo "  point-molmoe    M4 — MolmoE pointing"
	@echo "  qual            Build cross-method qualitative grid"
	@echo "  all             policy + oracle + demo + probes + qual"
	@echo "  DEVICE=cuda make probe-dinov3   # flip to GPU later"

env:
	@echo "pip install -r requirements.txt"
	@echo "pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121"
	@echo "pip install 'git+https://github.com/facebookresearch/sam2.git'"

data:
	bash scripts/download_umd.sh

split:
	$(PY) scripts/make_split.py --n $(N)

policy:
	$(PY) scripts/verify_pretrained_policy.py

oracle:
	$(PY) scripts/run_oracle_demo.py --frames 8

demo:
	$(PY) scripts/record_demo.py --out $(OUT)/figures/push_demo.mp4

sam2-weights:
	mkdir -p $(OUT)/checkpoints/sam2
	@echo "Download SAM2.1 weights to $(OUT)/checkpoints/sam2/sam2.1_hiera_base_plus.pt"
	@echo "  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"

probe-dinov3:
	$(PY) scripts/run_probes.py --method dinov3 --device $(DEVICE) --n $(N)

probe-dinov2:
	$(PY) scripts/run_probes.py --method dinov2 --device $(DEVICE) --n $(N)

probe-siglip2:
	$(PY) scripts/run_probes.py --method siglip2 --device $(DEVICE) --n $(N)

ground-qwen:
	$(PY) scripts/run_probes.py --method qwen25vl --device $(DEVICE) --n $(N)

point-molmoe:
	$(PY) scripts/run_probes.py --method molmoe --device $(DEVICE) --n $(N)

qual:
	$(PY) scripts/qual_grid.py --n 5

all: policy oracle demo probe-dinov3 probe-dinov2 probe-siglip2 qual

multi-demo:
	$(PY) scripts/multi_episode_demo.py --episodes 3

pickplace:
	$(PY) scripts/wrapper_pickandplace.py --frames 8

cross-domain:
	$(PY) scripts/cross_domain_demo.py --frames 5 --image-size 448 --n-train 130

random-baseline:
	$(PY) scripts/run_probes.py --method random_features --device $(DEVICE) --n 130 --image-size 448

probe-florence:
	$(PY) scripts/run_probes.py --method florence2 --device $(DEVICE) --eval-n 8 --image-size 448

summary:
	$(PY) scripts/summarize_probes.py
	$(PY) scripts/qual_grid.py --methods dinov2 dinov2_448 dinov2_448_full siglip2 openpi_siglip florence2 --n 5 --split-file data/umd/splits/val.json
	$(PY) scripts/hero_panel.py

clean:
	rm -rf $(OUT)/figures/*.png $(OUT)/figures/*.mp4 $(OUT)/tables/*.csv $(OUT)/logs/*
