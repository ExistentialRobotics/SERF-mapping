# Dataset Generation

This document describes how to generate SERF mapping datasets manually from the
repository root.

## Installation

Install the following extras only when generating datasets manually. They are
not required when using the pre-generated datasets linked from
[README.md](README.md).

```bash
git submodule update --init --recursive external/sam2 external/BEHAVIOR-1K

conda activate serf-mapping
python -m pip install -e external/sam2

cd external/BEHAVIOR-1K
./setup.sh --omnigibson --bddl --joylo
cd ../..

python -m pip install numpy==1.26.4
python -m omnigibson.utils.asset_utils --download_behavior_1k_assets
python -m omnigibson.utils.asset_utils --download_2025_challenge_task_instances
```

For BEHAVIOR system requirements, see the official
installation guide: https://behavior.stanford.edu/getting_started/installation.html

## Mapping Dataset Generation

### Environment Dataset Generation

The commands below use `task-0021` as an example and download the required
BEHAVIOR-1K inputs and SERF metadata:

```bash
hf download behavior-1k/2025-challenge-demos \
  --repo-type dataset \
  --include "data/task-0021/**" \
  --local-dir data/behavior-1k/2025-challenge-demos

hf download behavior-1k/2025-challenge-rawdata \
  --repo-type dataset \
  --include "task-0021/**" \
  --local-dir data/behavior-1k/2025-challenge-rawdata

hf download suk063/SERF \
  --repo-type dataset \
  --include "sampled_pose/task-0021/**" \
  --local-dir data

hf download suk063/SERF \
  --repo-type dataset \
  --include "instance_id_map/task-0021/**" \
  --local-dir data
```

Render the provided pre-sampled camera poses into the mapping dataset:

```bash
python dataset/render_from_sampled_pose.py \
  --data_folder data/behavior-1k/2025-challenge-rawdata \
  --task_id 21 \
  --demo_id 212800 \
  --output_folder data/mapping_dataset \
  --instance_id_map data/instance_id_map/task-0021/instance_id_map.json
```

Precompute SAM masks:

```bash
python dataset/precompute_sam.py \
  --input_path data/mapping_dataset/task-0021/train/episode_00212800.hdf5
```

Instance ID remapping:
OmniGibson may assign different instance IDs across runs, so segmentation IDs
should be remapped with a fixed map. We provide `dataset/remap_instance_ids.py`,
and the fixed ID maps used in SERF are available at
https://huggingface.co/datasets/suk063/SERF/tree/main/instance_id_map.

For already-generated HDF5 files, apply the fixed map in-place:

```bash
python dataset/remap_instance_ids.py \
  --input_dir data/mapping_dataset/task-0021/train \
  --instance_id_map data/instance_id_map/task-0021/instance_id_map.json
```

### Robot Dataset Generation

Generate the robot mapping dataset by capturing hemispherical camera views
around the robot:

```bash
python dataset/capture_robot_hemisphere.py
```

## Expert Demonstration Generation

Replay a BEHAVIOR-1K raw demonstration into SERF's replayed demonstration
format:

```bash
python dataset/replay_dataset.py \
  --data_folder data/behavior-1k/2025-challenge-rawdata \
  --task_id 21 \
  --demo_id 212800 \
  --output_folder data/demonstration_replay \
  --instance_id_map data/instance_id_map/task-0021/instance_id_map.json
```
