#!/bin/bash
#SBATCH -J first_training
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p shared
#SBATCH --time=4:00:00 
#SBATCH --output=/users/formation/tpirtmntll/out/job_%j.out
#SBATCH --error=/users/formation/tpirtmntll/err/job_%j.err


apptainer exec \
--env PATH=$HOME/.local/bin:$PATH \
--env UV_PROJECT_ENVIRONMENT=/tmpdir/$USER/envs/aai \
--env UV_NO_SYNC=true \
--env HF_HOME=/work/formation/$USER/huggingface \
--env HF_HUB_OFFLINE=1 \
--env HF_DATASETS_OFFLINE=1 \
--env TRANSFORMERS_OFFLINE=1 \
--bind /tmpdir,/work \
--nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif \
uv run python /users/formation/$USER/Modern_LLM/Project/train.py --dataset_type cauldron \
    --max_steps 90000