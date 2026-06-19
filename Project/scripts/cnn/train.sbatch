#!/bin/bash
#SBATCH -J cnn_training
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p shared
#SBATCH --time=8:00:00 
#SBATCH --output=/users/formation/tpirtmntll/out/job_%j.out
#SBATCH --error=/users/formation/tpirtmntll/err/job_%j.err


apptainer exec \
--env PATH=$HOME/.local/bin:$PATH \
--env UV_PROJECT_ENVIRONMENT=/tmpdir/tpirtmntll/envs/aai \
--env UV_NO_SYNC=true \
--env HF_HOME=/work/formation/tpirtmntll/huggingface \
--env HF_HUB_OFFLINE=1 \
--env HF_DATASETS_OFFLINE=1 \
--env TRANSFORMERS_OFFLINE=1 \
--bind /tmpdir,/work \
--nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif \
uv run python /users/formation/tpirtmntll/Modern_LLM/Project/train.py \
    --dataset_type cauldron \
    --max_steps 11000 \
    --checkpoint_dir '/work/formation/tpirtmntll/checkpoints_cnn' \
    --modality_projector_type "cnn" \
    --batch_size 8 \
    --gradient_accumulation_steps 16 \
    --eval_interval 25 \
    --log_interval 5