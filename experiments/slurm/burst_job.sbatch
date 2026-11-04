#!/usr/bin/env bash
# Runs one configuration on one burst GPU. Choose the partition on the command line:
#   sbatch --account=YOUR_ACCOUNT --partition=g2-standard-12 --export=ALL,REPRO_WORK=$SCRATCH/repro,TAG=cuda124 experiments/slurm/burst_job.sbatch
#   sbatch --account=YOUR_ACCOUNT --partition=c12m85-a100-1 --export=ALL,REPRO_WORK=$SCRATCH/repro,TAG=cuda124 experiments/slurm/burst_job.sbatch
# Set STEPS="correctness" or "accuracy_speed" to run only one. If the cluster needs a container,
# set CONTAINER to its image; everything then runs inside it with GPU access.
#SBATCH --job-name=repro
#SBATCH --dependency=singleton
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=repro_%x_%j.log
# singleton: one job at a time, because the kernel tests share each folder's build/ directory.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export GPUS=0
STEPS=${STEPS:-correctness accuracy_speed}
for s in $STEPS; do
  case $s in
    correctness) script=experiments/02_correctness.sh ;;
    accuracy_speed) script=experiments/03_accuracy_speed.sh ;;
    *) echo "unknown step $s"; exit 1 ;;
  esac
  if [ -n "${CONTAINER:-}" ]; then
    "$(command -v apptainer || command -v singularity)" exec --nv --bind "$REPRO_WORK" "$CONTAINER" bash "$script" "$TAG"
  else
    bash "$script" "$TAG"
  fi
done
