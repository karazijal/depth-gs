#!/bin/bash
#SBATCH --job-name=depthgs
#SBATCH --time=2-23:59:00
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --exclude=gnodec2,gnodeg3
#SBATCH --gres=gpu:1
#SBATCH --mail-user=lawrence.karazija@gmail.com
#SBATCH --mail-type=START,END,FAIL,ARRAY_TASKS
date; hostname; pwd
nvidia-smi
echo $TMPDIR CMD: "$@"
date +"%R activating conda env"
source ~/.bashrc
conda activate depthgs || exit
date +"%R changing directory"
cd ~/DepthRegularizedGS || exit
echo Nodelist: $SLURM_NODELIST
echo jid $SLURM_JOB_ID aid $SLURM_ARRAY_JOB_ID tid $SLURM_ARRAY_TASK_ID tc $SLURM_ARRAY_TASK_COUNT ws $SLURM_NTASKS gr $SLURM_PROCID lr $SLURM_LOCALID nr $SLURM_NODEID
date +"%R starting script"
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TRY_DETERMISM_LVL=2
echo $TRY_DETERMISM_LVL $NCCL_DEBUG
echo $SLURM_JOB_NAME
srun "$@"