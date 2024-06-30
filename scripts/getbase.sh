#!/bin/bash -l

#SBATCH --job-name=dgs_base
#SBATCH --time=48:00:00
#SBATCH --partition=ica100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --qos=qos_gpu
#SBATCH --mail-user=vtiyyal1@jh.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH -A mdredze80_gpu
#SBATCH --output="/home/vtiyyal1/askdocs/outputs/dgs_base.out"
#SBATCH --export=ALL

module load anaconda
module load cuda/11.7

conda info --envs

conda activate llmtrain_env


export TOKENIZERS_PARALLELISM=false
echo "Running python script..."

python getbasemodeloutputs.py --model_path "meta-llama/Llama-2-7b-chat-hf" --data_path "/home/vtiyyal1/data-mdredze1/vtiyyal1/askdocschat/high_quality_long_answers_data_apr10.json" --output_path "/home/vtiyyal1/data-mdredze1/vtiyyal1/askdocschat/doctor-guided-system/basellamaoutputs.json"
