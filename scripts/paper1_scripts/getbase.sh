#!/bin/bash -l

#SBATCH --job-name=das_base
#SBATCH --time=48:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --qos=qos_gpu
#SBATCH --mail-user=vtiyyal1@jh.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH -A mdredze80_gpu
#SBATCH --output="/home/vtiyyal1/askdocs/outputs/das_base.out"
#SBATCH --export=ALL

module load anaconda
module load cuda/11.7

conda info --envs

conda activate llmtrain_env

export TOKENIZERS_PARALLELISM=false
echo "Running python script for base model responses..."

python generate_base_response.py --model_path "meta-llama/Meta-Llama-3-8B-Instruct" --data_path "/data/mdredze1/vtiyyal1/askdocschat/length_filtered_high_quality_data_jul11_300.json" --output_path "/data/mdredze1/vtiyyal1/askdocschat/doctor_augmented/basellama3_8b_inst_outputs.json"
