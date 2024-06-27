#!/bin/bash -l

#SBATCH --job-name=analyze_results
#SBATCH --time=48:00:00
#SBATCH --partition=ica100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --qos=qos_gpu
#SBATCH --mail-user=vtiyyal1@jh.edu
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END
#SBATCH --mail-type FAIL
#SBATCH -A mdredze80_gpu
#SBATCH --output="/home/vtiyyal1/askdocs/outputs/analyze_results.out"
#SBATCH --export=ALL

module load anaconda
module load cuda/11.7

conda info --envs
conda activate llmtrain_env
pip install sentence_transformers
echo "Running analysis script..."

python analyze_results.py --results_file "/home/vtiyyal1/data-mdredze1/vtiyyal1/askdocschat/doctor-guided-system/dgs_trial7_results_ica.json" --empathy_model_name "vtiyyal1/empathy_model" --quality_model_name "vtiyyal1/quality_model"