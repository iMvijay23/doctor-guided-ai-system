#!/bin/bash -l

# Request an interactive session
salloc -p ica100 --job-name=interactive --gres=gpu:1 --time=03:00:00 --nodes=1 --ntasks-per-node=12 --qos=qos_gpu -A mdredze80_gpu <<EOF > interactive_jupyter.log 2>&1

# Load necessary modules
module load anaconda


# Activate conda environment
conda info --envs
conda activate /data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/envs/jupyter

# Start Jupyter Notebook server
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0

EOF
