#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --partition=normal
#SBATCH --job-name=gnn_wb

#SBATCH --nodes=1
#SBATCH --mem=501600mb
#SBATCH --output=log/TAG_Benchmark_%j.output
#SBATCH --error=error/TAG_Benchmark_%j.error
#SBATCH --gres=gpu:full:4  # Ensure you are allowed to use these many GPUs, otherwise reduce the number here
#SBATCH --chdir=/hkfs/work/workspace_haic/scratch/cc7738-TAGBench/Emb2LP/Planetoid/res_outputs

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikhelson.g@gmail.com

# Request GPU resources
source /hkfs/home/haicore/aifb/cc7738/anaconda3/etc/profile.d/conda.sh

cd /hkfs/work/workspace_haic/scratch/cc7738-TAGBench/Emb2LP/Planetoid
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12

#python3 planetoid.py --dataset cora --runs 10 --norm_func row_stochastic_matrix --mlp_num_layers 3 --hidden_channels 8192 --dropout 0.5 --epochs 100 --K 20 --alpha 0.2 --init RWR
# python3 planetoid.py --dataset citeseer --runs 10 --norm_func gcn_norm --mlp_num_layers 2 --hidden_channels 8192 --dropout 0.5 --epochs 100 --K 20 --alpha 0.2 --init RWR
python3 planetoid.py --dataset pubmed --runs 10 --norm_func col_stochastic_matrix --mlp_num_layers 3 --hidden_channels 512 --dropout 0.6 --epochs 300 --K 20 --alpha 0.2 --init KI
