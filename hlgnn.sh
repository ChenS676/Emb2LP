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

cd /hkfs/work/workspace_haic/scratch/cc7738-TAGBench/Emb2LP
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12

# running command: bash hlgnn.sh planetoid (or ogb)
if [ "$1" == "planetoid" ]; then
    cd Planetoid
    python3 planetoid.py --dataset cora --runs 1 --norm_func row_stochastic_matrix --mlp_num_layers 3 --hidden_channels 8192 --dropout 0.5 --epochs 100 --K 20 --alpha 0.2 --init RWR
    # python3 planetoid.py --dataset citeseer --runs 10 --norm_func gcn_norm --mlp_num_layers 2 --hidden_channels 8192 --dropout 0.5 --epochs 100 --K 20 --alpha 0.2 --init RWR
    # python3 planetoid.py --dataset pubmed --runs 10 --norm_func col_stochastic_matrix --mlp_num_layers 3 --hidden_channels 512 --dropout 0.6 --epochs 300 --K 20 --alpha 0.2 --init KI
    # python3 amazon.py --dataset photo --runs 1 --norm_func col_stochastic_matrix --mlp_num_layers 3 --hidden_channels 512 --dropout 0.6 --epochs 200 --K 20 --alpha 0.2 --init RWR
    # python3 amazon.py --dataset computers --runs 10 --norm_func row_stochastic_matrix --mlp_num_layers 3 --hidden_channels 512 --dropout 0.6 --epochs 200 --K 20 --alpha 0.2 --init RWR
else

    cd OGB
    # python3 main.py --dataset ogbl-collab --runs 10 --norm_func col_stochastic_matrix --predictor DOT --use_valedges_as_input True --year 2010 --epochs 800 --eval_last_best True --dropout 0.3 --use_node_feat True
    # python3 main.py --dataset ogbl-ddi --runs 10 --norm_func row_stochastic_matrix --emb_hidden_channels 512 --gnn_hidden_channels 512 --mlp_hidden_channels 512 --num_neg 3 --dropout 0.3 --loss_func WeightedHingeAUC
    # python3 main.py --dataset ogbl-ppa --runs 10 --norm_func col_stochastic_matrix --emb_hidden_channels 256 --mlp_hidden_channels 512 --gnn_hidden_channels 512 --grad_clip_norm 2.0 --epochs 500 --eval_steps 1 --num_neg 3 --dropout 0.5 --use_node_feat True --alpha 0.5 --loss_func WeightedHingeAUC
    python3 main.py --dataset ogbl-citation2 --runs 10 --norm_func row_stochastic_matrix --emb_hidden_channels 64 --mlp_hidden_channels 256 --gnn_hidden_channels 256 --grad_clip_norm 1.0 --epochs 100 --eval_steps 1 --num_neg 3 --dropout 0.3 --eval_metric mrr --neg_sampler local --use_node_feat True --alpha 0.6
fi