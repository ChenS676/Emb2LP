import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch
from torch_sparse import coalesce, SparseTensor
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from Planetoid.utils import *
from OGB.utils import *
from torch_geometric.utils import to_undirected
import os.path as osp
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
import time

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def dataloader_Planetoid(args):
    path = osp.join('~/dataset', args.dataset)
    dataset = Planetoid(path, args.dataset)
    
    split_edge = do_edge_split(dataset)
    data = dataset[0]
    data.edge_index = split_edge['train']['edge'].t()
    data = T.ToSparseTensor(remove_edge_index=False)(data)
    
    return data, split_edge

def dataloader_Amazon(args):
    path = osp.join('~/dataset', args.dataset)
    dataset = Amazon(path, args.dataset)
    split_edge = do_edge_split(dataset)
    data = dataset[0]
    data.edge_index = split_edge['train']['edge'].t()
    data = T.ToSparseTensor(remove_edge_index=False)(data)

    return data, split_edge

def dataloader_OGB(args):
    os.makedirs(os.path.expanduser(args.data_path), exist_ok=True)
    
    dataset = PygLinkPropPredDataset(name=args.dataset, root=os.path.expanduser(args.data_path))
    data = dataset[0]

    if hasattr(data, 'edge_weight'):
        if data.edge_weight is not None:
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)

    data = T.ToSparseTensor()(data)
    row, col, _ = data.adj_t.coo() # return coordinates non-zero elements in sparse matrix: rows, columns and values
    data.edge_index = torch.stack([col, row], dim=0)

    if hasattr(data, 'num_features'):
        num_node_feats = data.num_features
    else:
        num_node_feats = 0

    if hasattr(data, 'num_nodes'):
        num_nodes = data.num_nodes
    else:
        num_nodes = data.adj_t.size(0)

    split_edge = dataset.get_edge_split()

    # create log file and save args
    # log_file_name = 'log_' + args.data_name + '_' + str(int(time.time())) + '.txt'
    # log_file = os.path.join(args.res_dir, log_file_name)
    # with open(log_file, 'a') as f:
    #     f.write(str(args) + '\n')

    if hasattr(data, 'x'):
        if data.x is not None:
            data.x = data.x.to(torch.float)

    if args.dataset == 'ogbl-citation2':
        data.adj_t = data.adj_t.to_symmetric()

    if args.dataset == 'ogbl-collab':
        # only train edges after specific year
        if args.year > 0 and hasattr(data, 'edge_year'):
            selected_year_index = torch.reshape((split_edge['train']['year'] >= args.year).nonzero(as_tuple=False), (-1,))
            split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
            split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
            split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
            train_edge_index = split_edge['train']['edge'].t()
            # create adjacency matrix
            new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
            new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
            data.adj_t = SparseTensor(row=new_edge_index[0],
                                      col=new_edge_index[1],
                                      value=new_edge_weight.to(torch.float32))
            data.edge_index = new_edge_index

        # Use training + validation edges
        if args.use_valedges_as_input:
            full_edge_index = torch.cat([split_edge['valid']['edge'].t(), split_edge['train']['edge'].t()], dim=-1)
            full_edge_weight = torch.cat([split_edge['train']['weight'], split_edge['valid']['weight']], dim=-1)
            # create adjacency matrix
            new_edges = to_undirected(full_edge_index, full_edge_weight, reduce='add')
            new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
            data.adj_t = SparseTensor(row=new_edge_index[0],
                                      col=new_edge_index[1],
                                      value=new_edge_weight.to(torch.float32))
            data.edge_index = new_edge_index

            if args.use_coalesce:
                # Объединяет повторяющиеся ребра, суммирая веса и сортирует
                full_edge_index, full_edge_weight = coalesce(full_edge_index, full_edge_weight, num_nodes, num_nodes)

            # edge weight normalization
            split_edge['train']['edge'] = full_edge_index.t()
            deg = data.adj_t.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            # w_ij = w_ij / {sqrt(deg_i) * sqrt(deg_j)}
            split_edge['train']['weight'] = deg_inv_sqrt[full_edge_index[0]] * full_edge_weight * deg_inv_sqrt[full_edge_index[1]]


    if args.encoder.upper() == 'GCN' or args.encoder.upper() == 'SAGE':
        data.adj_t = gcn_normalization(data.adj_t)

    if args.encoder.upper() == 'WSAGE':
        data.adj_t = adj_normalization(data.adj_t)

    if args.encoder.upper() == 'TRANSFORMER':
        row, col, _ = data.adj_t.coo()
        data.adj_t = SparseTensor(row=row, col=col)

    return data, split_edge

group_list = {
    'planetoid': dataloader_Planetoid,
    'amazon'   : dataloader_Amazon,
    'ogb'      : dataloader_OGB
}

# Planetoid: cora, citeseer, pubmed
# Amazon   : photo, computers
# OGB      : ogbl-ppa, ogbl-collab, ogbl-citation2, ogbl-wikikg2, ogbl-ddi, ogbl-biokg, ogbl-vessel
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--group', type=str, default='planetoid')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--use_coalesce', type=str2bool, default=False)
    parser.add_argument('--data_path', type=str, default='~/dataset')
    parser.add_argument('--use_valedges_as_input', type=str2bool, default=False)
    parser.add_argument('--encoder', type=str, default='HLGNN')
    parser.add_argument('--year', type=int, default=2010)

    args = parser.parse_args()
    print(args)
    
    data, split_edge = group_list[args.group](args)
    print(data)
    
