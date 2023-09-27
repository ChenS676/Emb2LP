python main.py --data_name=ogbl-citation2 --use_node_feat=True --emb_hidden_channels=50 --mlp_hidden_channels=200 --gnn_hidden_channels=200 --grad_clip_norm=1 --eval_steps=1 --num_neg=3 --eval_metric=mrr --epochs=100 --neg_sampler=local --device=5 --init=PPR
python main.py --data_name=ogbl-citation2 --use_node_feat=True --emb_hidden_channels=50 --mlp_hidden_channels=200 --gnn_hidden_channels=200 --grad_clip_norm=1 --eval_steps=1 --num_neg=3 --eval_metric=mrr --epochs=100 --neg_sampler=local --device=5 --alpha=0.2
python main.py --data_name=ogbl-citation2 --use_node_feat=True --emb_hidden_channels=50 --mlp_hidden_channels=200 --gnn_hidden_channels=200 --grad_clip_norm=1 --eval_steps=1 --num_neg=3 --eval_metric=mrr --epochs=100 --neg_sampler=local --device=5 --alpha=0.5
python main.py --data_name=ogbl-citation2 --use_node_feat=True --emb_hidden_channels=50 --mlp_hidden_channels=200 --gnn_hidden_channels=200 --grad_clip_norm=1 --eval_steps=1 --num_neg=3 --eval_metric=mrr --epochs=100 --neg_sampler=local --device=5 --alpha=0.8
python main.py --data_name=ogbl-citation2 --use_node_feat=True --emb_hidden_channels=50 --mlp_hidden_channels=200 --gnn_hidden_channels=200 --grad_clip_norm=1 --eval_steps=1 --num_neg=3 --eval_metric=mrr --epochs=100 --neg_sampler=global --device=5
python main.py --data_name=ogbl-citation2 --use_node_feat=True --emb_hidden_channels=50 --mlp_hidden_channels=200 --gnn_hidden_channels=200 --grad_clip_norm=1 --eval_steps=1 --num_neg=3 --eval_metric=mrr --epochs=100 --neg_sampler=local --device=5 --predictor=MLPCAT
python main.py --data_name=ogbl-citation2 --use_node_feat=True --emb_hidden_channels=50 --mlp_hidden_channels=200 --gnn_hidden_channels=200 --grad_clip_norm=1 --eval_steps=1 --num_neg=3 --eval_metric=mrr --epochs=100 --neg_sampler=local --device=5 --predictor=BIL
python main.py --data_name=ogbl-citation2 --use_node_feat=True --emb_hidden_channels=50 --mlp_hidden_channels=200 --gnn_hidden_channels=200 --grad_clip_norm=1 --eval_steps=1 --num_neg=1 --eval_metric=mrr --epochs=100 --neg_sampler=local --device=5