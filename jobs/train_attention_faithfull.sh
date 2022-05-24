#!/bin/bash

# Matching DGSR as closelly as possible

python main.py --dataset beauty train --accelerator gpu --devices 1 --val_epochs 1 --epochs 20 --batch_size 50 --batch_accum 1 --num_loader_workers 3 --partial_save CTGR --train_style dgsr_softmax --loss_fn ce --dropout 0 --convolution GAT --conv_layers 3 --activation tanh --concat_previous  neighbour --newsampler --n_max_trans 50 --m_order 1 --sample_all
python main.py --dataset beauty train --accelerator gpu --devices 1 --val_epochs 1 --epochs 20 --batch_size 50 --batch_accum 1 --num_loader_workers 3 --partial_save CTGR --train_style dgsr_softmax --loss_fn ce --dropout 0 --convolution GAT --edge_attr positional --conv_layers 3 --activation tanh --concat_previous neighbour --newsampler --n_max_trans 50 --m_order 1 --sample_all
python main.py --dataset beauty train --accelerator gpu --devices 1 --val_epochs 1 --epochs 20 --batch_size 50 --batch_accum 1 --num_loader_workers 3 --partial_save CTGR --train_style dgsr_softmax --loss_fn ce --dropout 0 --convolution GAT --edge_attr continuous --conv_layers 3 --activation tanh --concat_previous neighbour --newsampler --n_max_trans 50 --m_order 1 --sample_all
python main.py --dataset beauty train --accelerator gpu --devices 1 --val_epochs 1 --epochs 20 --batch_size 50 --batch_accum 1 --num_loader_workers 3 --partial_save CTGR --train_style dgsr_softmax --loss_fn ce --dropout 0 --convolution GATv2 --conv_layers 3 --activation tanh --concat_previous  neighbour --newsampler --n_max_trans 50 --m_order 1 --sample_all
python main.py --dataset beauty train --accelerator gpu --devices 1 --val_epochs 1 --epochs 20 --batch_size 50 --batch_accum 1 --num_loader_workers 3 --partial_save CTGR --train_style dgsr_softmax --loss_fn ce --dropout 0 --convolution GATv2 --edge_attr positional --conv_layers 3 --activation tanh --concat_previous  neighbour --newsampler --n_max_trans 50 --m_order 1 --sample_all
python main.py --dataset beauty train --accelerator gpu --devices 1 --val_epochs 1 --epochs 20 --batch_size 50 --batch_accum 1 --num_loader_workers 3 --partial_save CTGR --train_style dgsr_softmax --loss_fn ce --dropout 0 --convolution GATv2 --edge_attr continuous --conv_layers 3 --activation tanh --concat_previous  neighbour --newsampler --n_max_trans 50 --m_order 1 --sample_all


# Also try: try more heads