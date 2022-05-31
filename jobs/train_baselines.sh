#!/bin/bash

python main.py --info train_baselines --dataset beauty train --accelerator gpu --devices 1 --val_epochs 1 --epochs 20 --batch_size 50 --batch_accum 1 --num_loader_workers 3 --partial_save CTGR --train_style dgsr_softmax --loss_fn ce --dropout 0 --convolution SAGE --conv_layers 3 --add_self_loops neighbour --newsampler --n_max_trans 50 --m_order 1 --sample_all


python main.py --info train_baselines --dataset beauty train --accelerator gpu --devices 1 --val_epochs 1 --epochs 20 --batch_size 50 --batch_accum 1 --num_loader_workers 3 --partial_save CTGR --train_style dgsr_softmax --loss_fn ce --homogenous --dropout 0 --convolution LG --conv_layers 3 --add_self_loops neighbour --newsampler --n_max_trans 50 --m_order 1 --sample_all
#python main.py --info train_baselines --dataset beauty train --accelerator gpu --devices 1 --val_epochs 1 --epochs 20 --batch_size 50 --batch_accum 1 --num_loader_workers 3 --partial_save CTGR --train_style dgsr_softmax --loss_fn ce --homogenous --dropout 0 --convolution LG --layered_embedding mean --conv_layers 3 --add_self_loops neighbour --newsampler --n_max_trans 50 --m_order 1 --sample_all


python main.py --info train_baselines --dataset beauty train --accelerator gpu --devices 1 --val_epochs 1 --epochs 20 --batch_size 50 --batch_accum 1 --num_loader_workers 3 --partial_save CTGR --train_style dgsr_softmax --loss_fn ce --homogenous --dropout 0 --convolution SG --conv_layers 3 --add_self_loops neighbour --newsampler --n_max_trans 50 --m_order 1 --sample_all


python main.py --info train_baselines --dataset beauty train --accelerator gpu --devices 1 --val_epochs 1 --epochs 20 --batch_size 50 --batch_accum 1 --num_loader_workers 3 --partial_save CTGR --train_style dgsr_softmax --loss_fn ce --split_conv --dropout 0 --convolution GCN --conv_layers 3 --add_self_loops neighbour --newsampler --n_max_trans 50 --m_order 1 --sample_all

# Also try: layered_embedding mean,