#!/bin/bash

python main.py --info train_ckg --dataset beauty train --accelerator gpu --devices 1 --val_epochs 1 --epochs 20 --batch_size 10 --batch_accum 5 --num_loader_workers 3 --partial_save CTGR --train_style dgsr_softmax --loss_fn ce --dropout 0 --add_self_loops --convolution CKGGATv2 --ckg --conv_layers 3  neighbour --newsampler --n_max_trans 50 --m_order 1 --sample_all

python main.py --info train_ckg --dataset beauty train --accelerator gpu --devices 1 --val_epochs 1 --epochs 20 --batch_size 10 --batch_accum 5 --num_loader_workers 3 --partial_save CTGR --train_style dgsr_softmax --loss_fn ce --dropout 0 --convolution CKGSAGE --ckg --conv_layers 3 --add_self_loops neighbour --newsampler --n_max_trans 50 --m_order 1 --sample_all

python main.py --info train_baselines_faithfull --dataset beauty train --accelerator gpu --devices 1 --val_epochs 1 --epochs 20 --batch_size 10 --batch_accum 5 --num_loader_workers 3 --partial_save CTGR --train_style dgsr_softmax --loss_fn ce --dropout 0 --convolution CKGSAGE --ckg --conv_layers 3 --activation tanh --concat_previous neighbour --newsampler --n_max_trans 50 --m_order 1 --sample_all

python main.py --info train_ckg_faithfull --dataset beauty train --accelerator gpu --devices 1 --val_epochs 1 --epochs 20 --batch_size 10 --batch_accum 5 --num_loader_workers 3 --partial_save CTGR --train_style dgsr_softmax --loss_fn ce --dropout 0 --convolution CKGGATv2 --ckg --conv_layers 3 --activation tanh --concat_previous  neighbour --newsampler --n_max_trans 50 --m_order 1 --sample_all
