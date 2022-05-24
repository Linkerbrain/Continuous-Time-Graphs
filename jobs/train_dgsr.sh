#!/bin/bash

python main.py --dataset beauty train --accelerator gpu --devices 1 --partial_save --val_epochs 1 --epochs 20 --batch_size 25 --batch_accum 2 --num_loader_workers 3 DGSR --edge_attr none --embedding_size 50 --num_DGRN_layers 3 --train_style dgsr_softmax --loss_fn ce neighbour --newsampler --sample_all --n_max_trans 50 --m_order 1
python main.py --dataset beauty train --accelerator gpu --devices 1 --partial_save --val_epochs 1 --epochs 20 --batch_size 25 --batch_accum 2 --num_loader_workers 3 DGSR --edge_attr positional --embedding_size 50 --num_DGRN_layers 3 --train_style dgsr_softmax --loss_fn ce neighbour --newsampler --sample_all --n_max_trans 50 --m_order 1
python main.py --dataset beauty train --accelerator gpu --devices 1 --partial_save --val_epochs 1 --epochs 20 --batch_size 25 --batch_accum 2 --num_loader_workers 3 DGSR --edge_attr continuous --embedding_size 50 --num_DGRN_layers 3 --train_style dgsr_softmax --loss_fn ce neighbour --newsampler --sample_all --n_max_trans 50 --m_order 1

# Also try: going outside