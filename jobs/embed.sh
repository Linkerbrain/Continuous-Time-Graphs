# example flow

# train a model
python main.py --dataset beauty train --accelerator gpu --devices 1 --partial_save --val_epochs 1 --epochs 5 --batch_size 20 --batch_accum 3 --num_loader_workers 8 CKCONV --train_style dgsr_softmax --loss_fn ce --embedding_size 50 --num_layers 3 --sumconv neighbour --newsampler --sample_all --n_max_trans 50 --m_order 1

# load your best checkpoint and save its embeddings
python main.py --dataset beauty train --notrain --noval --notest --load_checkpoint CTGRLOD-5 --save_embed SumConvFirst --accelerator gpu --devices 1 --partial_save --val_epochs 1 --epochs 5 --batch_size 20 --batch_accum 3 --num_loader_workers 8 CKCONV --train_style dgsr_softmax --loss_fn ce --embedding_size 25 --num_layers 3 --sumconv neighbour --newsampler --sample_all --n_max_trans 50 --m_order 1

# load those embeddings into a fresh model with load_embed and freeze_embed
python main.py --info train_attention_faithfull --dataset beauty train --load_embed SumConvFirst --freeze_embed --accelerator gpu --devices 1 --val_epochs 1 --epochs 20 --batch_size 25 --batch_accum 2 --num_loader_workers 8 --partial_save CTGR --train_style dgsr_softmax --loss_fn ce --dropout 0 --convolution GAT --edge_attr positional --conv_layers 3 --activation tanh --concat_previous neighbour --newsampler --n_max_trans 50 --m_order 1 --sample_all
