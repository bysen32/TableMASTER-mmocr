#!/bin/bash
# NUM_PROC=$1
# shift
# torchrun --nproc_per_node=$NUM_PROC train.py "$@"
# torchrun --nproc_per_node=2 train.py --config configs/convtrans.yaml --epochs=1 --model=convtrans_efficientnet_b0_depth2 --amp
# torchrun --nproc_per_node=2 train.py --config configs/convtrans.yaml --epochs=120 --model=convtrans_efficientnet_b0_depth2 --amp
# torchrun --nproc_per_node=2 train.py --config configs/convtrans.yaml --epochs=120 --model=convtrans_efficientnet_b0_depth4 --amp
# torchrun --nproc_per_node=2 train.py --config configs/convtrans.yaml --epochs=120 --model=convtrans_efficientnet_b0_depth6 --amp
# torchrun --nproc_per_node=2 train.py --config configs/convtrans.yaml --epochs=120 --model=convtrans_efficientnet_b0_depth8 --amp

# Ablation Experiment -- Alpha
# torchrun --nproc_per_node=2 train.py --config configs/efficientnet_b0.yaml --epochs=120 --amp
# torchrun --nproc_per_node=2 train.py --config configs/efficientnet_b0.yaml --epochs=120 --amp
torchrun --nproc_per_node=2 train.py --config configs/efficient_b0.yaml --epochs=20 --amp
