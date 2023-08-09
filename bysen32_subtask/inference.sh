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
python inference.py /media/ubuntu/Date12/TableStruct/new_data/test_A_jpg480max --model efficientnet_b0 \
--checkpoint ./output/train/model_best.pth.tar \
--num-classes 2 \
--results-dir /media/ubuntu/Date12/TableStruct/new_data \
--results-file test_A_jpg480max_wireornot.json \
--results-format json
