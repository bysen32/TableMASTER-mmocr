# CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./tools/dist_train.sh ./configs/textdet/psenet/psenet_r50_fpnf_600e_pubtabnet.py ./work_dir/1210_PseNet_textdet 8
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 ./tools/dist_train.sh ./configs/textdet/psenet/psenet_r50_fpnf_600e_iftable.py ./work_dir/0803_PseNet_textdet 2
