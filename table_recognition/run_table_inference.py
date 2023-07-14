import os
import sys
import time
import subprocess

if __name__ == "__main__":
    # detection
    # subprocess.call("CUDA_VISIBLE_DEVICES=0 python -u ./table_recognition/table_inference.py 8 0 0 &"
    #                 "CUDA_VISIBLE_DEVICES=1 python -u ./table_recognition/table_inference.py 8 1 0 &"
    #                 "CUDA_VISIBLE_DEVICES=2 python -u ./table_recognition/table_inference.py 8 2 0 &"
    #                 "CUDA_VISIBLE_DEVICES=3 python -u ./table_recognition/table_inference.py 8 3 0 &"
    #                 "CUDA_VISIBLE_DEVICES=4 python -u ./table_recognition/table_inference.py 8 4 0 &"
    #                 "CUDA_VISIBLE_DEVICES=5 python -u ./table_recognition/table_inference.py 8 5 0 &"
    #                 "CUDA_VISIBLE_DEVICES=6 python -u ./table_recognition/table_inference.py 8 6 0 &"
    #                 "CUDA_VISIBLE_DEVICES=7 python -u ./table_recognition/table_inference.py 8 7 0", shell=True)
    # time.sleep(60)

    # structure
    # subprocess.call("CUDA_VISIBLE_DEVICES=0 python -u ./table_recognition/table_inference.py 8 0 2 &"
    #                 "CUDA_VISIBLE_DEVICES=1 python -u ./table_recognition/table_inference.py 8 1 2 &"
    #                 "CUDA_VISIBLE_DEVICES=2 python -u ./table_recognition/table_inference.py 8 2 2 &"
    #                 "CUDA_VISIBLE_DEVICES=3 python -u ./table_recognition/table_inference.py 8 3 2 &"
    #                 "CUDA_VISIBLE_DEVICES=4 python -u ./table_recognition/table_inference.py 8 4 2 &"
    #                 "CUDA_VISIBLE_DEVICES=5 python -u ./table_recognition/table_inference.py 8 5 2 &"
    #                 "CUDA_VISIBLE_DEVICES=6 python -u ./table_recognition/table_inference.py 8 6 2 &"
    #                 "CUDA_VISIBLE_DEVICES=7 python -u ./table_recognition/table_inference.py 8 7 2", shell=True)

    A = structure_master_ckpt           = './checkpoints/wire_10fold2_best_valid96.30.pth'
    B = structure_master_config         = './configs/textrecog/master/table_master_ResnetExtract_Ranger_0705_wire.py'
    C = test_folder                     = '/media/ubuntu/Date12/TableStruct/new_data/test_A_jpg480max_wire'
    D = structure_master_result_folder  = './output/structure_result/test_A_jpg480max_wire'
        
    subprocess.call(f"CUDA_VISIBLE_DEVICES=0 python -u ./table_recognition/table_inference.py 2 0 {A} {B} {C} {D} &"
                    f"CUDA_VISIBLE_DEVICES=1 python -u ./table_recognition/table_inference.py 2 1 {A} {B} {C} {D}", shell=True)


    A = structure_master_ckpt           = './checkpoints/wireless_10fold2_best.pth'
    B = structure_master_config         = './configs/textrecog/master/table_master_ResnetExtract_Ranger_0705_wireless.py'
    C = test_folder                     = '/media/ubuntu/Date12/TableStruct/new_data/test_A_jpg480max_wireless'
    D = structure_master_result_folder  = './output/structure_result/test_A_jpg480max_wireless'

    # subprocess.call(f"CUDA_VISIBLE_DEVICES=0 python -u ./table_recognition/table_inference.py 2 0 {A} {B} {C} {D} &"
    #                f"CUDA_VISIBLE_DEVICES=1 python -u ./table_recognition/table_inference.py 2 1 {A} {B} {C} {D}", shell=True)

    # recognition
    # subprocess.call("CUDA_VISIBLE_DEVICES=0 python -u ./table_recognition/table_inference.py 8 1 1", shell=True)