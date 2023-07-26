import os
import sys
import json
import glob
import shutil
import multiprocessing
import shutil
from collections import defaultdict
import numpy as np
import time
from utils_bobo.utils import extend_text_lines
from utils_bobo.cal_f1 import table_to_relations
from utils_bobo.format_translate import table_to_html, format_html
from utils_bobo.format_translate import table_to_html as layout_label2html_label
from tqdm import tqdm


class PubtabnetParser(object):
    def __init__(self, is_toy=True, split='valid', foldk="10fold0", chunks_nums=16):
        self.split = split
        # self.data_root = '/userhome/dataset/TSR/Fly_TSR/'
        dataset = "train_jpg480max"
        self.data_root = '/media/ubuntu/Date12/TableStruct/new_data'
        self.layout_label_dir = os.path.join(self.data_root, f'{dataset}_gt_json')
        self.split_file_path  = os.path.join(self.data_root, f'{dataset}_{foldk}.json')
        self.image_dir        = os.path.join(self.data_root, dataset)
        self.save_root        = os.path.join(self.data_root, 'tablemaster_wireless', foldk, 'cell_box_label')
        self.error_file_path  = os.path.join(self.data_root, f"{dataset}_error.json")
        self.structure_txt_folder = os.path.join(self.save_root, f"StructureLabelAddEmptyBbox_{split}")
        self.is_toy = is_toy
        self.dataset_size = 10 if is_toy else 99999999999
        self.chunks_nums = chunks_nums

        # alphabet path
        self.alphabet_path = self.save_root + '/structure_alphabet.txt'

        # make save folder
        self.make_folders()

        # empty box token dict, encoding for the token which is showed in image is blank.
        self.empty_bbox_token_dict = {
            "": '<eb></eb>',
        }
        self.empty_bbox_token_reverse_dict = {v: k for k, v in self.empty_bbox_token_dict.items()}


    @property
    def data_generator(self):
        '''
        return list [00000.json, ...]
        '''
        img_ids = json.load(open(self.split_file_path, 'r'))[self.split][:self.dataset_size]
        error_ids = json.load(open(self.error_file_path, 'r')).keys()
        ids = [id for id in img_ids if id not in error_ids]

        return ids


    def make_folders(self):
        if os.path.exists(self.structure_txt_folder):
            shutil.rmtree(self.structure_txt_folder)
        os.makedirs(self.structure_txt_folder)
        # if not os.path.exists(self.recognition_folder):
        #     os.makedirs(self.recognition_folder)
        # if not os.path.exists(self.detection_txt_folder):
        #     os.makedirs(self.detection_txt_folder)
        # if not os.path.exists(self.recognition_txt_folder):
        #     os.makedirs(self.recognition_txt_folder)
        # for i in range(self.chunks_nums):
        #     recog_img_folder = os.path.join(self.recognition_folder, str(i))
        #     if not os.path.exists(recog_img_folder):
        #         os.makedirs(recog_img_folder)


    def divide_img(self, filenames):
        """
        This function is used to divide all files to nums chunks.
        nums is equal to process nums.
        :param filenames:
        :param nums:
        :return:
        """
        counts = len(filenames)
        nums_per_chunk = counts // self.chunks_nums
        img_chunks = []
        for n in range(self.chunks_nums):
            if n == self.chunks_nums - 1:
                s = n * nums_per_chunk
                img_chunks.append(filenames[s:])
            else:
                s = n * nums_per_chunk
                e = (n + 1) * nums_per_chunk
                img_chunks.append(filenames[s:e])
        return img_chunks


    def get_filenames(self):
        filenames = self.data_generator
        count = len(filenames)

        return filenames, count

    def merge_token(self, token_list):
        """
        This function used to merge the common tokens of raw tokens, and reduce the max length.
        eg. merge '<td>' and '</td>' to '<td></td>' which are always appear together.
        :param token_list: [list]. the raw tokens from the json line file.
        :return: merged tokens.
        """
        pointer = 0
        merge_token_list = []
        # </tbody> is the last token str.
        while token_list[pointer] != '</tbody>':
            if token_list[pointer] == '<td>':
                tmp = token_list[pointer] + token_list[pointer+1]
                merge_token_list.append(tmp)
                pointer += 2
            else:
                merge_token_list.append(token_list[pointer])
                pointer += 1
        merge_token_list.append('</tbody>')
        return merge_token_list

    def insert_empty_bbox_token(self, token_list, cells):
        """
        This function used to insert the empty bbox token(from empty_bbox_token_dict) to token_list.
        check every '<td></td>' and '<td'(table cell token), if 'bbox' not in cell dict, is a empty bbox.
        :param token_list: [list]. merged tokens.
        :param cells: [list]. list of table cell dict, each dict include cell's content and coord.
        :return: tokens add empty bbox str.
        """
        bbox_idx = 0
        add_empty_bbox_token_list = []
        for token in token_list:
            if token == '<td></td>' or token == '<td':
                if cells[bbox_idx]['transcript'] == '':
                    empty_bbox_token = self.empty_bbox_token_dict['']
                    add_empty_bbox_token_list.append(empty_bbox_token)
                else:
                    add_empty_bbox_token_list.append(token)
                bbox_idx += 1
            else:
                add_empty_bbox_token_list.append(token)
        return add_empty_bbox_token_list

    def count_merge_token_nums(self, token_list):
        """
        This function used to get the number of cells by token_list
        :param token_list: token_list after encoded (merged and insert empty bbox token str).
        :return: cells nums.
        """
        count = 0
        for token in token_list:
            if token == '<td':
                count += 1
            elif token == '<td></td>':
                count += 1
            elif token in self.empty_bbox_token_reverse_dict.keys():
                count += 1
            else:
                pass
        return count

    def get_structure_alphabet(self):
        """
        This function will return the alphabet which is used to Table Structure MASTER training.
        :return:
        """
        start_time = time.time()
        print("get structure alphabet ...")
        alphabet = []
        jsons = json.load(open(self.split_file_path, 'r'))
        datas = jsons['train'] + jsons['valid']
        with open(self.alphabet_path, 'w') as f:
            for json_name in tqdm(datas):
                # record structure token
                base_name = json_name.split('.')[0]
                html, table = self.base_name2html(base_name)
                cells = table['cells']
                token_list = html['html']['structure']['tokens']
                merged_token = self.merge_token(token_list)
                # encoded_token = self.insert_empty_bbox_token(merged_token, cells)
                encoded_token = merged_token
                for et in encoded_token:
                    if et not in alphabet:
                        alphabet.append(et)
                        f.write(et + '\n')
        print("get structure alphabet cost time {} s.".format(time.time()-start_time))
    
    def get_layout_label(self, base_name):
        fpath = os.path.join(self.layout_label_dir, f"{base_name}-gt.json")
        layout_label = json.load(open(fpath, 'r'))
        layout_label['layout'] = np.array(layout_label['layout'])

        return layout_label
    
    def get_html_label(self, base_name):
        layout_label = self.get_layout_label(base_name)
        try:
            html_label = layout_label2html_label(layout_label)
        except:
            raise Exception("table_to_html error: ", base_name)
        return html_label
    
    def get_rc_label(self, base_name):
        path = os.path.join(self.image_dir, f'{base_name}.json')
        rc_label = json.load(open(path, 'r'))
        rc_label['is_wireless'] = True # TODO!!!
        return rc_label

    def base_name2html(self, base_name):
        layout_label = json.load(open(os.path.join(self.layout_label_dir, base_name + '-gt.json'), 'r'))
        layout_label['layout'] = np.array(layout_label['layout'])
        try:
            html = table_to_html(layout_label)
        except:
            raise Exception("table_to_html error: ", base_name)

        return html, layout_label 

    def gen_gt_cell(self, layout_label, rc_label): # 处理有线表格
        segs = []
        for idx, cell in enumerate(layout_label['cells']):
            transcript = cell['transcript']
            if transcript:
                str_ids = transcript.split('-') # 文本 line ids
                cell_segs = []
                cell_ids = [int(float(i)) for i in str_ids]
                for cell_id in cell_ids:
                    cell_segs.append(rc_label['line'][cell_id]) # 使用文本包围框 多个分离多边形是否有问题？
            else:
                cell_segs = [[[0, 0], [0, 0], [0, 0], [0, 0]]]
            cell['segmentation'] = cell_segs
            cell['transcript'] = ''

        return layout_label
    
    def parse_single_chunk(self, this_chunk, chunks_idx):
        """
        This function will parse single chunk's image info.
        It will get :
            1. a label file for Table Structure Master training.
            2. some cropped images and text in table image's cell, for Table OCR Master training.
            3. get cell-level coord, for pre-Labeling (Training text line detection model).
        :param this_chunk: a list of image file names, only process the file in it.
        :param chunks_idx: [int]. chunk's id, is used to name folder.
        :return:
        """

        for image_id in tqdm(self.data_generator):

            if image_id not in this_chunk:
                continue

            # parse info for Table Structure Master.
            """
            Structure txt include 3 lines:
                1. table image's path
                2. encoded structure token
                3. cell coord from json line file 
            """
            train_label = {}
            # record image path
            image_path = os.path.join(self.image_dir, image_id + '.jpg')
            train_label['file_path'] = image_path

            # record structure token
            html_label = self.get_html_label(image_id)
            layout_label = self.get_layout_label(image_id)

            token_list = html_label['html']['structure']['tokens']
            merged_token = self.merge_token(token_list)
            cells = layout_label['cells']
            encoded_token = self.insert_empty_bbox_token(merged_token, cells)
            train_label['label'] = ','.join(encoded_token)

            cell_nums = len(cells)
            cell_count = self.count_merge_token_nums(encoded_token)
            assert cell_nums == cell_count
            
            # record bbox coord
            bboxes = []
            for cell in cells:
                bboxes.append(cell['bbox'])
            train_label['bbox'] = bboxes

            rc_label = self.get_rc_label(image_id)
            train_label['rc_label'] = rc_label

            # 0706 不分有线无线
            # if not info['is_wireless']:
            #     # print(pred_json)
            #     label = self.gen_gt_cell(label, info)
            #     label['cells'] = extend_text_lines(label['cells'], info['line'])
            layout_label['layout'] = layout_label['layout'].tolist()
            train_label['layout_label'] = layout_label

            label_htmls = format_html(html_label)
            train_label['label_htmls'] = label_htmls

            # write all label data into json_path
            json_path = os.path.join(self.structure_txt_folder, f"{image_id}.json")
            json.dump(train_label, open(json_path, 'w'), indent=4)


    def parse_images(self, img_chunks):
        """
        single process to parse raw data.
        It will take day to finish 500777 train files parsing.
        :param img_chunks:
        :return:
        """
        for i, img_chunk in enumerate(img_chunks):
            self.parse_single_chunk(img_chunk, i)


    def parse_images_mp(self, img_chunks, nproc):
        """
        multiprocessing to parse raw data.
        It will take about 7 hours to finish 500777 train files parsing.
        One process to do one chunk parsing.
        :param img_chunks:
        :param nproc:
        :return:
        """
        p = multiprocessing.Pool(nproc)
        for i in range(nproc):
            this_chunk = img_chunks[i]
            p.apply_async(self.parse_single_chunk, (this_chunk,i,))
        p.close()
        p.join()
