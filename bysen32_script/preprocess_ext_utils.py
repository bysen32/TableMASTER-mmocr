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
from tqdm import tqdm


class PubtabnetParser(object):
    def __init__(self, is_toy=True, split='valid', foldk="10fold0", chunks_nums=16):
        self.split = split
        # self.data_root = '/userhome/dataset/TSR/Fly_TSR/'
        dataset = "train"
        self.data_root = '/media/ubuntu/Date12/TableStruct/ext_data'
        self.labels_path = os.path.join(self.data_root, f'{dataset}')
        self.split_file  = os.path.join(self.data_root, f'{dataset}_{foldk}.json')
        self.raw_img_root = os.path.join(self.data_root, dataset)
        # self.gt_table_path = os.path.join(self.data_root, 'train_jpg_gt_json')
        # 这个保存的位置有问题 save_root
        self.save_root = os.path.join(self.data_root, 'tablemaster', f'{foldk}', 'cell_box_label')
        self.structure_txt_folder = self.save_root + '/StructureLabelAddEmptyBbox_{}/'.format(split)
        self.is_toy = is_toy
        self.dataset_size = 10 if is_toy else 99999999999
        self.chunks_nums = chunks_nums

        # alphabet path
        self.alphabet_path = self.save_root + '/structure_alphabet.txt'
        # self.alphabet_path_2 = self.save_root + 'textline_recognition_alphabet.txt'

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
        return json.load(open(self.split_file, 'r'))[self.split][:self.dataset_size]


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
                    add_empty_bbox_token_list.append("<eb></eb>")
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
        jsons = json.load(open(self.split_file, 'r'))
        datas = jsons['train'] + jsons['valid']
        with open(self.alphabet_path, 'w') as f:
            for json_name in tqdm(datas):
                # record structure token
                base_name = json_name.split('.')[0]
                html, table = self.base_name2html(base_name)
                cells = table['cells']
                token_list = html['html']['structure']['tokens']
                merged_token = self.merge_token(token_list)
                encoded_token = self.insert_empty_bbox_token(merged_token, cells)
                # encoded_token = merged_token
                for et in encoded_token:
                    if et not in alphabet:
                        alphabet.append(et)
                        f.write(et + '\n')
        print("get structure alphabet cost time {} s.".format(time.time()-start_time))

    def base_name2html(self, base_name):
        table = json.load(open(os.path.join(self.labels_path, base_name + '.json'), 'r'))
        table['layout'] = np.array(table['layout'])
        html = table_to_html(table)

        return html, table

    def gen_gt_cell(self, label, info): # 处理有线表格
        segs = []
        for idx, cell in enumerate(label['cells']):
            str_ids = cell['transcript'].split('-') # 文本 line ids
            # print(str_ids)
            if str_ids[0] == '':
                cell_segs = [[[0, 0], [0, 0], [0, 0], [0, 0]]]
            else:
                cell_ids = [int(float(i)) for i in str_ids]

                cell_segs = []
                # print("1. transcript:", cell['transcript'])
                # print("2. cell len", len(info['cell']))
                for cell_id in cell_ids:
                    # cell_segs.append(info['cell'][cell_id])
                    cell_segs.append(info['line'][cell_id]) # 使用文本包围框

            segs.append(cell_segs)

        for idx in range(len(label['cells'])):
            label['cells'][idx]['segmentation'] = segs[idx]
            label['cells'][idx]['transcript'] = ''

        return label
    
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

        for json_name in tqdm(self.data_generator):

            base_name = json_name.split('.')[0]
            if json_name not in this_chunk:
                continue

            # parse info for Table Structure Master.
            """
            Structure txt include 3 lines:
                1. table image's path
                2. encoded structure token
                3. cell coord from json line file 
            """
            json_filename = base_name + '.json'
            json_filepath = os.path.join(self.structure_txt_folder, json_filename)
            # structure_fid = open(txt_filepath, 'w')

            json_info = {}
            ########## 1. file_path ##########
            # record image path
            image_path = os.path.join(self.raw_img_root, base_name + '.jpg')
            # structure_fid.write(image_path + '\n')
            json_info['file_path'] = image_path


            ########## 2. label ##########
            html, table = self.base_name2html(base_name)
            # record structure token
            cells = table['cells']
            cell_nums = len(cells)
            token_list = html['html']['structure']['tokens']
            merged_token = self.merge_token(token_list)
            encoded_token = self.insert_empty_bbox_token(merged_token, cells)
            # encoded_token = merged_token # 不用上面的函数，因为不需要加空bbox
            encoded_token_str = ','.join(encoded_token)
            # structure_fid.write(encoded_token_str + '\n')
            json_info['label'] = encoded_token_str

            # record bbox coord
            cell_count = self.count_merge_token_nums(encoded_token)
            assert cell_nums == cell_count
            
            ########## 3. bbox ##########
            bboxes = []
            for cell in cells:
                # bbox_line = ','.join([str(b) for b in cell['bbox']]) + '\n'
                # structure_fid.write(bbox_line)
                bboxes.append(cell['bbox'])
            json_info['bbox'] = bboxes

            ########## 4. cell coord ##########
            # info_file = os.path.join(self.raw_img_root, base_name + '.json')
            # info = json.load(open(info_file, 'r'))
            lines = []
            for bbox in bboxes:
                x0, y0, x1, y1 = bbox
                lines.append([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])
            json_info['line'] = lines

            ########## 5. layout ##########
            label_path = os.path.join(self.labels_path, base_name + '.json')
            label = json.load(open(label_path, 'r'))
            json_info['layout'] = label['layout']

            ########## 6. label_relations ##########
            label['layout'] = np.array(label['layout'])
            label_relations = table_to_relations(label)
            json_info['label_relations'] = label_relations

            ########## 7. label_htmls ##########
            # label_htmls = table_to_html(label)
            # label_htmls = format_html(label_htmls)
            # json_info['label_htmls'] = label_htmls

            json.dump(json_info, open(json_filepath, 'w'), indent=4)


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
