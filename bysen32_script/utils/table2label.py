import Polygon
from scipy.spatial import ConvexHull
import numpy as np
import copy
# import tools.data.utils.polygon_helper as polygon_helper
from utils.utils import get_shared_line, parse_relation_from_table, get_span_cells, \
    get_shared_line_id, sort_shared_line, format_layout, parse_gt_label, extend_text_lines

from utils.format_translate import segmentation_to_bbox



def fuse_gt_info(label, table):
    for idx, cell in enumerate(label['cells']):
        if cell['transcript'] == '':
            bbox = [0, 0, 0, 0]
            segmentation = [[[0, 0], [0, 0], [0, 0], [0, 0]]]
        else:
            ids = [int(i) for i in cell['transcript'].split('-')]
            segmentation = []
            if table['is_wireless']: # 无线表格
                for id in ids:
                    segmentation.append(table['line'][id])
            else: # 有线表格
                for id in ids:
                    segmentation.append(table['cell'][id]) 

            bbox = segmentation_to_bbox(segmentation)

        label['cells'][idx]['bbox'] = bbox
        label['cells'][idx]['segmentation'] = segmentation
        label['cells'][idx]['transcript'] = ''

    # 计算cells与line的关系, update transcript
    label['cells'] = extend_text_lines(label['cells'], table['line'])

    for idx, cell in enumerate(label['cells']):
        if cell['transcript'] == '': # 清空不包含文本的cell
            label['cells'][idx]['bbox'] = [0, 0, 0, 0]
            label['cells'][idx]['segmentation'] = [[[0, 0], [0, 0], [0, 0], [0, 0]]]
        else:
            # ids = [int(i) for i in cell['transcript'].split('-')]
            # segmentation = []
            # for id in ids:
            #     segmentation.append(table['line'][id])
            # bbox = segmentation_to_bbox(segmentation)
            # label['cells'][idx]['bbox'] = bbox
            # label['cells'][idx]['segmentation'] = segmentation
            pass # 包含文本的cell不做处理预测外边界框
    return label


def table2layout(table):
    table = parse_relation_from_table(table)
    span_indice, row_span_indice, col_span_indice = get_span_cells(table['row_adj'], table['col_adj'])
    shared_row_lines = get_shared_line(table['row_adj'], table['cell_adj'], table, row_span_indice)
    shared_col_lines = get_shared_line(table['col_adj'], table['cell_adj'], table, col_span_indice)
    shared_row_line_ids = get_shared_line_id(table['row_adj'], table['cell_adj'], row_span_indice)
    shared_col_line_ids = get_shared_line_id(table['col_adj'], table['cell_adj'], col_span_indice)

    shared_row_line_ids, shared_row_lines, shared_col_line_ids, shared_col_lines = \
            sort_shared_line(shared_row_line_ids, shared_row_lines, shared_col_line_ids, shared_col_lines)
    gt_label = parse_gt_label(table['cell_adj'], table['row_adj'], table['col_adj'], shared_row_line_ids, shared_col_line_ids)
    return gt_label


def table2label(table):
    table = parse_relation_from_table(table)
    span_indice, row_span_indice, col_span_indice = get_span_cells(table['row_adj'], table['col_adj'])
    shared_row_lines = get_shared_line(table['row_adj'], table['cell_adj'], table, row_span_indice)
    shared_col_lines = get_shared_line(table['col_adj'], table['cell_adj'], table, col_span_indice)
    shared_row_line_ids = get_shared_line_id(table['row_adj'], table['cell_adj'], row_span_indice)
    shared_col_line_ids = get_shared_line_id(table['col_adj'], table['cell_adj'], col_span_indice)

    shared_row_line_ids, shared_row_lines, shared_col_line_ids, shared_col_lines = \
            sort_shared_line(shared_row_line_ids, shared_row_lines, shared_col_line_ids, shared_col_lines)
    gt_label = parse_gt_label(table['cell_adj'], table['row_adj'], table['col_adj'], shared_row_line_ids, shared_col_line_ids)

    gt_label = fuse_gt_info(gt_label, table)
    return gt_label


def judge_error(table, gt_label):
    num_row = len(table['row'])
    num_col = len(table['col'])
    num_row_layout = len(gt_label['layout'])
    num_col_layout = len(gt_label['layout'][-1])
    if num_row != num_row_layout:
        return True
    if num_col != num_col_layout:
        return True
    return False
