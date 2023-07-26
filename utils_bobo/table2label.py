import Polygon
from scipy.spatial import ConvexHull
import numpy as np
import copy
# import tools.data.utils.polygon_helper as polygon_helper
from utils_bobo.utils import get_shared_line, parse_relation_from_table, get_span_cells, \
    get_shared_line_id, sort_shared_line, format_layout, parse_gt_label, extend_text_lines

from utils_bobo.format_translate import segmentation_to_bbox



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

    # 计算逻辑单元cells与文本line的关系, 更新字段transcript, ::主要是更新有线表格的transcript
    label['cells'] = extend_text_lines(label['cells'], table['line'], table['line_valid'])
    for idx, cell in enumerate(label['cells']):
        if cell['transcript'] == '': # 清空不包含文本的有线表格单元
            label['cells'][idx]['bbox'] = [0, 0, 0, 0]
            label['cells'][idx]['segmentation'] = [[[0, 0], [0, 0], [0, 0], [0, 0]]]
        else:
            ids = [int(i) for i in cell['transcript'].split('-')]
            segmentation = []
            for id in ids:
                segmentation.append(table['line'][id])
            bbox = segmentation_to_bbox(segmentation)
            label['cells'][idx]['bbox'] = bbox
            label['cells'][idx]['segmentation'] = segmentation
            pass # 包含文本的cell不做处理预测外边界框
    return label


def table2layout(table):
    table = parse_relation_from_table(table)
    span_indice, row_span_indice, col_span_indice = get_span_cells(table['row_adj'], table['col_adj']) # 计算跨行/列的line idx
    shared_row_lines = get_shared_line(table['row_adj'], table['cell_adj'], table, row_span_indice, table['line_valid']) #非跨行的line的坐标
    shared_col_lines = get_shared_line(table['col_adj'], table['cell_adj'], table, col_span_indice, table['line_valid']) #非跨列的line的坐标
    shared_row_line_ids = get_shared_line_id(table['row_adj'], table['cell_adj'], row_span_indice, table['line_valid']) #非跨行的line的idx
    shared_col_line_ids = get_shared_line_id(table['col_adj'], table['cell_adj'], col_span_indice, table['line_valid']) #非跨列的line的idx

    shared_row_line_ids, shared_row_lines, shared_col_line_ids, shared_col_lines = \
            sort_shared_line(shared_row_line_ids, shared_row_lines, shared_col_line_ids, shared_col_lines)
    gt_label = parse_gt_label(table['cell_adj'], table['row_adj'], table['col_adj'], shared_row_line_ids, shared_col_line_ids, table['line_valid'])
    return gt_label


def table2label(table):
    gt_label = table2layout(table)
    gt_label = fuse_gt_info(gt_label, table)
    return gt_label


def judge_error(table, label):
    count = len(table['line']) # 注意，这里不区分有线无线了，fuse_gt_info已处理
    flag = np.zeros(count)
    
    # 1. 检查 transcript 全包含
    for idx, cell in enumerate(label['cells']):
        if cell['transcript']:
            ids = [int(i) for i in cell['transcript'].split('-')]
            for id in ids:
                flag[id] = 1
    if flag.sum() != count:
        return False, f"line idx:{np.argwhere(flag == 0).tolist()} not find"
    
    layout = np.array(label['layout'])

    # 2. 检查 layout 分布
    for cell_idx in range(len(label['cells'])):
        cell_positions = np.argwhere(layout == cell_idx)
        row_span = [np.min(cell_positions[:, 0]), np.max(cell_positions[:, 0]) + 1]
        col_span = [np.min(cell_positions[:, 1]), np.max(cell_positions[:, 1]) + 1]
        
        try:
            assert np.all(layout[row_span[0]:row_span[1], col_span[0]:col_span[1]] == cell_idx)
        except:
            return False, f"layout error: cell_idx: {cell_idx}, row_span: {row_span}, col_span: {col_span}"
    
    # 3. 检测 layout 与 cells 的个数
    try:
        assert layout.max() + 1 == len(label['cells'])
    except:
        return False, f"layout.max()+1 != len(label['cells']): {layout.max()+1} != {len(label['cells'])}"
    
    return True, ''

    # 这里不再使用
    # num_row = len(table['row'])
    # num_col = len(table['col'])
    # num_row_layout = len(label['layout'])
    # num_col_layout = len(label['layout'][-1])
    # if num_row != num_row_layout:
    #     print(f"num_row: {num_row} != num_row_layout: {num_row_layout}")
    #     return True
    # if num_col != num_col_layout:
    #     print(f"num_col: {num_col} != num_col_layout: {num_col_layout}")
    #     return True
    # return False
