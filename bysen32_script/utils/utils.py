import numpy as np
import Polygon
import copy


# parse relation adjacency matrix
def parse_relation_from_table(table, iou_threshold=0.8): # 使用高IOU阈值
    if table['is_wireless']:
        line_polys = table['line']
    else:
        line_polys = table['cell']

    # line_polys = table['line'] # row计算对应的是line

    # parse row relation adjacency matrix
    row_adj = np.identity(len(line_polys), dtype=np.int64)
    row_polys = table['row']
    line_has_row = np.zeros(len(line_polys), dtype=np.int64)
    for row_poly in row_polys: # 行多边形
        same_row_idxs = []
        row_polygon = Polygon.Polygon(row_poly)
        for idx, line_poly in enumerate(line_polys): # 行多边形与cell/line多边形的iou
            line_polygon = Polygon.Polygon(line_poly)
            iou = (row_polygon & line_polygon).area() / min(row_polygon.area(), line_polygon.area())
            if iou >= iou_threshold:
                same_row_idxs.append(idx)
                line_has_row[idx] = 1
        # map to row relation adjacency matrix
        for i in same_row_idxs: # 同行的cell或line
            for j in same_row_idxs:
                row_adj[i,j] = 1
    table['row_adj'] = row_adj # 计算各个cell/line的同行关系

    # parse col relation adjacency matrix
    col_adj = np.identity(len(line_polys), dtype=np.int64)
    col_polys = table['col']
    line_has_col = np.zeros(len(line_polys), dtype=np.int64)
    for col_poly in col_polys:
        same_col_idxs = []
        col_polygon = Polygon.Polygon(col_poly)
        for idx, line_poly in enumerate(line_polys):
            line_polygon = Polygon.Polygon(line_poly)
            iou = (col_polygon & line_polygon).area() / min(col_polygon.area(), line_polygon.area())
            if iou >= iou_threshold:
                same_col_idxs.append(idx)
                line_has_col[idx] = 1
        # map to col relation adjacency matrix
        for i in same_col_idxs:
            for j in same_col_idxs:
                col_adj[i,j] = 1
    table['col_adj'] = col_adj

    # parse cell relation adjacency matrix
    cell_adj = np.array((row_adj + col_adj)==2, dtype=np.int64)
    line_valid = np.array((line_has_row + line_has_col)==2, dtype=np.int64) # 行列都被分配的 line 才合法
    table['line_valid'] = line_valid
    table['cell_adj'] = cell_adj

    # 使用最大团算法 防止cell计算错误
    #cell_adj_new = np.zeros_like(cell_adj)
    #import networkx as nx
    #G = nx.Graph()
    #G.add_edges_from(np.argwhere(cell_adj==1))
    #cliques = list(nx.find_cliques(G))
    #cliques = sorted(cliques, key=lambda x: len(x), reverse=True)
    #idx_flag = np.zeros(len(cell_adj), dtype=np.int32)
    #for idxs in cliques:
    #    for idxi in idxs:
    #        for idxj in idxs:
    #            if idx_flag[idxi] == 0 and idx_flag[idxj] == 0:
    #                cell_adj_new[idxi, idxj] = 1
    #    for idx in idxs: # 统一标记
    #        idx_flag[idx] = 1
    #table['cell_adj'] = cell_adj_new

    return table


# get spanning cells idx
def get_span_cells(row_adj, col_adj):
    row_span_indice = []
    for row_idx, row in enumerate(row_adj):
        idx_r = list(np.where(row == 1)[0])
        if len(idx_r) > 2:
            idx_r.remove(row_idx) # 移除自身
            for idx1 in idx_r: # justify this cell is spanning cell or not
                for idx2 in idx_r:
                    if row_adj[idx1, idx2] != 1:
                        row_span_indice.append(row_idx)

    col_span_indice = []
    for col_idx, col in enumerate(col_adj):
        idx_c = list(np.where(col == 1)[0])
        if len(idx_c) > 2:
            idx_c.remove(col_idx)
            for idx1 in idx_c: # justify this cell is spanning cell or not
                for idx2 in idx_c: # 与自身同列的cell/line 分属于不同的列 说明是自身跨列
                    if col_adj[idx1, idx2] != 1:
                        col_span_indice.append(col_idx)
    span_text_indice = list(set(row_span_indice + col_span_indice))
    row_span_text_indice = list(set(row_span_indice))
    col_span_text_indice = list(set(col_span_indice))
    return span_text_indice, row_span_text_indice, col_span_text_indice


def get_shared_line(adj_mat, adj_cell, table, span_index):
    if table['is_wireless']:
        text_box = table['line']
    else:
        text_box = table['cell']

    # text_box = table['line']

    all_index = list(range(len(text_box))) # 枚举所有的cell/line
    for sidx in span_index: # 剔除span的cell/line
        all_index.remove(sidx)
    
    adj_mat_wo_span = adj_mat[all_index][:,all_index] # 提取非span的cell/line 子矩阵
    adj_cell_wo_span = adj_cell[all_index][:,all_index] # 提取非span的cell/line

    text_box_wo_span = [text_box[idx_] for idx_ in range(len(text_box)) if idx_ not in span_index] # 提取非span的cell/line的多边形
    
    neglect = []
    text_share_all = []
    for ridx, adj in enumerate(adj_mat_wo_span): # 枚举非span的cell/line
        if ridx not in neglect:
            text_idx = adj.nonzero()[0] # 将同行的cell/line idx提取出来
            text_share = []
            neglect.extend(text_idx)
            neglect_share_cell = []
            for tidx in text_idx:
                if tidx not in neglect_share_cell:
                    text_idx_c = adj_cell_wo_span[tidx].nonzero()[0]
                    neglect_share_cell.extend(text_idx_c)
                    text_share.append([text_box_wo_span[idx_] for idx_ in text_idx_c])
            text_share_all.append(text_share)
    
    return text_share_all


def get_shared_line_id(adj_mat, adj_cell, span_index):   
    neglect = []
    text_share_all = []
    for ridx, adj in enumerate(adj_mat): # 列邻居矩阵，逐line枚举
        if ridx in span_index: # 排除span列的 line
            continue
        if ridx not in neglect:
            text_idx = adj.nonzero()[0] # 获得所有同列的line
            # remove span index
            text_idx = [idx for idx in text_idx if idx not in span_index] # 从中剔除span的line
            text_share = []
            neglect.extend(text_idx)
            neglect_share_cell = []
            for tidx in text_idx: # 枚举同列的单列line
                if tidx not in neglect_share_cell:
                    text_idx_c = adj_cell[tidx].nonzero()[0] # 它们各自同单元格的lines
                    neglect_share_cell.extend(text_idx_c)
                    for idx_ in text_idx_c:
                        if idx_ not in span_index:
                            text_share.append(idx_)
            text_share_all.append(text_share)

    return text_share_all


def sort_shared_line(share_text_id_row, shared_text_row, share_text_id_col, shared_text_col):
    # sort rows from top to down
    row_locs = []
    for row_text in shared_text_row:
        points_ = np.vstack([np.vstack(itm) for itm in row_text])
        row_loc_ = np.mean(points_, axis=0)[1]
        row_locs.append(row_loc_)
    row_index = np.argsort(row_locs)
    share_text_id_row = [share_text_id_row[idx_] for idx_ in row_index]
    shared_text_row = [shared_text_row[idx_] for idx_ in row_index]

    # sort cols from left to right
    col_locs = []
    for col_text in shared_text_col:
        points_ = np.vstack([np.vstack(itm) for itm in col_text])
        col_loc_ = np.mean(points_, axis=0)[0]
        col_locs.append(col_loc_)
    col_index = np.argsort(col_locs)
    share_text_id_col = [share_text_id_col[idx_] for idx_ in col_index]
    shared_text_col = [shared_text_col[idx_] for idx_ in col_index]

    return share_text_id_row, shared_text_row, share_text_id_col, shared_text_col


def format_layout(layout):
    new_layout = np.full_like(layout, -1)
    row_nums, col_nums = layout.shape
    cell_id = 0
    for row_id in range(row_nums):
        for col_id in range(col_nums):
            if new_layout[row_id, col_id] == -1:
                y, x = np.where(layout==layout[row_id, col_id])
                new_layout[y, x] = cell_id
                cell_id += 1
    assert new_layout.min() >= 0
    return new_layout


def parse_gt_label(cell_adj, row_adj, col_adj, shared_row_line_ids, shared_col_line_ids, line_valid):
    num_row = len(shared_row_line_ids)
    num_col = len(shared_col_line_ids)
    if num_row == 0 or num_col == 0:
        table = dict(
            layout=np.zeros((1,1), dtype=np.int64).tolist(),
            cells=[dict(
                col_start_idx=0,
                row_start_idx=0,
                col_end_idx=0,
                row_end_idx=0,
                transcript='0'
            )]
        )
        return table

    layout = np.arange(int(num_row*num_col)).reshape(num_row, num_col)
    start_id = int(num_row*num_col)

    neglect = [] # passed assigned cell ids 已分配的cell/line
    assign_text_id = dict() # save assigned cell ids
    for index, adj in enumerate(cell_adj):
        if index in neglect: # justify assign or not
            continue
        if line_valid[index] == 0:
            continue
        cell_ids = adj.nonzero()[0] # 同cell的line id
        neglect.extend(cell_ids) # 同单元的所有line idx标记

        span_row_ids = []
        span_col_ids = []

        # find all span row line ids
        span_row_line_ids = []
        for ids in cell_ids: # 通过同cell的line ids找到与它们同row的line ids
            span_row_line_ids.extend(row_adj[ids].nonzero()[0])
        span_row_line_ids = list(set(span_row_line_ids))
        for row_id, text_ids in enumerate(shared_row_line_ids): # 逐行枚举 line ids
            for idx in text_ids:
                if idx in span_row_line_ids and line_valid[idx] == 1:
                    span_row_ids.append(row_id)
                    break
        
        # find all span col line ids
        span_col_line_ids = []
        for ids in cell_ids:
            span_col_line_ids.extend(col_adj[ids].nonzero()[0]) # 与当前cell同列的line ids
        span_col_line_ids = list(set(span_col_line_ids))
        for col_id, text_ids in enumerate(shared_col_line_ids): # 枚举不跨列的line ids
            for idx in text_ids:
                if idx in span_col_line_ids and line_valid[idx] == 1:
                    span_col_ids.append(col_id)
                    break
        
        start_row = min(span_row_ids)
        end_row = max(span_row_ids)
        start_col = min(span_col_ids)
        end_col = max(span_col_ids)
        layout[start_row:end_row+1, start_col:end_col+1] = start_id + index

        sorted(cell_ids)
        cell_ids = [str(item) for item in cell_ids]
        span = '%d-%d-%d-%d' % (start_col, start_row, end_col, end_row)
        assign_text_id[span] = '-'.join(cell_ids)

    layout = format_layout(layout)
    # cells
    cells = list()
    num_cells = layout.max() + 1
    for cell_id in range(num_cells):
        ys, xs = np.split(np.argwhere(layout==cell_id), 2, 1)
        start_row = ys.min()
        end_row = ys.max()
        start_col = xs.min()
        end_col = xs.max()
        span = '%d-%d-%d-%d' % (start_col, start_row, end_col, end_row)
        if span in assign_text_id.keys():
            transcript = assign_text_id[span]
        else:
            transcript = ''
        cell = dict(
            col_start_idx=int(start_col),
            row_start_idx=int(start_row),
            col_end_idx=int(end_col),
            row_end_idx=int(end_row),
            transcript=transcript
        )
        cells.append(cell)
        
    table = dict(
        layout=layout.tolist(),
        cells=cells
    )
    return table


def extend_text_lines(cells, lines):
    def segmentation_to_polygon(segmentation):
        polygon = Polygon.Polygon()
        for contour in segmentation:
            polygon = polygon + Polygon.Polygon(contour)
        return polygon

    lines = copy.deepcopy(lines)

    cells_poly = [segmentation_to_polygon(item['segmentation']) for item in cells]
    # lines_poly = [segmentation_to_polygon(item['segmentation']) for item in lines]
    lines_poly = [segmentation_to_polygon([item]) for item in lines]

    assign_ids = dict()
    for idx in range(len(cells_poly)):
        assign_ids[idx] = list()

    # 计算cell和line的覆盖
    for line_idx, line_poly in enumerate(lines_poly):
        if line_poly.area() == 0:
            continue
        line_area = line_poly.area()
        max_overlap = 0
        max_overlap_idx = None
        for cell_idx, cell_poly in enumerate(cells_poly):
            overlap = (cell_poly & line_poly).area() / line_area
            if overlap > max_overlap:
                max_overlap_idx = cell_idx
                max_overlap = overlap
            if overlap > 0.99:
                break
        if max_overlap > 0:
            assign_ids[max_overlap_idx].append(line_idx)
    
    for idx, value in assign_ids.items():
        sorted(value)
        value = [str(item) for item in value]
        cells[idx]['transcript'] = '-'.join(value)
        
    return cells