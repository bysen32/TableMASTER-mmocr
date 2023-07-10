import numpy as np

def remove_empty_bboxes(bboxes):
    """
    remove [0., 0., 0., 0.] in structure master bboxes.
    len(bboxes.shape) must be 2.
    :param bboxes:
    :return:
    """
    new_bboxes = []
    for bbox in bboxes:
        if sum(bbox) == 0.:
            continue
        new_bboxes.append(bbox)
    return np.array(new_bboxes)

def format_tokens(master_token):
    # creat virtual master token
    virtual_master_token_list = []
    # insert virtual master token
    master_token_list = master_token.split(',')
    if master_token_list[-1] == '</tbody>':
        # complete predict(no cut by max length)
        # This situation insert virtual master token will drop TEDs score in val set.
        # So we will not extend virtual token in this situation.

        # fake extend virtual
        master_token_list[:-1].extend(virtual_master_token_list)

        # real extend virtual
        # master_token_list = master_token_list[:-1]
        # master_token_list.extend(virtual_master_token_list)
        # master_token_list.append('</tbody>')

    elif master_token_list[-1] == '<td></td>':
        master_token_list.append('</tr>')
        master_token_list.extend(virtual_master_token_list)
        master_token_list.append('</tbody>')
    else:
        master_token_list.extend(virtual_master_token_list)
        master_token_list.append('</tbody>')

    return master_token_list

def get_html(tokens_list, bboxs):
    new_tokens_list = []
    cells = []

    count = 0
    for token in tokens_list:
        if token == '<td></td>' or token == '<td':
            bbox = bboxs[count]
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            bbox_new = [x-w/2, y-h/2, x+w/2, y+h/2]
            count += 1
            if token == '<td></td>':
                new_tokens_list.extend(['<td>', '</td>'])
            else:
                new_tokens_list.append('<td')
            cell = dict()
            cell['tokens'] = ''
            cell['bbox'] = list(bbox_new)
            cells.append(cell)
        elif token in ['<eb></eb>', '<eb1></eb1>', '<eb2></eb2>', '<eb3></eb3>', '<eb4></eb4>', '<eb5></eb5>', '<eb6></eb6>', '<eb7></eb7>', '<eb8></eb8>', '<eb9></eb9>', '<eb10></eb10>']:
            new_tokens_list.extend(['<td>', '</td>'])
            cell = dict()
            cell['tokens'] = ''
            cell['bbox'] = [0, 0, 0, 0]
            cells.append(cell)
        else:
            new_tokens_list.append(token)

    html = dict(
        html=dict(
            cells=cells,
            structure=dict(
                tokens=new_tokens_list
            )
        )
    )
    return html

def html_to_table(html):
    tokens = html['html']['structure']['tokens']

    layout = [[]]

    def extend_table(x, y):
        # assert (x >= 0) and (y >= 0)
        nonlocal layout

        if x >= len(layout[0]):
            for row in layout:
                row.extend([-1] * (x - len(row) + 1))
        
        if y >= len(layout):
            for _ in range(y - len(layout) + 1):
                layout.append([-1] * len(layout[0]))

    def set_cell_val(x, y, val):
        # assert (x >= 0) and (y >= 0)
        nonlocal layout
        extend_table(x, y)
        layout[y][x] = val

    def get_cell_val(x, y):
        # assert (x >= 0) and (y >= 0)
        nonlocal layout
        extend_table(x, y)
        return layout[y][x]

    def parse_span_val(token):
        span_val = int(token[token.index('"') + 1:token.rindex('"')])
        return span_val

    def maskout_left_rows():
        nonlocal row_idx, layout
        layout = layout[:max(row_idx+1, 1)]

    row_idx = -1
    col_idx = -1
    line_idx = -1
    inside_head = False
    inside_body = False
    head_rows = list()
    body_rows = list()
    col_span = 1
    row_span = 1
    for token in tokens:
        if token == '<thead>':
            inside_head = True
            maskout_left_rows()
        elif token == '</thead>':
            inside_head = False
            maskout_left_rows()
        elif token == '<tbody>':
            inside_body = True
            maskout_left_rows()
        elif token == '</tbody>':
            inside_body = False
            maskout_left_rows()
        elif token == '<tr>':
            row_idx += 1
            col_idx = -1
            if inside_head:
                head_rows.append(row_idx)
            if inside_body:
                body_rows.append(row_idx)
        elif token in ['<td>', '<td']:
            line_idx += 1
            col_idx += 1
            row_span = 1
            col_span = 1
            while get_cell_val(col_idx, row_idx) != -1:
                col_idx += 1
        elif 'colspan' in token:
            col_span = parse_span_val(token)
        elif 'rowspan' in token:
            row_span = parse_span_val(token)
        elif token == '</td>':
            for cur_row_idx in range(row_idx, row_idx + row_span):
                for cur_col_idx in range(col_idx, col_idx + col_span):
                    set_cell_val(cur_col_idx, cur_row_idx, line_idx)
            col_idx += col_span - 1

    # check_continuous(head_rows)
    # check_continuous(body_rows)
    # assert len(set(head_rows) | set(body_rows)) == len(layout)
    layout = np.array(layout)
    # assert np.all(layout >= 0)


    # ↓↓↓↓↓↓↓↓↓↓↓  这里处理所有的-1  ↓↓↓↓↓↓↓↓↓↓↓
    # n_row, n_col = layout.shape
    # forbiden = np.zeros((n_row, n_col))
    # for i in range(n_row): # 标记span列
    #     for j in range(n_col-1):
    #         if layout[i, j] == layout[i, j+1] and layout[i, j] != -1:
    #             forbiden[i, j] = forbiden[i, j+1] = 1
    # for i in range(n_row):
    #     for j in range(n_col):
    #         if layout[i, j] == -1:
    #             if j > 0 and forbiden[i, j-1] == 0:
    #                 layout[i, j] = layout[i, j-1] # 继承左边的单元格
    #             elif i > 0 :
    #                 layout[i, j] = layout[i-1, j] # 继承上边的单元格

    # # 处理仍然存在的-1
    # for i in range(n_row):
    #     for j in range(n_col):
    #         if layout[i, j] == -1:
    #             if layout.max()+1 < len(html['html']['cells']): # 还能合法赋值
    #                 layout[i, j] = layout.max()+1
    #             else:
    #                 break
    # ↑↑↑↑↑↑↑↑↑↑↑↑ 处理完毕 ↑↑↑↑↑↑↑↑↑↑↑↑

    cells_info = list()
    for cell_idx, cell in enumerate(html['html']['cells']):
        transcript = cell['tokens']
        cell_info = dict(
            transcript=transcript
        )
        if 'bbox' in cell:
            x1, y1, x2, y2 = cell['bbox']
            cell_info['bbox'] = [x1, y1, x2, y2]
            cell_info['segmentation'] = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]
        cells_info.append(cell_info)
    
    table = dict(
        layout=layout,
        cells=cells_info,
        head_rows=head_rows,
        body_rows=body_rows
    )
    return table

def format_table_1(table): 
    layout = table['layout']
    num = layout.max() + 1
    idx = 0
    new_cells = []
    cell_cord = set()
    for i, row in enumerate(layout):
        for j, cell_id in enumerate(row):
            if cell_id == -1:
                layout[i, j] = num + idx
                idx += 1
                empty_cell = dict(
                    col_start_idx=j,
                    row_start_idx=i,
                    col_end_idx=j,
                    row_end_idx=i,
                    transcript = '',
                    bbox = [0, 0, 0, 0],
                    segmentation = [[[0, 0], [0, 0], [0, 0], [0, 0]]]
                )
                new_cells.append(empty_cell)
            else:
                if cell_id not in cell_cord:
                    cell_cord.add(cell_id)
                    new_cells.append(table['cells'][cell_id])

    new_layout = format_layout(layout)
    assert len(new_cells) == new_layout.max() + 1

    table = dict(
        layout=new_layout,
        cells=new_cells
    )

    return table

def format_table(table): 
    layout = table['layout']

    cells = table['cells']
    # num_cells = layout.max() + 1
    try:
        num_cells = max(len(cells), layout.max() + 1)
    except:
        num_cells = len(cells)
    for cell_id in range(num_cells):
        ys, xs = np.split(np.argwhere(layout==cell_id), 2, 1)
        start_row = ys.min()
        end_row = ys.max()
        start_col = xs.min()
        end_col = xs.max()
        cell = dict(
            col_start_idx=int(start_col),
            row_start_idx=int(start_row),
            col_end_idx=int(end_col),
            row_end_idx=int(end_row),
        )
        cells[cell_id].update(cell)
        
    table = dict(
        layout=layout.tolist(),
        cells=cells
    )
    return table

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