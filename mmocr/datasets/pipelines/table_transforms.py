from typing import Any
from mmdet.datasets.builder import PIPELINES
from Polygon import Polygon
import shapely

import os
import cv2
import random
import numpy as np
import copy
from utils_bobo.table2label import table2label
from utils_bobo.format_translate import table_to_html
from mmocr.datasets.utils.parser import build_empty_bbox_mask, align_bbox_mask, build_bbox_mask

def visual_table_resized_bbox(results):
    bboxes = results['img_info']['bbox']
    img = results['img']
    for bbox in bboxes:
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), thickness=1)
    return img

def visual_table_xywh_bbox(results):
    img = results['img']
    bboxes = results['img_info']['bbox']
    for bbox in bboxes:
        draw_bbox = np.empty_like(bbox)
        draw_bbox[0] = bbox[0] - bbox[2] / 2
        draw_bbox[1] = bbox[1] - bbox[3] / 2
        draw_bbox[2] = bbox[0] + bbox[2] / 2
        draw_bbox[3] = bbox[1] + bbox[3] / 2
        img = cv2.rectangle(img, (int(draw_bbox[0]), int(draw_bbox[1])), (int(draw_bbox[2]), int(draw_bbox[3])), (0, 255, 0), thickness=1)
    return img

@PIPELINES.register_module()
class TableResize:
    """Image resizing and padding for Table Recognition OCR, Table Structure Recognition.

    Args:
        height (int | tuple(int)): Image height after resizing.
        min_width (none | int | tuple(int)): Image minimum width
            after resizing.
        max_width (none | int | tuple(int)): Image maximum width
            after resizing.
        keep_aspect_ratio (bool): Keep image aspect ratio if True
            during resizing, Otherwise resize to the size height *
            max_width.
        img_pad_value (int): Scalar to fill padding area.
        width_downsample_ratio (float): Downsample ratio in horizontal
            direction from input image to output feature.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.
    """

    def __init__(self,
                 img_scale=None,
                 min_size=None,
                 ratio_range=None,
                 interpolation=None,
                 keep_ratio=True,
                 long_size=None):
        self.img_scale = img_scale
        self.min_size = min_size
        self.ratio_range = ratio_range
        self.interpolation = cv2.INTER_LINEAR
        self.long_size = long_size
        self.keep_ratio = keep_ratio

    def _get_resize_scale(self, w, h):
        if self.keep_ratio:
            if self.img_scale is None and isinstance(self.ratio_range, list): # 随机缩放
                choice_ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])
                return (int(w * choice_ratio), int(h * choice_ratio))
            elif isinstance(self.img_scale, tuple) and -1 in self.img_scale: # 设定宽或高，另一边自适应
                if self.img_scale[0] == -1:
                    resize_w = w / h * self.img_scale[1]
                    return (int(resize_w), self.img_scale[1])
                else:
                    resize_h = h / w * self.img_scale[0]
                    return (self.img_scale[0], int(resize_h))
            else:
                return (int(w), int(h))
        else:
            if isinstance(self.img_scale, tuple): # 设定宽和高
                return self.img_scale
            else:
                raise NotImplementedError

    def _resize_bboxes(self, results):
        img_shape = results['img_shape']
        if 'img_info' in results.keys():
            # train and validate phase
            if results['img_info'].get('bbox', None) is not None:
                bboxes = results['img_info']['bbox']
                scale_factor = results['scale_factor']
                # bboxes[..., 0::2], bboxes[..., 1::2] = \
                #     bboxes[..., 0::2] * scale_factor[1], bboxes[..., 1::2] * scale_factor[0]
                bboxes[..., 0::2] = np.clip(bboxes[..., 0::2] * scale_factor[1], 0, img_shape[1]-1)
                bboxes[..., 1::2] = np.clip(bboxes[..., 1::2] * scale_factor[0], 0, img_shape[0]-1)
                results['img_info']['bbox'] = bboxes
            else:
                raise ValueError('results should have bbox keys.')
        else:
            # testing phase
            pass

    def _resize_img(self, results):
        img = results['img']
        h, w, _ = img.shape

        if self.min_size is not None:
            if w > h:
                w = self.min_size / h * w
                h = self.min_size
            else:
                h = self.min_size / w * h
                w = self.min_size

        if self.long_size is not None:
            if w < h:
                w = self.long_size / h * w
                h = self.long_size
            else:
                h = self.long_size / w * h
                w = self.long_size

        img_scale = self._get_resize_scale(w, h)
        if img_scale[0] >= img.shape[1]: # 放大
            resize_img = cv2.resize(img, img_scale, interpolation=cv2.INTER_CUBIC)
        else: # 缩小
            resize_img = cv2.resize(img, img_scale, interpolation=cv2.INTER_LINEAR)

        scale_factor = (resize_img.shape[0] / img.shape[0], resize_img.shape[1] / img.shape[1])

        results['img'] = resize_img
        results['img_shape'] = resize_img.shape
        results['pad_shape'] = resize_img.shape
        pre_scale_factor = results.get('scale_factor', (1.0, 1.0))
        results['scale_factor'] = scale_factor
        self._resize_bboxes(results)
        results['scale_factor'] = pre_scale_factor[0] * scale_factor[0], pre_scale_factor[1] * scale_factor[1]
        # results['keep_ratio'] = self.keep_ratio

    def __call__(self, results):
        self._resize_img(results)
        return results
    

@PIPELINES.register_module()
class TableAspect:
    """Image resizing and padding for Table Recognition OCR, Table Structure Recognition.

    Args:
        height (int | tuple(int)): Image height after resizing.
        min_width (none | int | tuple(int)): Image minimum width
    """

    def __init__(self,
                 ratio=(0.75, 1.3333333333333333),
                 p=0.5
                 ):
        self.p = p
        self.ratio = ratio

    def _get_resize_scale(self, w, h):

        if random.random() < self.p:
            choice_ratio = random.uniform(self.ratio[0], self.ratio[1])
            return (int(w * choice_ratio), int(h))
        else:
            return (w, h)

    def _resize_bboxes(self, results):
        img_shape = results['img_shape']
        if 'img_info' in results.keys():
            # train and validate phase
            if results['img_info'].get('bbox', None) is not None:
                bboxes = results['img_info']['bbox']
                scale_factor = results['scale_factor']
                # bboxes[..., 0::2], bboxes[..., 1::2] = \
                #     bboxes[..., 0::2] * scale_factor[1], bboxes[..., 1::2] * scale_factor[0]
                bboxes[..., 0::2] = np.clip(bboxes[..., 0::2] * scale_factor[1], 0, img_shape[1]-1)
                bboxes[..., 1::2] = np.clip(bboxes[..., 1::2] * scale_factor[0], 0, img_shape[0]-1)
                results['img_info']['bbox'] = bboxes
            else:
                raise ValueError('results should have bbox keys.')
        else:
            # testing phase
            pass

    def _resize_img(self, results):
        img = results['img']
        h, w, _ = img.shape

        img_scale = self._get_resize_scale(w, h)
        if img_scale[0] >= img.shape[1]: # 放大
            resize_img = cv2.resize(img, img_scale, interpolation=cv2.INTER_CUBIC)
        else: # 缩小
            resize_img = cv2.resize(img, img_scale, interpolation=cv2.INTER_LINEAR)
        scale_factor = (resize_img.shape[0] / img.shape[0], resize_img.shape[1] / img.shape[1])

        results['img'] = resize_img
        results['img_shape'] = resize_img.shape
        results['pad_shape'] = resize_img.shape
        pre_scale_factor = results.get('scale_factor', (1.0, 1.0))
        results['scale_factor'] = scale_factor
        self._resize_bboxes(results)
        results['scale_factor'] = pre_scale_factor[0] * scale_factor[0], pre_scale_factor[1] * scale_factor[1]
        # results['keep_ratio'] = self.keep_ratio

    def __call__(self, results):
        if random.random() < self.p:
            self._resize_img(results)
        return results



@PIPELINES.register_module()
class RandomLineMask:
    def __init__(self, ratio=(0.3, 0.7), p=0.5):
        self.p = p
        self.ratio = ratio 
    
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
            elif token == '<eb></eb>':
                count += 1
            else:
                pass
        return count
    
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
    
    def _update_label(self, results):
        rc_label = results['rc_label']

        # 基于rc_label重新计算！
        layout_label = table2label(rc_label)
        layout_label['layout'] = np.array(layout_label['layout'])
        html_label = table_to_html(layout_label)

        token_list = html_label['html']['structure']['tokens']
        merged_token = self.merge_token(token_list)
        cells = layout_label['cells']
        encoded_token = self.insert_empty_bbox_token(merged_token, cells)

        cell_num = len(cells)
        cell_count = self.count_merge_token_nums(encoded_token)
        assert cell_num == cell_count

        # 得到 text 与 bboxes
        text = ','.join(encoded_token)
        bboxes = []
        for cell in cells:
            bboxes.append(cell['bbox'])

        # label_htmls = format_html(html_label)

        empty_bbox_mask = build_empty_bbox_mask(bboxes)
        bboxes, empty_bbox_mask = align_bbox_mask(bboxes, empty_bbox_mask, text)
        empty_bbox_mask = np.array(empty_bbox_mask)

        bbox_masks = build_bbox_mask(text)
        bbox_masks = bbox_masks * empty_bbox_mask

        results['img_info']['bbox']       = np.array(bboxes)
        results['img_info']['bbox_masks'] = bbox_masks
        results['text'] = text

    def _random_line_mask(self, results):
        img = results['img']
        rc_label = copy.deepcopy(results['img_info']['rc_label'])

        unique_values, value_counts = np.unique(img.reshape(-1, img.shape[-1]), axis=0, return_counts=True)
        bg_value = unique_values[np.argmax(value_counts)].tolist()

        lines = rc_label['line']
        line_cnt = len(rc_label['line'])
        mask_count = random.uniform(self.ratio[0], self.ratio[1]) * len(lines)

        if line_cnt < 10:
            mask_count = int(line_cnt * 0.5)
        if line_cnt < 50:
            mask_count = int(line_cnt * 0.4)
        elif line_cnt < 100:
            mask_count = int(line_cnt * 0.3)
        elif line_cnt < 200:
            mask_count = int(line_cnt * 0.2)
        else:
            mask_count = int(line_cnt * 0.1)

        for _ in range(mask_count):
            idx = random.randint(0, len(lines)-1)
            line_seg = lines[idx]
            cv2.fillPoly(img, np.array([line_seg], dtype=np.int32), color=bg_value)
            lines.pop(idx)
        
        results['img'] = img
        results['rc_label'] = rc_label
    
    def __call__(self, results):
        if random.random() < self.p:
            self._random_line_mask(results)
            self._update_label(results)
        return results



@PIPELINES.register_module()
class TableRotate:
    def __init__(self, rotate_ratio=(-1, 1), p=0.5):
        self.rotate_ratio = rotate_ratio
        self.p = p
    
    def _rotate_img(self, results):
        img = results['img']
        h, w = img.shape[:2]
        center = (w//2, h//2)
        ratio = random.uniform(self.rotate_ratio[0], self.rotate_ratio[1])
        M = cv2.getRotationMatrix2D(center, ratio, 1.0)
        rotate_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,\
            borderMode=cv2.BORDER_REPLICATE)
        results['img'] = rotate_img
        results['img_shape'] = rotate_img.shape
        results['pad_shape'] = rotate_img.shape
        # results['rotate_ratio'] = ratio
        results['rotate_matrix'] = M

    def _rotate_bboxes(self, results):
        if 'img_info' in results.keys():
            if results['img_info'].get('bbox', None) is not None: # update ['img_info']['bbox]
                bboxes = results['img_info']['bbox']
                M = results['rotate_matrix']
                for idx, bbox in enumerate(bboxes):
                    x0, y0, x1, y1 = bbox
                    [x0, y0], [x1, y1] = cv2.transform(np.array([[[x0, y0], [x1, y1]]]), M).squeeze().astype(np.int32)
                    bboxes[idx] = [x0, y0, x1, y1]
                results['img_info']['bbox'] = bboxes
            else:
                raise ValueError('results should have bbox keys.')
        else:
            pass

    def __call__(self, results):
        if random.random() < self.p:
            self._rotate_img(results)
            self._rotate_bboxes(results)
        return results



@PIPELINES.register_module()
class RandomLineFill(RandomLineMask):
    def __init__(self, ratio=0.1, p=0.6):
        self.p = p
        self.ratio = ratio
    
    def _random_line_fill(self, results):
        rc_label = results['rc_label']
        img_array = results['img']
        
        def segmentation_to_polygon(segmentation):
            polygon = shapely.Polygon()
            for contour in segmentation:
                polygon = polygon.union(shapely.Polygon(contour))
            return polygon

        row_polys = [segmentation_to_polygon([row]) for row in rc_label['row']]
        col_polys = [segmentation_to_polygon([col]) for col in rc_label['col']]
        line_polys = [segmentation_to_polygon([line]) for line in rc_label['line']]

        H, W = img_array.shape[:2]

        temp_img = copy.deepcopy(img_array)
        min_area = float('inf')
        line_num = len(rc_label['line'])
        for row_poly in row_polys:
            for col_poly in col_polys:
                try:
                    if not row_poly.intersects(col_poly):
                        continue
                    cell_poly = row_poly.intersection(col_poly)
                except:
                    continue

                if cell_poly.area < 0.002 * H * W:
                    continue

                min_area = min(min_area, cell_poly.area)

                for line_poly in line_polys:
                    if line_poly.intersects(cell_poly): # 一点重叠都不要有
                        break
                else:
                    if random.random() < self.ratio:
                        line_idxs = list(range(len(line_polys)))
                        random.shuffle(line_idxs)
                        cell_w, cell_h = cell_poly.bounds[2]-cell_poly.bounds[0], cell_poly.bounds[3]-cell_poly.bounds[1]
                        cell_cx, cell_cy = cell_poly.centroid.coords[0]
                        cell_cx, cell_cy = int(cell_cx), int(cell_cy)
                        for idx in line_idxs:
                            line_poly = line_polys[idx]
                            line_w, line_h = line_poly.bounds[2]-line_poly.bounds[0], line_poly.bounds[3]-line_poly.bounds[1]
                            if line_poly.area < cell_poly.area * 0.25 and  cell_w * 0.2 < line_w < cell_w * 0.5 and  cell_h * 0.2 < line_h < cell_h * 0.8:
                                x0, y0, x1, y1 = line_poly.bounds
                                w, h = int(x1-x0+1), int(y1-y0+1)
                                x0, y0 = int(x0), int(y0)
                                cell_x0, cell_y0 = int(cell_cx-w//2), int(cell_cy-h//2)
                                if cell_x0 < 0 or cell_y0 < 0 or cell_x0+w > W or cell_y0+h > H:
                                    continue
                                try:
                                    img_array[cell_y0:cell_y0+h, cell_x0:cell_x0+w] = img_array[y0:y0+h, x0:x0+w]
                                    line_poly = shapely.Polygon([[cell_x0, cell_y0], [cell_x0, cell_y0+h],
                                                        [cell_x0+w, cell_y0+h], [cell_x0+w, cell_y0]])
                                    rc_label['line'].append(cell_poly.intersection(line_poly).exterior.coords[:-1])
                                except:
                                    pass
                                break
        results['img'] = img_array
        results['rc_label'] = rc_label
    
    def __call__(self, results):
        if random.random() < self.p:
            self._random_line_fill(results)
            self._update_label(results)
        return results

@PIPELINES.register_module()
class BlackTest:
    def __call__(self, results):
        rc_label = results['rc_label']
        img = results['img']
        for line in rc_label['line']:
            cv2.polylines(img, [np.array(line, dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=1)
        results['img'] = img
        return results


@PIPELINES.register_module()
class RandomRCMask(RandomLineMask):
    def __init__(self, ratio=0.5, p=0.5):
        self.p = p
        self.ratio = ratio
    
    def _remove_overlay_line(self, seg, lines, iou_threshold=0.5):
        new_line = []
        rc_poly = Polygon(seg)
        for i in range(len(lines)):
            line_poly = Polygon(lines[i])
            iou = (rc_poly & line_poly).area() / line_poly.area()
            if iou > iou_threshold:
                continue
            new_line.append(lines[i])
        return new_line
    
    
    def _boundingbox_crop(self, img_array, rc_label):
        pts = []
        if rc_label['is_wireless']:
            for line in rc_label['line']:
                pts.extend(line)
        else:
            for cell in rc_label['cell']:
                pts.extend(cell)
        pts = np.array(pts, dtype=np.int32)
        gap = 10
        x0, y0, x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max())
        x0 = max(0, x0-gap)
        y0 = max(0, y0-gap)
        x1 = min(img_array.shape[1], x1+gap)
        y1 = min(img_array.shape[0], y1+gap)
        img_array = img_array[y0:y1, x0:x1]

        for idx, row in enumerate(rc_label['row']):
            row = np.array(row)
            row[:, 0] -= x0
            row[:, 1] -= y0
            rc_label['row'][idx] = row.tolist()
        for idx, col in enumerate(rc_label['col']):
            col = np.array(col)
            col[:, 0] -= x0
            col[:, 1] -= y0
            rc_label['col'][idx] = col.tolist()
        for idx, line in enumerate(rc_label['line']):
            line = np.array(line)
            line[:, 0] -= x0
            line[:, 1] -= y0
            rc_label['line'][idx] = line.tolist()
        for idx, cell in enumerate(rc_label['cell']):
            cell = np.array(cell)
            cell[:, 0] -= x0
            cell[:, 1] -= y0
            rc_label['cell'][idx] = cell.tolist()

        return img_array, rc_label

    
    def _random_rc_mask(self, results):
        rc_label = results['rc_label']
        img_array = results['img']
        ratio = self.ratio

        ori_img_array = copy.deepcopy(img_array)
        ori_rc_label = copy.deepcopy(rc_label)

        unique_values, value_counts = np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0, return_counts=True)
        bg_color = unique_values[np.argmax(value_counts)]

        if random.random() < 0.5:
            top_cnt = 0
            bottom_cnt = random.randint(0, int(len(rc_label['row'])*ratio))
            new_row = []
            for i in range(len(rc_label['row'])):
                if i < top_cnt or i >= len(rc_label['row']) - bottom_cnt:
                    row_seg = rc_label['row'][i]
                    cv2.fillPoly(img_array, np.array([row_seg], dtype=np.int32), color=bg_color.tolist())
                    rc_label['line'] = self._remove_overlay_line(row_seg, rc_label['line'])
                    rc_label['cell'] = self._remove_overlay_line(row_seg, rc_label['cell'])
                else:
                    new_row.append(rc_label['row'][i])
            rc_label['row'] = new_row
        else:
            left_cnt = 0
            right_cnt = random.randint(0, int(len(rc_label['col'])*ratio))
            new_col = []
            for i in range(len(rc_label['col'])):
                if i < left_cnt or i >= len(rc_label['col']) - right_cnt:
                    col_seg = rc_label['col'][i]
                    cv2.fillPoly(img_array, np.array([col_seg], dtype=np.int32), color=bg_color.tolist())
                    rc_label['line'] = self._remove_overlay_line(col_seg, rc_label['line'])
                    rc_label['cell'] = self._remove_overlay_line(col_seg, rc_label['cell'])
                else:
                    new_col.append(rc_label['col'][i])
            rc_label['col'] = new_col

        img_array, rc_label = self._boundingbox_crop(img_array, rc_label)

        struct_label = table2label(rc_label)
        ori_struct_label = table2label(ori_rc_label)
        n_row, n_col = np.array(struct_label['layout']).shape
        ori_n_row, ori_n_col = np.array(ori_struct_label['layout']).shape



        if n_row == ori_n_row or n_col == ori_n_col: # keep one dimension
            results['img'] = img_array
            results['rc_label'] = rc_label
            results['img_shape'] = img_array.shape
            results['ori_shape'] = img_array.shape
        else:
            results['img'] = ori_img_array
            results['rc_label'] = ori_rc_label
            results['img_shape'] = ori_img_array.shape
            results['ori_shape'] = ori_img_array.shape

    def __call__(self, results):
        if random.random() < self.p:
            self._random_rc_mask(results)
            self._update_label(results)
        return results



@PIPELINES.register_module()
class TablePad:
    """Pad the image & mask.
    Two padding modes:
    (1) pad to fixed size.
    (2) pad to the minium size that is divisible by some number.
    """
    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=None,
                 keep_ratio=False,
                 return_mask=False,
                 mask_ratio=2,
                 train_state=True,
                 ):
        self.size = size[::-1]
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.keep_ratio = keep_ratio
        self.return_mask = return_mask
        self.mask_ratio = mask_ratio
        self.training = train_state
        # only one of size or size_divisor is valid.
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad(self, img, size, pad_val):
        if not isinstance(size, tuple):
            raise NotImplementedError

        if len(size) < len(img.shape):
            shape = size + (img.shape[-1], )
        else:
            shape = size

        pad = np.empty(shape, dtype=img.dtype)
        pad[...] = pad_val

        h, w = img.shape[:2]
        size_w, size_h = size[:2]
        if h > size_h or w > size_w:
            if self.keep_ratio:
                if h / size_h > w / size_w:
                    size = (int(w / h * size_h), size_h)
                else:
                    size = (size_w, int(h / w * size_w))
            img = cv2.resize(img, size[::-1], cv2.INTER_LINEAR)
        pad[:img.shape[0], :img.shape[1], ...] = img
        if self.return_mask:
            mask = np.empty(size, dtype=img.dtype)
            mask[...] = 0
            mask[:img.shape[0], :img.shape[1]] = 1

            # mask_ratio is mean stride of backbone in (height, width)
            if isinstance(self.mask_ratio, int):
                mask = mask[::self.mask_ratio, ::self.mask_ratio]
            elif isinstance(self.mask_ratio, tuple):
                mask = mask[::self.mask_ratio[0], ::self.mask_ratio[1]]
            else:
                raise NotImplementedError

            mask = np.expand_dims(mask, axis=0)
        else:
            mask = None
        return pad, mask

    def _divisor(self, img, size_divisor, pad_val):
        pass

    def _pad_img(self, results):
        if self.size is not None:
            padded_img, mask = self._pad(results['img'], self.size, self.pad_val)
        elif self.size_divisor is not None:
            raise NotImplementedError
        results['img'] = padded_img
        results['mask'] = mask
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        self._pad_img(results)
        #visual_img = visual_table_resized_bbox(results)
        #cv2.imwrite('/data_0/cache/{}_visual.jpg'.format(os.path.basename(results['filename']).split('.')[0]), visual_img)
        # if self.training:
            # scaleBbox(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(size={}, size_divisor={}, pad_val={})'.format(
            self.size, self.size_divisor, self.pad_val)
        return repr_str


def xyxy2xywh(bboxes):
    """
    Convert coord (x1,y1,x2,y2) to (x,y,w,h).
    where (x1,y1) is top-left, (x2,y2) is bottom-right.
    (x,y) is bbox center and (w,h) is width and height.
    :param bboxes: (x1, y1, x2, y2)
    :return:
    """
    new_bboxes = np.empty_like(bboxes)
    new_bboxes[..., 0] = (bboxes[..., 0] + bboxes[..., 2]) / 2 # x center
    new_bboxes[..., 1] = (bboxes[..., 1] + bboxes[..., 3]) / 2 # y center
    new_bboxes[..., 2] = bboxes[..., 2] - bboxes[..., 0] # width
    new_bboxes[..., 3] = bboxes[..., 3] - bboxes[..., 1] # height
    return new_bboxes


def normalize_bbox(bboxes, img_shape):
    bboxes[..., 0], bboxes[..., 2] = bboxes[..., 0] / img_shape[1], bboxes[..., 2] / img_shape[1]
    bboxes[..., 1], bboxes[..., 3] = bboxes[..., 1] / img_shape[0], bboxes[..., 3] / img_shape[0]
    return bboxes


@PIPELINES.register_module()
class TableBboxEncode:
    """Encode table bbox for training.
    convert coord (x1,y1,x2,y2) to (x,y,w,h)
    normalize to (0,1)
    adjust key 'bbox' and 'bbox_mask' location in dictionary 'results'
    """
    def __init__(self):
        pass

    def __call__(self, results):
        bboxes = results['img_info']['bbox']
        bboxes = xyxy2xywh(bboxes)
        img_shape = results['img'].shape
        bboxes = normalize_bbox(bboxes, img_shape)
        flag = self.check_bbox_valid(bboxes)
        if not flag:
            print('Box invalid in {}'.format(results['filename']))
        results['img_info']['bbox'] = bboxes
        self.adjust_key(results)
        # self.visual_normalized_bbox(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

    def check_bbox_valid(self, bboxes):
        low = (bboxes >= 0.) * 1
        high = (bboxes <= 1.) * 1
        matrix = low + high
        for idx, m in enumerate(matrix):
            if m.sum() != 8:
                return False
        return True

    def visual_normalized_bbox(self, results):
        """
        visual after normalized bbox in results.
        :param results:
        :return:
        """
        save_path = '/data_0/cache/{}_normalized.jpg'.\
            format(os.path.basename(results['filename']).split('.')[0])
        img = results['img']
        img_shape = img.shape
        # x,y,w,h
        bboxes = results['img_info']['bbox']
        bboxes[..., 0::2] = bboxes[..., 0::2] * img_shape[1]
        bboxes[..., 1::2] = bboxes[..., 1::2] * img_shape[0]
        # x,y,x,y
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[..., 0] = bboxes[..., 0] - bboxes[..., 2] / 2
        new_bboxes[..., 1] = bboxes[..., 1] - bboxes[..., 3] / 2
        new_bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2] / 2
        new_bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3] / 2
        # draw
        for new_bbox in new_bboxes:
            img = cv2.rectangle(img, (int(new_bbox[0]), int(new_bbox[1])),
                                   (int(new_bbox[2]), int(new_bbox[3])), (0, 255, 0), thickness=1)
        cv2.imwrite(save_path, img)

    def adjust_key(self, results):
        """
        Adjust key 'bbox' and 'bbox_mask' location in dictionary 'results'.
        :param results:
        :return:
        """
        bboxes = results['img_info'].pop('bbox')
        bboxes_masks = results['img_info'].pop('bbox_masks')
        results['bbox'] = bboxes
        results['bbox_masks'] = bboxes_masks
        return results

