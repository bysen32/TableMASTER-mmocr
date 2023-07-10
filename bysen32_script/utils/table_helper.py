import Polygon
import shapely 
import cv2



def filter_overlay(pts, iou_threshold=0.5): # 将重复的多边形去掉
    new_pts = []
    for i in range(len(pts)):
        polyi = Polygon.Polygon(pts[i])
        if polyi.area() < 1:
            continue
        overlay = False
        for j in range(i+1, len(pts)):
            polyj = Polygon.Polygon(pts[j])
            iou = (polyi & polyj).area() / (polyi | polyj).area()
            if iou > iou_threshold:
                overlay = True
                break
        if not overlay:
            new_pts.append(pts[i])
    return new_pts

# def filter_contain(pts):
#     import shapely
#     new_pts = []
#     for i in range(len(pts)):
#         polyi = shapely.geometry.Polygon(pts[i])
#         if polyi.area < 1:
#             continue
#         contain = False
#         for j in range(len(pts)):
#             if i == j:
#                 continue
#             polyj = shapely.geometry.Polygon(pts[j])
#             if polyi.contains(polyj): # 全包含计算
#                 contain = True
#                 break
#         if not contain:
#             new_pts.append(pts[i])
#     return new_pts

def filter_contain(pts):
    new_pts = []
    for i in range(len(pts)):
        polyi = Polygon.Polygon(pts[i])
        if polyi.area() < 1:
            continue
        contain = False
        for j in range(len(pts)):
            if i == j:
                continue
            polyj = Polygon.Polygon(pts[j])
            if polyi.area() > polyj.area(): # 只有大的包含小的才计算 iou
                if (polyi & polyj).area() / polyj.area() > 0.95:
                    contain = True
                    break
        if not contain:
            new_pts.append(pts[i])
    return new_pts



def filter_empty(pts, lines):
    new_pts = []
    for i in range(len(pts)):
        polyi = Polygon.Polygon(pts[i])
        if polyi.area() < 1:
            continue
        empty = True
        for j in range(len(lines)):
            polyj = Polygon.Polygon(lines[j])
            iou = (polyi & polyj).area() / polyj.area()
            if iou >= 0.01: # 有交集计算
                empty = False
                break
        if not empty:
            new_pts.append(pts[i])
    return new_pts


# def correct_polygons(ptss):
#     new_ptss = []
#     for pts in ptss:
#         new_pts = polygon_helper.correct_polygon(pts)
#         new_ptss.append(new_pts)
#     return new_ptss


def correct_table(table):
    # 暂时没法解决，存在自相交多边形
    # table['row'] = correct_polygons(table['row'])
    # table['col'] = correct_polygons(table['col'])

    # 修复重叠关系的标注
    table['row'] = filter_overlay(table['row']) 
    table['col'] = filter_overlay(table['col'])
    table['line'] = filter_overlay(table['line'])
    table['cell'] = filter_overlay(table['cell'])

    # 修复包含关系的标注
    # table['row'] = filter_contain(table['row'])
    # table['col'] = filter_contain(table['col'])
    # table['line'] = filter_contain(table['line'])
    # table['cell'] = filter_contain(table['cell'])

    # 清除不包含任何line的 row/col
    # table['row'] = filter_empty(table['row'], table['line'])
    # table['col'] = filter_empty(table['col'], table['line'])

    return table

def polygons_all_simple(ptss):
    for pts in ptss:
        poly = shapely.geometry.Polygon(pts)
        if not poly.is_simple:
            return False
    return True

def contained_in(ptss1, ptss2):
    for pts in ptss1:
        poly = Polygon.Polygon(pts)
        contained = False
        for pts2 in ptss2:
            poly2 = Polygon.Polygon(pts2)
            iou = (poly & poly2).area() / min(poly.area(), poly2.area())
            if iou > 0.5:
                contained = True
                break
        if not contained:
            return False
    return True

def table_valid(table):
    if not polygons_all_simple(table['row']):
        return False
    if not polygons_all_simple(table['col']):
        return False

    # 所有的line必须都在row/col中
    if not contained_in(table['line'], table['row']):
        return False
    if not contained_in(table['line'], table['col']):
        return False

    return True

def table2label_valid(table, label):
    num_row = len(table['row'])
    num_col = len(table['col'])
    num_row_layout = len(label['layout'])
    num_col_layout = len(label['layout'][-1])
    if num_row != num_row_layout:
        return False
    if num_col != num_col_layout:
        return False
    return True


def resize_image_short(img, short_side=480):
    width, height = img.shape[1], img.shape[0]
    if height > short_side and width > short_side:
        # 短边缩小至480，需要注意极端宽高比情况
        if width >= height:
            scale = short_side * 1.0 / height
            height = short_side
            width = round(width * scale)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        else:
            scale = short_side * 1.0 / width
            width = short_side
            height = round(height * scale)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    elif height < short_side and width < short_side:
        # 短边放大至480，需要注意极端宽高比情况
        if height >= width:
            scale = short_side * 1.0 / width
            width = short_side
            height = round(height * scale)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        else:
            scale = short_side * 1.0 / height
            height = short_side
            width = round(width * scale)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    return img

def resize_image_long(img, long_side=480):
    width, height = img.shape[1], img.shape[0]
    if height >= long_side or width >= long_side:
        # 长边缩小至480，需要注意极端宽高比情况
        if height >= width:
            scale = long_side * 1.0 / height
            height = long_side
            width = round(width * scale)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        else:
            scale = long_side * 1.0 / width
            width = long_side
            height = round(height * scale)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    else:
        # 长边放大至480，需要注意极端宽高比情况
        if width >= height:
            scale = long_side * 1.0 / width
            width = long_side
            height = round(height * scale)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        else:
            scale = long_side * 1.0 / height
            height = long_side
            width = round(width * scale)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    return img