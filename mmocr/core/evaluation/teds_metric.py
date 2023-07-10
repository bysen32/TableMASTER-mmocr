import numpy as np
from . import teds_utils as util

from .utils_teds.utils import extend_text_lines
from .utils_teds.cal_f1 import table_to_relations, evaluate_f1
from .utils_teds.metric import TEDSMetric
from .utils_teds.format_translate import table_to_html, format_html

def cal_pred(pred_info, label_info):
    
    tokens_str = pred_info['text']
    bbox = pred_info['bbox']

    new_bbox = util.remove_empty_bboxes(bbox)
    tokens_list = util.format_tokens(tokens_str)

    html = util.get_html(tokens_list, new_bbox)

    table = util.html_to_table(html)

    try:
        pred = util.format_table(table)
    except:
        ### 去除layout中的-1
        # print(img)
        table = util.format_table_1(table)
        pred = util.format_table(table)

    # extend text line to predicted result
    pred['cells'] = extend_text_lines(pred['cells'], label_info['line'])
    pred['layout'] = np.array(pred['layout'])

    # calculate F1-Measure
    pred_relations = table_to_relations(pred)

    # calculate TEDS-Struct
    pred_htmls = table_to_html(pred)
    pred_htmls = format_html(pred_htmls)

    return pred_relations, pred_htmls

def eval_teds_metric(pred_relations_list, label_relations_list, pred_htmls_list, label_htmls_list):

    f1 = evaluate_f1(label_relations_list, pred_relations_list, num_workers=100)
    teds_metric = TEDSMetric(num_workers=100, structure_only=False)
    teds_info = teds_metric(pred_htmls_list, label_htmls_list)

    # calculate final metric base on macro
    f1_score = 0
    teds_score = 0
    metric = 0
    for idx in range(len(teds_info)):
        f1_score += f1[idx][-1]
        teds_score += teds_info[idx]

        metric += 0.5 * f1[idx][-1] + 0.5 * teds_info[idx]

    f1_score = f1_score / len(teds_info)
    teds_score = teds_score / len(teds_info)
    metric = metric / len(teds_info)

    eval_res = {}
    eval_res['f1'] = f1_score
    eval_res['teds'] = teds_score
    eval_res['metric'] = metric

    for key, value in eval_res.items():
        eval_res[key] = float('{:.5f}'.format(value))

    return eval_res

