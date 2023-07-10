from mmdet.datasets.builder import DATASETS
from mmocr.core.evaluation.ocr_metric import eval_ocr_metric
from mmocr.core.evaluation.teds_metric import eval_teds_metric, cal_pred
from mmocr.datasets.base_dataset import BaseDataset

from tqdm import tqdm


@DATASETS.register_module()
class OCRDataset(BaseDataset):

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['img_info']['ann_file'] = self.ann_file
        results['text'] = results['img_info']['text']

    def evaluate(self, results, metric='acc', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str: float]
        """
        gt_texts = []
        pred_texts = []
        for i in range(len(self)):
            item_info = self.data_infos[i]
            text = item_info['text']
            gt_texts.append(text)
            pred_texts.append(results[i]['text'])
            # print('======')
            # print(self.data_infos[i])  ## {filename, text, bbox, bbox_masks}
            # print(results[i]) ## {text, score, bbox}
            # print('======')
        eval_results = eval_ocr_metric(pred_texts, gt_texts)

        return eval_results
    
@DATASETS.register_module()
class TEDSDataset(BaseDataset):

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['img_info']['ann_file'] = self.ann_file
        results['text'] = results['img_info']['text']

    def evaluate(self, results, metric='acc', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str: float]
        """
        pred_relations_list = []
        pred_htmls_list = []
        label_relations_list = []
        label_htmls_list = []

        print("Cal pred data to htmls...")
        for i in tqdm(range(len(self))):
            label_info = self.data_infos[i]
            pred_info = results[i]
            
            pred_relations, pred_htmls = cal_pred(pred_info, label_info)
            pred_relations_list.append(pred_relations)
            pred_htmls_list.append(pred_htmls)

            label_relations_list.append(label_info['label_relations'])
            label_htmls_list.append(label_info['label_htmls'])

        eval_results = eval_teds_metric(pred_relations_list, label_relations_list, pred_htmls_list, label_htmls_list)

        return eval_results