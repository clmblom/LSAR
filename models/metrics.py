import editdistance
import utils.polyfuncs as pf
import shapely.ops as so
import utils.util


class CER:
    def __init__(self):
        self.edits = 0
        self.length = 0

    def update(self, targets, predictions):
        for target, prediction in zip(targets, predictions):
            self.edits += editdistance.eval(target, prediction)
            self.length += len(target)

    def reset(self):
        self.edits = 0
        self.length = 0

    def calc_cer(self):
        if self.length == 0:
            print("Saving a crash, length=1")
            length = 1
        else:
            length = self.length
        return (self.edits/length)*100


class LosenMetric:
    def __init__(self):
        self.text_cer = CER()
        self.class_wer = CER()

    def update(self, targets, predictions):
        tar_words = []
        tar_classes = []
        pred_words = []
        pred_classes = []
        for target, prediction in zip(targets, predictions):
            text, classes = utils.util.separate_string(target)
            tar_words.append(text)
            tar_classes.append(classes)
            text, classes = utils.util.separate_string(prediction)
            pred_words.append(text)
            pred_classes.append(classes)
        self.text_cer.update(tar_words, pred_words)
        self.class_wer.update(tar_classes, pred_classes)

    def calc_metric(self):
        return self.text_cer.calc_cer(), self.class_wer.calc_cer()

    def reset(self):
        self.text_cer.reset()
        self.class_wer.reset()


class SegmentationMetric:
    def __init__(self):
        self.m1s = 0 # p covers exactly one gt, under-segmentation
        self.m1s_len = 0
        self.m2s = 0 # p covers any gt
        self.m2s_len = 0
        self.m3s = 0  # gt is covered by 1/p, over-segmentation
        self.m3s_len = 0
        self.m4s = 0  # gt is covered by any p
        self.m4s_len = 0
        self.ious = 0
        self.ious_len = 0


    def update(self, targets, predictions):
        # preds is a list of polygon points [[(x_0, y_0), ..., (x_n, y_n)], ...]
        # gts is a list of polygon points [[(x_0, y_0), ..., (x_m, y_m)], ...]
        gt_polys = pf.contours_to_polygons(targets,
                                           make_square=False,
                                           rotated=False,
                                           shrink_factor=1,
                                           force_shrink_factor=True)
        pred_polys = pf.contours_to_polygons(predictions,
                                             make_square=False,
                                             rotated=False,
                                             shrink_factor=1,
                                             force_shrink_factor=True)
        metric_dict = self._calc_metric(gt_polys, pred_polys)

        self.m1s += sum(metric_dict['m1'])
        self.m1s_len += len(metric_dict['m1'])
        self.m2s += sum(metric_dict['m2'])
        self.m2s_len += len(metric_dict['m2'])
        self.m3s += sum(metric_dict['m3'])
        self.m3s_len += len(metric_dict['m3'])
        self.m4s += sum(metric_dict['m4'])
        self.m4s_len += len(metric_dict['m4'])
        self.ious += sum(metric_dict['iou'])
        self.ious_len += len(metric_dict['iou'])

    def reset(self):
        self.m1s = 0
        self.m1s_len = 0
        self.m2s = 0
        self.m2s_len = 0
        self.m3s = 0
        self.m3s_len = 0
        self.m4s = 0
        self.m4s_len = 0
        self.ious = 0
        self.ious_len = 0

    def calc_metric(self):
        m1 = self.m1s / (self.m1s_len if self.m1s_len else 1)
        m2 = self.m2s / (self.m2s_len if self.m2s_len else 1)
        m3 = self.m3s / (self.m3s_len if self.m3s_len else 1)
        m4 = self.m4s / (self.m4s_len if self.m4s_len else 1)
        iou = self.ious / (self.ious_len if self.ious_len else 1)
        m = m1*m2*m3*m4
        return {'m1': m1, 'm2': m2, 'm3': m3, 'm4': m4, 'iou': iou, 'm': m}

    def _calc_metric(self, targets, predictions):
        m1 = []
        m2 = []
        m3 = []
        m4 = []
        iou = []

        for target in targets:
            overlapping_polygons = pf.get_overlapping_polygons(target,
                                                               predictions)  # The pred polygons that overlaps the gt polygon
            if overlapping_polygons:
                m3.append(1 / len(overlapping_polygons))
                m4.append(1)

                prediction_union = so.cascaded_union(overlapping_polygons)
                iou.append(target.intersection(prediction_union).area / target.union(prediction_union).area)
            else:
                m4.append(0)
        for prediction in predictions:
            overlapping_polygons = pf.get_overlapping_polygons(prediction,
                                                               targets) # The gt polygons that overlaps the pred polygon
            if overlapping_polygons:
                m1.append(1 if len(overlapping_polygons) == 1 else 0)
                m2.append(1)
            else:
                m2.append(0)

        return {'m1': m1, 'm2': m2, 'm3': m3, 'm4': m4, 'iou': iou}



