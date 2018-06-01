# coding:utf8
"""
track.py
implement by lppl
a track description
"""
from utils import kalman_filter
import numpy as np
from box import Box


class BaseTrack(object):
    """
    a basic track   
    """
    def __init__(self, det, feature, frm_id, track_id, kalman_filter=None):
        """
        det has the format x1 y1 x2 y2
        :param det: 
        :param kalman_filter: 
        :param feature: deap feature
        """
        self.frm_id_list = []
        self.means = []
        self.covs = []
        self.features = []
        self.track_id = track_id
        self.age = 0 # track lost time count
        mean, cov = kalman_filter.initiate(Box.xyxy2xyah(det))
        self.frm_id_list.append(frm_id)
        self.means.append(mean)
        self.covs.append(cov)
        self.features.append(feature)
        self.color = np.random.randint(0, 255, (3, ))

    def update(self, det=None, feature=None, frm_id=-1, kalman_filter=None):
        """
        :param det: 
        :param feature: 
        :return: 
        """
        mean, cov = self.means[-1], self.covs[-1]
        pred_mean, pred_cov = kalman_filter.predict(mean, cov)
        if det is None:
            self.means.append(pred_mean)
            self.covs.append(pred_cov)
            self.age += 1
        else:
            self.age = 0
            update_mean, update_cov = kalman_filter.update(pred_mean, pred_cov, Box.xyxy2xyah(det))
            self.means.append(update_mean)
            self.covs.append(update_cov)
            self.frm_id_list.append(frm_id)
            self.features.append(feature)

    @property
    def curr_feature(self):
        return self.features[-1].copy()

    @property
    def mean(self):
        return self.means[-1].copy()

    @property
    def covariance(self):
        return self.covs[-1].copy()

    @property
    def is_lost(self):
        return self.age == 0

    @property
    def curr_pos(self):
        """
        return format x1 y1 x2 y2
        :return: 
        """
        mean = self.mean
        x, y, a, h = mean[0], mean[1], mean[2], mean[3]
        w = a * h
        return np.array([x, y, x + w, y + h])












