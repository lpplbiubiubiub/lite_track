# coding:utf8
"""
box level operate
"""
import numpy as np


class Box(object):
    """
    box level operate 
    Box is atom of a tracklet
    """
    def __init__(self):
        pass

    @staticmethod
    def xyxy2xyah(det):
        """
        a = w / h
        :param det: 
        :return: 
        """
        x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
        w, h = x2 - x1, y2 - y1
        a = w / (h + 0.)
        return np.array([x1, y1, a, h])
