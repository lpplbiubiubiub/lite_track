import cv2
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.utils import linear_assignment_
from utils import kalman_filter
from box import Box


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)
    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))
    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    """
    Simple linear assignment
    :type cost_matrix: np.ndarray
    :type thresh: float
    :return: matches, unmatched_a, unmatched_b
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    cost_matrix[cost_matrix > thresh] = thresh + 1e-4
    indices = linear_assignment_.linear_assignment(cost_matrix)

    return _indices_to_matches(cost_matrix, indices, thresh)


def IOU(box1, box2):
    x1, y1, x2, y2 = box1[0], box1[1], box1[2], box1[3]
    a1, b1, a2, b2 = box2[0], box2[1], box2[2], box2[3]
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (a2 - a1) * (b2 - b1)
    x11 = max(x1, a1)
    y11 = max(y1, b1)
    x22 = min(x2, a2)
    y22 = min(y2, b2)
    intersect_w = x22 - x11
    intersect_h = y22 - y11
    if intersect_h <= 0 or intersect_w <= 0:
        return 0.
    intersect_area = intersect_w * intersect_h
    return intersect_area / (0. + area1 + area2 - intersect_area)


def iou_distance(tracks, detections):
    tracks_pos = [track.curr_pos for track in tracks]
    # compute iou is too slow
    nb_tracks = len(tracks)
    nb_dets = len(detections)
    dists = np.zeros(shape=(nb_tracks, nb_dets))
    for i in range(nb_tracks):
        for j in range(nb_dets):
            dists[i][j] = IOU(tracks_pos[i], detections[j])
    return 1. - dists


def nearest_reid_distance(tracks, detections, det_features, metric='cosine'):
    """
    Compute cost based on ReID features
    :type tracks: list[STrack]
    :type detections: list[BaseTrack]

    :rtype cost_matrix np.ndarray
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix

    for i, track in enumerate(tracks):
        cost_matrix[i, :] = np.maximum(0.0, cdist(track.features, det_features, metric).min(axis=0))

    return cost_matrix


def mean_reid_distance(tracks, det_feas, metric='cosine'):
    """
    Compute cost based on ReID features
    :type tracks: list[STrack]
    :type detections: list[BaseTrack]
    :type metric: str

    :rtype cost_matrix np.ndarray
    """
    cost_matrix = np.empty((len(tracks), len(det_feas)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix

    track_features = np.asarray([track.curr_feature for track in tracks], dtype=np.float32)
    det_features = np.asarray([det_fea for det_fea in det_feas], dtype=np.float32)
    cost_matrix = cdist(track_features, det_features, metric)

    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    """
    Gating some thresh
    :param kf: 
    :param cost_matrix: 
    :param tracks: 
    :param detections: det has format like x1y1x2y2
    :param only_position: 
    :return: 
    """
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([Box.xyxy2xyah(det) for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix
