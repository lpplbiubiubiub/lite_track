# coding:utf8
"""
Online tracker
"""
import sys
import os
import numpy as np
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(main_path)
from utils import KalmanFilter
from track import BaseTrack
from configure import P
import matching
from vis import Visualizer


class OnlineTracker(object):
    """
    online tracker, has some match
    """
    def __init__(self):
        self.kf = KalmanFilter()
        self.min_ap_dist = 0.64
        self.tracks = []
        self.remove_tracks = []
        self.track_id = 0

        self.vis_tool = Visualizer("MOTDN")

    def match(self, img, detections, features, frm_id):
        # get track state
        is_lost_arr = np.array([track.is_lost for track in self.tracks])
        tracked_track_idx = np.where(is_lost_arr)[0]
        lost_track_idx = np.where(is_lost_arr==False)[0]
        tracked_stracks = map(lambda x: self.tracks[x], tracked_track_idx)
        lost_stracks = map(lambda x: self.tracks[x], lost_track_idx)
        print len(self.tracks)
        self.tracks = []
        # first match, we match active track with detection
        ## 1 匹配跟踪的轨迹
        dists = matching.nearest_reid_distance(tracked_stracks, detections, features, metric='euclidean')
        # dist不参与运算 只是起到gating的作用 滤除过长的轨迹
        dists = matching.gate_cost_matrix(self.kf, dists, tracked_stracks, detections)
        # match has format (track_id, det_id)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.min_ap_dist)
        for itracked, idet in matches:
            print "first match is ", tracked_stracks[itracked].track_id
            tracked_stracks[itracked].update(detections[idet], features[idet], frm_id, self.kf)
            self.tracks.append(tracked_stracks[itracked])
        detections = [detections[idet] for idet in u_detection]
        features = [features[idet] for idet in u_detection]
        ## 2 匹配消失的轨迹
        dists = matching.nearest_reid_distance(lost_stracks, detections, features, metric='euclidean')
        dists = matching.gate_cost_matrix(self.kf, dists, lost_stracks, detections)
        matches, u_lost, u_detection = matching.linear_assignment(dists, thresh=self.min_ap_dist)
        for itracked, idet in matches:
            print "second match is ", lost_stracks[itracked].track_id
            lost_stracks[itracked].update(detections[idet], features[idet], frm_id, self.kf)
            self.tracks.append(lost_stracks[itracked])
        ## 3 1中未匹配的轨迹继续匹配
        ## 匹配方式改为IOU匹配
        detections = [detections[i] for i in u_detection]
        features = [features[idet] for idet in u_detection]
        r_tracked_stracks = [tracked_stracks[idx] for idx in u_track]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            print "third match is ", r_tracked_stracks[itracked].track_id
            r_tracked_stracks[itracked].update(detections[idet], features[idet], frm_id, self.kf)
            self.tracks.append(r_tracked_stracks[itracked])
        # 未匹配到的更新
        for idx in u_track:
            r_tracked_stracks[idx].update(frm_id=frm_id, kalman_filter=self.kf)
            self.tracks.append(r_tracked_stracks[idx])
        detections = [detections[i] for i in u_detection]
        features = [features[idet] for idet in u_detection]
        ## 4 2中未匹配的轨迹继续匹配
        r_lost_stracks = [lost_stracks[idx] for idx in u_lost]
        dists = matching.iou_distance(r_lost_stracks, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            print "forth match is ", r_lost_stracks[itracked].track_id
            r_lost_stracks[itracked].update(detections[idet], features[idet], frm_id, self.kf)
            self.tracks.append(r_lost_stracks[itracked])
        # check u_unconfirmed and delete it if satisfy
        for idx in u_unconfirmed:
            r_lost_stracks[idx].update(frm_id=frm_id, kalman_filter=self.kf)
            if r_lost_stracks[idx].age >= P['max_lost_track_time']:
                continue
            self.tracks.append(r_lost_stracks[idx])
        detections = [detections[i] for i in u_detection]
        features = [features[idet] for idet in u_detection]
        # 生成新的轨迹
        for det, fea in zip(detections, features):
            self.tracks.append(BaseTrack(det, fea, frm_id, self.track_id, self.kf))
            self.track_id += 1

    def vis_track(self, img):
        # raw_input()
        show_thresh = 1
        track_pos_list = [track.curr_pos.astype(np.int32) for track in self.tracks if track.age < show_thresh]
        track_color_list = [track.color for track in self.tracks if track.age < show_thresh]
        track_id_list = [track.track_id for track in self.tracks if track.age < show_thresh]
        track_conf_list = [0.4 for track in self.tracks if track.age < show_thresh]
        self.vis_tool.image_track(img, track_id_list, track_pos_list, track_color_list, track_conf_list, "track")

if __name__ == "__main__":
    pass