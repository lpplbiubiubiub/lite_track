# coding:utf8
from tracker import OnlineTracker
from configure import P
import os
import numpy as np
import cv2


class MOTDT(object):
    """

    """

    def __init__(self, image_path, det_file, feature_file, dst_file, vis):
        assert os.path.isdir(image_path), "{} doesn't exist".format(image_path)
        assert os.path.isfile(det_file), "{} doesn't exist".format(det_file)
        assert os.path.isfile(feature_file), "{} doesn't exist".format(feature_file)
        image_list = sorted(os.listdir(image_path), key=lambda x: int(x.split(".")[0]))
        assert len(image_list) > 0, "{} containing no image".format(image_path)
        self.image_list = map(lambda x: os.path.join(image_path, x), image_list)
        cv_image = cv2.imread(os.path.join(image_path, image_list[0]))
        self.tracker = OnlineTracker()
        P['image_width'] = cv_image.shape[1]
        P['image_height'] = cv_image.shape[0]
        self._detect = np.loadtxt(det_file, delimiter=",")
        self._det_feature = np.loadtxt(feature_file, delimiter=",")
        self._detect, self._det_feature = MOTDT.filter_detect(self._detect, self._det_feature)
        idx_list = self._detect[:, 0].astype(np.int32)
        self._idx_list = np.arange(np.min(idx_list), np.max(idx_list))
        self._dst_file = dst_file
        self._dst_pickle = os.path.basename(dst_file).replace(".txt", ".pkl")

    def run(self):
        for idx, i in enumerate(self._idx_list):
            print("processing {} / {}".format(i, len(self._idx_list)))
            tem_idx = self._detect[:, 0] == i
            tem_detection = self._detect[tem_idx][:, 2:6]
            tem_detection[:, 2] += tem_detection[:, 0]
            tem_detection[:, 3] += tem_detection[:, 1]
            tem_det_feature = self._det_feature[tem_idx][:]
            cv_img = cv2.imread(self.image_list[idx])
            self.tracker.match(cv_img, tem_detection, tem_det_feature, i)
            self.tracker.vis_track(cv_img)


    @staticmethod
    def filter_detect(detect, detect_feature):
        """
        1,-1,1359.1,413.27,120.26,362.77,2.3092,-1,-1,-1
        :param detect: np.array
        :return: 
        """
        valid_detect_idx = (detect[:, 2] >= 0) * (detect[:, 2] < P['image_width']) * (
        detect[:, 2] + detect[:, 4] >= 0) * \
                           (detect[:, 2] + detect[:, 4] < P['image_width']) * (detect[:, 6] >= P['conf_thresh']) * \
                           (detect[:, 3] >= 0) * (detect[:, 3] < P['image_height']) * (
                           detect[:, 3] + detect[:, 5] >= 0) * \
                           (detect[:, 3] + detect[:, 5] < P['image_height'])
        valid_det = detect[valid_detect_idx]
        valid_detect_feature = detect_feature[valid_detect_idx]
        return valid_det, valid_detect_feature


if __name__ == "__main__":
    # 02 04 05 09 10 11 13
    # 01 03 06 07 08 12 14
    seq_idx_list = ["10"]
    poi = None
    for seq_idx in seq_idx_list:
        poi = MOTDT(image_path="/home/xksj/Data/lp/MOT16/train/MOT16-{}/img1".format(seq_idx),
                  det_file="/home/xksj/Data/lp/MOT16/high_level_det/MOT16-{}_det.txt".format(seq_idx),
                  feature_file="/home/xksj/Data/lp/MOT16/high_level_det/MOT16-{}_feat.txt".format(seq_idx),
                  dst_file="../data/MOT16-{}.txt".format(seq_idx),
                  vis=True)
        poi.run()
