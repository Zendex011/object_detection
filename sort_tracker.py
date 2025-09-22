# sort_tracker.py - Pure Python SORT (no OpenCV compilation required)
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) +
              (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh + 1e-6)
    return o

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        self.kf.R *= 0.01
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

    def update(self, bbox):
        self.time_since_update = 0
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / (h + 1e-6)
        return np.array([x, y, s, r]).reshape((4,1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / (w + 1e-6)
        x1 = x[0] - w / 2.
        y1 = x[1] - h / 2.
        x2 = x[0] + w / 2.
        y2 = x[1] + h / 2.
        return np.array([x1, y1, x2, y2])

class Sort:
    def __init__(self, max_age=5, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, dets=np.empty((0, 5))):
        if len(self.trackers) == 0:
            self.trackers = [KalmanBoxTracker(det[:4]) for det in dets]
            return np.array([np.append(det[:4], i) for i, det in enumerate(dets)])

        # Predict existing tracks
        predicted_boxes = np.array([t.predict() for t in self.trackers])
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, predicted_boxes)

        for t, m in zip(self.trackers, matched):
            t.update(dets[m, :4])

        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :4]))

        results = []
        for t in self.trackers:
            if t.time_since_update < self.max_age:
                results.append(np.append(t.kf.x[:4].reshape(-1), t.id))

        return np.array(results)

    def associate_detections_to_trackers(self, dets, trks):
        if len(trks) == 0:
            return [], np.arange(len(dets)), []

        iou_matrix = np.zeros((len(dets), len(trks)), dtype=np.float32)
        for d, det in enumerate(dets):
            for t, trk in enumerate(trks):
                iou_matrix[d, t] = iou(det[:4], trk)

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matches = []
        unmatched_dets = []
        unmatched_trks = list(range(len(trks)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] < self.iou_threshold:
                unmatched_dets.append(r)
            else:
                matches.append(r)
                unmatched_trks.remove(c)

        unmatched_dets += [i for i in range(len(dets)) if i not in matches]
        return matches, unmatched_dets, unmatched_trks
