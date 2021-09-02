# 얼굴 검출하고 r,b,g 신호 추출하는 함수

import cv2
import numpy as np

from uti import skinsegment

def get_rgb(frame, is_tracking, tracker, detector, w, h, prev_bbox, r_buffer, g_buffer, b_buffer):
    track_toler = 1
    detect_th = 0.5
    curr_bbox = []
    # 검출된 얼굴이 존재하지 않는 경우 검출 수행
    # print(is_tracking)
    if not is_tracking:
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (250, 250)), 1.0, (250, 250), [104., 117., 123.],
                                     False,
                                     True)
        detector.setInput(blob)
        detections = detector.forward()

        bboxes = [detections[0, 0, i, 3:7] for i in range(detections.shape[2]) if
                  detections[0, 0, i, 2] >= detect_th]
        if len(bboxes) > 0:
            # 가장 큰 얼굴 하나만 검출
            bboxes = sorted(bboxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
            bboxes = [(rect * np.array([w, h, w, h])).astype('int') for rect in bboxes]
            curr_bbox = [bboxes[0][0], bboxes[0][1], bboxes[0][2] - bboxes[0][0],
                         bboxes[0][3] - bboxes[0][1]]  # (xs,ys,xe,ye) -> (x,y,w,h)

            # 트래커에 현재 얼굴 위치 등록
            tracker.init(frame, curr_bbox)
            is_tracking = True

    # 검출된 얼굴이 존재하는 경우 위치 업데이트
    elif is_tracking:
        is_tracking, curr_bbox = tracker.update(frame)
        curr_bbox = [curr if abs(curr - prev) > track_toler else prev for curr, prev in
                     zip(curr_bbox, prev_bbox)]
        prev_bbox = curr_bbox

    try:
        face = frame[curr_bbox[1]:curr_bbox[1] + curr_bbox[3], curr_bbox[0]:curr_bbox[0] + curr_bbox[2]]
        mask = skinsegment.create_skin_mask(face)
        n_pixel = max(1, np.sum(mask))

        b, g, r = cv2.split(face)
        r[mask == 0] = 0
        g[mask == 0] = 0
        b[mask == 0] = 0

        r = r.astype(np.float32)
        g = g.astype(np.float32)
        b = b.astype(np.float32)

        r_mean = np.sum(r) / n_pixel
        g_mean = np.sum(g) / n_pixel
        b_mean = np.sum(b) / n_pixel

        r_buffer.append(r_mean)
        g_buffer.append(g_mean)
        b_buffer.append(b_mean)

        cv2.rectangle(frame, (curr_bbox[0], curr_bbox[1]),
                      (curr_bbox[0] + curr_bbox[2], curr_bbox[1] + curr_bbox[3]), (0, 0, 255), 5)

    except:
        is_tracking = False
        r_buffer.append(0.0)
        g_buffer.append(0.0)
        b_buffer.append(0.0)

    return is_tracking