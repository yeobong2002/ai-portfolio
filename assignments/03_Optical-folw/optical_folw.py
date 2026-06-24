import cv2
import numpy as np

video          = cv2.VideoCapture('../../data/traffic.mp4')
prev_pts       = None
prev_gray_frame = None
tracks         = None

while True:
    retval, frame = video.read()
    if not retval:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_pts is not None:
        # ── Optical Flow 계산 (Lucas-Kanade) ─────────
        pts, status, errors = cv2.calcOpticalFlowPyrLK(
            prev_gray_frame,
            gray_frame,
            prev_pts,
            None,
            winSize  = (15, 15),  # 탐색 윈도우 크기
            maxLevel = 5,         # 피라미드 레벨 수 (Coarse-to-Fine)
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,   # 최대 반복 횟수
                0.03  # 수렴 기준 (epsilon)
            )
        )

        # status == 1 인 포인트만 추적 성공
        good_pts = pts[status == 1]

        # 추적 궤적 누적
        if tracks is None:
            tracks = good_pts
        else:
            tracks = np.vstack((tracks, good_pts))

        # 궤적 시각화 (초록 점)
        for p in tracks:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)

    else:
        # ── 첫 프레임: 추적할 특징점 검출 ──────────
        pts = cv2.goodFeaturesToTrack(
            gray_frame,
            500,   # 최대 특징점 수
            0.05,  # 품질 기준 (낮을수록 많이 검출)
            10     # 특징점 간 최소 거리
        )
        pts = pts.reshape(-1, 1, 2)  # calcOpticalFlowPyrLK 입력 형식

    # 이전 프레임 업데이트
    prev_pts        = pts
    prev_gray_frame = gray_frame

    cv2.imshow('frame', frame)
    key = cv2.waitKey() & 0xff
    if key == 27:          # ESC: 종료
        break
    if key == ord('c'):    # c: 추적 초기화
        tracks   = None
        prev_pts = None

cv2.destroyAllWindows()