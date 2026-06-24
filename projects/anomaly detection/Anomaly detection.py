"""
Optical Flow 기반 실시간 이상 행동 탐지 시스템
================================================
컴퓨터비전 수업 기말 프로젝트

알고리즘 구성:
  1. Lucas-Kanade Sparse Optical Flow  → 특징점(코너) 추적
  2. Farneback Dense Optical Flow      → 전체 움직임 벡터장 시각화
  3. MOG2 배경 차분                    → 움직임 마스크 추출
  4. 움직임 강도 임계값 기반 이상 탐지  → 경보 출력

이상 판정 지표 (MOTION_METRIC):
  - UCSD Ped2 정량 평가 결과, 전체 평균(mean)은 군중 장면에서 국소적 이상이
    희석되는 한계가 있었고, 상위 5% 픽셀(p95)을 사용했을 때 정확도가
    31.6% -> 44.6%로 개선되었습니다. 실시간 코드도 동일한 방식을 사용합니다.

실행 방법:
  python anomaly_detection.py            # 웹캠(0번) 사용
  python anomaly_detection.py --video <파일경로>   # 동영상 파일 사용
  python anomaly_detection.py --mode dense        # Dense Optical Flow 시각화
"""

import cv2
import numpy as np
import argparse
import time
from collections import deque


# ─── 1. 하이퍼파라미터 설정 ─────────────────────────────────────────────────

# Lucas-Kanade Optical Flow 파라미터
LK_PARAMS = dict(
    winSize=(21, 21),       # 탐색 윈도우 크기 (클수록 큰 움직임 추적 가능)
    maxLevel=3,             # 피라미드 레벨 (해상도 단계)
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# Shi-Tomasi 코너 검출 파라미터 (추적할 특징점 검출)
FEATURE_PARAMS = dict(
    maxCorners=200,         # 최대 특징점 수
    qualityLevel=0.01,      # 품질 임계값 (낮을수록 더 많은 점 검출)
    minDistance=10,         # 특징점 간 최소 거리 (픽셀)
    blockSize=7             # 코너 검출 블록 크기
)

# 이상 행동 탐지 파라미터
# MOTION_METRIC: "mean"(전체 평균) 또는 "p95"(상위 5% — UCSD 평가에서 더 우수했음)
MOTION_METRIC = "p95"
ANOMALY_THRESHOLD = 2.3     # Test004(2x 확대) + Dense P95 기준: 1.1335 x SCALE(2) ≈ 2.3
                            # Sparse 모드 또는 다른 영상 사용 시 +/- 키로 현장 재조정 권장
ALERT_DURATION = 60         # 경보 표시 프레임 수
REFRESH_INTERVAL = 5        # 특징점 갱신 주기 (프레임)
HISTORY_SIZE = 30           # 움직임 강도 히스토리 크기 (그래프용)


def compute_motion_score(values):
    """
    움직임 벡터 크기(magnitude) 배열로부터 '이상 판정용 점수'를 계산합니다.

    - "mean": 전체 평균 (값이 많은 영역에 의해 희석될 수 있음)
    - "p95" : 상위 5% 값 → 화면 일부에서 발생하는 국소적·급격한 움직임을
              더 잘 포착함 (UCSD Ped2 평가에서 채택)
    """
    if values is None or len(values) == 0:
        return 0.0
    if MOTION_METRIC == "mean":
        return float(np.mean(values))
    elif MOTION_METRIC == "p95":
        return float(np.percentile(values, 95))
    else:
        raise ValueError(f"알 수 없는 MOTION_METRIC: {MOTION_METRIC}")


# ─── 2. 유틸리티 함수 ────────────────────────────────────────────────────────

def draw_motion_graph(frame, history: deque, threshold: float):
    """
    우측 상단에 움직임 강도 그래프를 그립니다.
    history: 최근 N프레임의 평균 움직임 강도 값
    """
    h, w = frame.shape[:2]
    gx, gy, gw, gh = w - 220, 10, 200, 80  # 그래프 영역

    # 반투명 배경
    overlay = frame.copy()
    cv2.rectangle(overlay, (gx - 5, gy - 5), (gx + gw + 5, gy + gh + 5),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # 임계선
    threshold_y = int(gy + gh - (threshold / (threshold * 2)) * gh)
    cv2.line(frame, (gx, threshold_y), (gx + gw, threshold_y), (0, 100, 255), 1)

    # 그래프 선 그리기
    if len(history) > 1:
        max_val = max(max(history), threshold * 1.5) + 0.1
        pts = []
        for i, val in enumerate(history):
            px = gx + int(i * gw / (HISTORY_SIZE - 1))
            py = gy + gh - int((val / max_val) * gh)
            pts.append((px, py))
        for i in range(len(pts) - 1):
            color = (0, 255, 100) if history[i] < threshold else (0, 80, 255)
            cv2.line(frame, pts[i], pts[i + 1], color, 2)

    cv2.putText(frame, "Motion Graph", (gx, gy + gh + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def draw_status_panel(frame, motion_score: float, alert_count: int, fps: float):
    """좌측 상단 상태 패널 표시"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (260, 110), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    is_anomaly = motion_score > ANOMALY_THRESHOLD
    status_color = (0, 60, 220) if is_anomaly else (0, 200, 80)
    status_text = "!! ANOMALY DETECTED !!" if is_anomaly else "NORMAL"

    cv2.putText(frame, status_text, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
    cv2.putText(frame, f"Motion({MOTION_METRIC}): {motion_score:.2f}  (thresh {ANOMALY_THRESHOLD})",
                (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
    cv2.putText(frame, f"Alerts  : {alert_count}",
                (12, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
    cv2.putText(frame, f"FPS     : {fps:.1f}",
                (12, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)


def visualize_dense_flow(flow):
    """
    Dense Optical Flow를 HSV 컬러맵으로 시각화합니다.
    - 색상(Hue)    → 움직임 방향
    - 밝기(Value)  → 움직임 강도
    """
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255  # 채도 최대

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2          # 방향 → 색상
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # 강도 → 밝기

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ─── 3. 핵심 클래스 ──────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Optical Flow 기반 이상 행동 탐지기

    동작 흐름:
      매 프레임마다 이전 프레임과의 Optical Flow를 계산하고,
      움직임 벡터의 평균 크기(magnitude)가 임계값을 초과하면 이상으로 판단합니다.
    """

    def __init__(self, mode: str = "sparse"):
        self.mode = mode                     # "sparse" or "dense"
        self.prev_gray = None                # 이전 프레임 (그레이스케일)
        self.prev_pts = None                 # 이전 특징점 (Sparse 전용)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=40, detectShadows=False
        )
        self.alert_timer = 0                 # 경보 표시 타이머
        self.alert_count = 0                 # 총 이상 감지 횟수
        self.frame_count = 0
        self.motion_history = deque(maxlen=HISTORY_SIZE)
        self.prev_time = time.time()
        self.fps = 0.0

        # 특징점 궤적 시각화용 마스크
        self.track_mask = None

    def _update_fps(self):
        now = time.time()
        self.fps = 1.0 / max(now - self.prev_time, 1e-9)
        self.prev_time = now

    def _get_features(self, gray, fg_mask):
        """배경 마스크를 활용해 움직이는 영역에서만 특징점 검출"""
        # 침식(erosion)으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_clean = cv2.erode(fg_mask, kernel, iterations=1)

        pts = cv2.goodFeaturesToTrack(gray, mask=fg_clean, **FEATURE_PARAMS)
        return pts

    def process_sparse(self, frame):
        """
        Lucas-Kanade Sparse Optical Flow 처리
        - 특정 특징점(코너)만 추적
        - 각 점의 이동 벡터를 화살표로 시각화
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = self.bg_subtractor.apply(frame)
        output = frame.copy()

        if self.track_mask is None or self.track_mask.shape[:2] != frame.shape[:2]:
            self.track_mask = np.zeros_like(frame)

        motion_score = 0.0

        if self.prev_gray is not None:
            # ① 특징점 주기적 갱신
            if self.prev_pts is None or self.frame_count % REFRESH_INTERVAL == 0:
                self.prev_pts = self._get_features(self.prev_gray, fg_mask)
                self.track_mask = np.zeros_like(frame)

            if self.prev_pts is not None and len(self.prev_pts) > 0:
                # ② Lucas-Kanade Optical Flow 계산
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.prev_pts, None, **LK_PARAMS
                )

                # ③ 추적 성공한 점만 선별
                good_new = next_pts[status == 1]
                good_old = self.prev_pts[status == 1]

                # ④ 움직임 벡터 크기 계산 → 이상 탐지
                #    (mean: 전체 평균 / p95: 상위 5% — UCSD 평가에서 더 우수)
                if len(good_new) > 0:
                    diff = good_new - good_old
                    magnitudes = np.linalg.norm(diff, axis=1)  # 각 점의 이동 거리
                    motion_score = compute_motion_score(magnitudes)

                    # ⑤ 궤적 및 화살표 시각화
                    for new, old in zip(good_new, good_old):
                        a, b = int(new[0]), int(new[1])
                        c, d = int(old[0]), int(old[1])
                        mag = np.linalg.norm([a - c, b - d])
                        color = (0, 60, 255) if mag > ANOMALY_THRESHOLD * 2 else (0, 230, 120)
                        self.track_mask = cv2.arrowedLine(
                            self.track_mask, (c, d), (a, b), color, 2, tipLength=0.4
                        )
                        cv2.circle(output, (a, b), 3, color, -1)

                self.prev_pts = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None

        # ⑥ 궤적 오버레이 (점점 흐려지는 효과)
        self.track_mask = (self.track_mask * 0.92).astype(np.uint8)
        output = cv2.add(output, self.track_mask)

        # ⑦ 배경 마스크 시각화 (우측 하단 미니맵)
        h, w = frame.shape[:2]
        mini_h, mini_w = h // 5, w // 5
        fg_color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        mini = cv2.resize(fg_color, (mini_w, mini_h))
        output[h - mini_h - 5: h - 5, w - mini_w - 5: w - 5] = mini

        self.prev_gray = gray.copy()
        return output, motion_score

    def process_dense(self, frame):
        """
        Farneback Dense Optical Flow 처리
        - 모든 픽셀의 움직임 벡터 계산
        - HSV 컬러맵으로 방향/강도 시각화
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output = frame.copy()
        motion_score = 0.0

        if self.prev_gray is not None:
            # ① Dense Optical Flow 계산
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                pyr_scale=0.5,   # 피라미드 스케일
                levels=3,        # 피라미드 레벨
                winsize=15,      # 평균화 윈도우
                iterations=3,    # 반복 횟수
                poly_n=5,        # 다항식 이웃 크기
                poly_sigma=1.2,  # 가우시안 표준편차
                flags=0
            )

            # ② 움직임 강도 계산 (mean: 전체 평균 / p95: 상위 5% — UCSD 평가에서 더 우수)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_score = compute_motion_score(mag.flatten())

            # ③ Flow 시각화 (반투명 오버레이)
            flow_vis = visualize_dense_flow(flow)
            cv2.addWeighted(output, 0.55, flow_vis, 0.45, 0, output)

            # ④ 이상 영역 강조 (빨간 오버레이)
            if motion_score > ANOMALY_THRESHOLD:
                anomaly_mask = (mag > ANOMALY_THRESHOLD * 2).astype(np.uint8) * 255
                anomaly_mask = cv2.dilate(anomaly_mask, None, iterations=3)
                red_overlay = np.zeros_like(frame)
                red_overlay[:, :, 2] = anomaly_mask
                cv2.addWeighted(output, 0.8, red_overlay, 0.2, 0, output)

        self.prev_gray = gray.copy()
        return output, motion_score

    def run(self, source):
        global ANOMALY_THRESHOLD  # 함수 내에서 +/- 키로 값을 변경하므로 최상단에 선언

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] 영상 소스를 열 수 없습니다: {source}")
            return

        print(f"[INFO] 모드: {self.mode.upper()} Optical Flow")
        print(f"[INFO] 이상 판정 지표: {MOTION_METRIC} (UCSD 평가 결과 채택)")
        print(f"[INFO] 이상 탐지 임계값: {ANOMALY_THRESHOLD}")
        print(f"[INFO] 종료: q 키, 임계값 ↑: +, 임계값 ↓: -")

        while True:
            ret, frame = cap.read()
            if not ret:
                # 동영상 파일이면 처음부터 반복
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.prev_gray = None
                self.prev_pts = None
                continue

            self._update_fps()
            self.frame_count += 1

            # ── 프레임 처리 ──
            if self.mode == "sparse":
                output, motion_score = self.process_sparse(frame)
            else:
                output, motion_score = self.process_dense(frame)

            self.motion_history.append(motion_score)

            # ── 이상 탐지 판정 ──
            if motion_score > ANOMALY_THRESHOLD:
                if self.alert_timer == 0:
                    self.alert_count += 1
                self.alert_timer = ALERT_DURATION

            # ── UI 오버레이 ──
            draw_status_panel(output, motion_score, self.alert_count, self.fps)
            draw_motion_graph(output, self.motion_history, ANOMALY_THRESHOLD)

            # ── 경보 테두리 ──
            if self.alert_timer > 0:
                alpha = min(1.0, self.alert_timer / 20)
                h, w = output.shape[:2]
                thickness = max(4, int(alpha * 12))
                cv2.rectangle(output, (0, 0), (w - 1, h - 1), (0, 0, 220), thickness)
                self.alert_timer -= 1

            # ── 모드 표시 ──
            mode_label = "Sparse L-K" if self.mode == "sparse" else "Dense Farneback"
            cv2.putText(output, f"[{mode_label}]", (10, output.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

            cv2.imshow("Anomaly Detection - Optical Flow", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                ANOMALY_THRESHOLD += 0.5
                print(f"[INFO] 임계값 → {ANOMALY_THRESHOLD:.1f}")
            elif key == ord('-'):
                ANOMALY_THRESHOLD = max(0.5, ANOMALY_THRESHOLD - 0.5)
                print(f"[INFO] 임계값 → {ANOMALY_THRESHOLD:.1f}")

        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[결과] 총 이상 감지 횟수: {self.alert_count}회")


# ─── 4. 메인 실행 ────────────────────────────────────────────────────────────

# 기본 입력 동영상 경로 (PyCharm에서 인자 없이 바로 실행할 때 사용됩니다)
# - 비워두면("") 웹캠(0번)을 사용합니다.
# - 동영상을 쓰려면 아래에 실제 경로를 입력하세요.
#   예) VIDEO_PATH = r"C:\Users\test_video.mp4"
VIDEO_PATH = r"C:\Users\IBYEO\PycharmProjects\grad_school\course_cv\src\project\final\demo_test004.mp4"


def main():
    parser = argparse.ArgumentParser(
        description="Optical Flow 기반 실시간 이상 행동 탐지 시스템"
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="입력 동영상 파일 경로 (지정하지 않으면 위 VIDEO_PATH 또는 웹캠 사용)"
    )
    parser.add_argument(
        "--mode", type=str, default="sparse",
        choices=["sparse", "dense"],
        help="Optical Flow 모드 선택 (기본값: sparse)"
    )
    args = parser.parse_args()

    # 우선순위: --video 인자 > VIDEO_PATH 설정 > 웹캠(0)
    if args.video:
        source = args.video
    elif VIDEO_PATH:
        source = VIDEO_PATH
    else:
        source = 0

    print(f"[INFO] 입력 소스: {'웹캠' if source == 0 else source}")

    detector = AnomalyDetector(mode=args.mode)
    detector.run(source)


if __name__ == "__main__":
    main()