"""
UCSD Ped2 - Optical Flow 기반 이상 행동 탐지 평가 스크립트
============================================================

확인된 데이터 구조:
  UCSDped2/
    Train/Train001 ~ Train016/   (각 폴더 안에 001.tif ~ NNN.tif)
    Test/Test001 ~ Test012/      (각 폴더 안에 001.tif ~ NNN.tif)
    Test/Test001_gt ~ Test012_gt/ (픽셀 단위 GT, 이번 평가에서는 미사용)
    Test/UCSDped2.m              (gt_frame = [시작:끝] 형식, 줄 순서 = Test001~012)

동작 흐름:
  1. Train 16개 영상 → 정상(NORMAL) 움직임의 평균/표준편차 계산 → 임계값(threshold) 산출
  2. Test 12개 영상 → Farneback Optical Flow magnitude 계산
  3. .m 파일의 정답(이상 프레임 구간)과 비교하여 프레임 단위 정확도 산출
  4. 영상별 결과 그래프(PNG) 저장

실행 전 설정:
  - BASE_DIR 을 실제 UCSDped2 폴더 경로로 수정하세요.
  - QUICK_TEST = True 로 설정하면 일부 영상만으로 빠르게 동작 확인 가능합니다.
"""

import re
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ── 한글 폰트 설정 (Windows: 맑은 고딕) ──
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지

# ─── 설정 ────────────────────────────────────────────────────────────────────

BASE_DIR = r"C:\Users\IBYEO\PycharmProjects\grad_school\course_cv\src\project\final\data\UCSDped2"
OUTPUT_DIR = "ucsd_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

QUICK_TEST = False   # True로 설정하면 Train 3개 / Test 2개만으로 빠르게 테스트

# Farneback Dense Optical Flow 파라미터 (수업 10주차 + 기존 구현과 동일)
FARNEBACK_PARAMS = dict(
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)


# ─── 1. 프레임 시퀀스 → Optical Flow magnitude 리스트 ──────────────────────────

# def compute_magnitudes(folder):
#     """
#     폴더 안의 .tif 프레임들을 순서대로 읽어 Farneback Optical Flow를 계산하고,
#     프레임별 평균 magnitude 리스트를 반환합니다.
#
#     반환 길이는 입력 프레임 수와 동일합니다.
#     (첫 프레임은 비교 대상이 없으므로 0.0으로 채워 Ground Truth와 인덱스를 맞춥니다.)
#     """
#     files = sorted(glob.glob(os.path.join(folder, "*.tif")))
#     mags = [0.0]
#     prev = None
#
#     for f in files:
#         gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
#         if prev is not None:
#             flow = cv2.calcOpticalFlowFarneback(prev, gray, None, **FARNEBACK_PARAMS)
#             mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#             mags.append(float(np.mean(mag)))
#         prev = gray
#
#     return mags
def compute_magnitudes(folder, metric="p95"):
    files = sorted(glob.glob(os.path.join(folder, "*.tif")))
    mags = [0.0]
    prev = None

    for f in files:
        gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, gray, None, **FARNEBACK_PARAMS)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            if metric == "mean":
                value = float(np.mean(mag))
            elif metric == "p95":
                # 상위 5% 픽셀의 움직임 강도 → 국소적 빠른 움직임 포착
                value = float(np.percentile(mag, 95))

            mags.append(value)
        prev = gray

    return mags


# ─── 2. .m 파일에서 모든 GT 구간 파싱 ───────────────────────────────────────────

def parse_all_ground_truths(m_file_path):
    """
    UCSDped2.m 에서 'gt_frame = [시작:끝]' 패턴을 순서대로 모두 추출합니다.
    줄 순서 = Test001, Test002, ... Test012 순서와 일치합니다.

    반환 예: [(61, 180), (95, 180), (1, 146), ...]
    """
    with open(m_file_path) as f:
        content = f.read()

    matches = re.findall(r"gt_frame\s*=\s*\[(\d+):(\d+)\]", content)
    return [(int(a), int(b)) for a, b in matches]


# ─── 3. Train 데이터로 정상 움직임 통계 → 임계값 산출 ────────────────────────────

def compute_threshold():
    """
    Train 영상들(정상 프레임만 존재)에서 magnitude의 평균/표준편차를 구하고,
    '평균 + 3*표준편차'를 이상 탐지 임계값으로 사용합니다.

    (통계학적으로 정규분포 가정 시 약 99.7% 범위를 벗어나는 값을 이상치로 간주)
    """
    train_folders = sorted(glob.glob(os.path.join(BASE_DIR, "Train", "Train*")))
    if QUICK_TEST:
        train_folders = train_folders[:3]

    print(f"[Train] {len(train_folders)}개 영상에서 정상 움직임 통계 수집 중...")

    all_normal_mags = []
    for folder in train_folders:
        mags = compute_magnitudes(folder)
        all_normal_mags += mags[1:]   # 0.0으로 채운 첫 프레임은 통계에서 제외
        print(f"  - {os.path.basename(folder)}: {len(mags)} 프레임 처리 완료")

    mean_normal = np.mean(all_normal_mags)
    std_normal = np.std(all_normal_mags)
    threshold = mean_normal + 3 * std_normal

    print(f"\n  정상 평균(mean)      : {mean_normal:.4f}")
    print(f"  정상 표준편차(std)   : {std_normal:.4f}")
    print(f"  산출된 임계값 (μ+3σ) : {threshold:.4f}\n")

    return threshold


# ─── 4. Test 데이터 평가 + 그래프 저장 ──────────────────────────────────────────

def evaluate_test(video_num, threshold, gt_ranges):
    """
    Test{video_num:03d} 영상에 대해:
      - magnitude 계산
      - GT(이상 프레임 구간)와 비교하여 프레임 단위 정확도 계산
      - 결과 그래프(PNG) 저장
    """
    folder = os.path.join(BASE_DIR, "Test", f"Test{video_num:03d}")
    mags = compute_magnitudes(folder)
    n = len(mags)

    # Ground Truth 배열 생성 (1-indexed 구간 -> 0-indexed 배열)
    start, end = gt_ranges[video_num - 1]
    gt = np.zeros(n)
    gt[start - 1: min(end, n)] = 1

    # 예측 (임계값 초과 -> 이상)
    pred = (np.array(mags) > threshold).astype(int)

    # 프레임 단위 정확도
    acc = np.mean(gt == pred)

    # ── 시각화 ──
    plt.figure(figsize=(10, 4))
    plt.plot(mags, color="steelblue", linewidth=1.2, label="Motion Magnitude")
    plt.axhline(threshold, color="orange", linestyle="--", label="Threshold (Train 기반)")
    plt.fill_between(
        range(n), 0, max(mags) * 1.05 if max(mags) > 0 else 1,
        where=gt == 1, color="red", alpha=0.15,
        label="Ground Truth (실제 이상 구간)"
    )
    plt.title(f"UCSD Ped2 - Test{video_num:03d}  (프레임 정확도 {acc*100:.1f}%)")
    plt.xlabel("Frame")
    plt.ylabel("Mean Optical Flow Magnitude")
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"test{video_num:03d}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

    return acc


# ─── 메인 실행 ──────────────────────────────────────────────────────────────

def main():
    # 1. Train 데이터 기반 임계값 산출
    threshold = compute_threshold()

    # 2. Ground Truth 로드
    m_file = os.path.join(BASE_DIR, "Test", "UCSDped2.m")
    gt_ranges = parse_all_ground_truths(m_file)
    print(f"[Test] GT 구간 {len(gt_ranges)}개 로드 완료\n")

    # 3. Test 영상 평가
    test_indices = range(1, 13)
    if QUICK_TEST:
        test_indices = range(1, 3)

    print(f"{'영상':<10} {'정확도':>8}   {'GT 이상구간'}")
    print("-" * 40)

    accs = []
    for i in test_indices:
        acc = evaluate_test(i, threshold, gt_ranges)
        start, end = gt_ranges[i - 1]
        print(f"Test{i:03d}    {acc*100:6.1f}%   {start} ~ {end}")
        accs.append(acc)

    print("-" * 40)
    print(f"전체 평균 정확도: {np.mean(accs)*100:.1f}%")
    print(f"\n결과 그래프는 '{OUTPUT_DIR}/' 폴더에 저장되었습니다.")


if __name__ == "__main__":
    main()