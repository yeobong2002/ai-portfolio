"""
UCSD Ped2 - 움직임 크기(Magnitude) + 방향(Direction) 결합 이상 탐지 분석
=========================================================================

가설:
  일부 이상행동(특히 Test009, Test010 - magnitude 기반에서 정확도가 매우 낮았던 영상)은
  "속도"가 아니라 "이동 방향의 일관성 붕괴"로 나타날 수 있다.

지표:
  - Magnitude score : 프레임 내 상위 5% 픽셀의 flow magnitude (P95)
                       -> 기존 evaluate_ucsd.py와 동일한 지표
  - Direction score : magnitude로 가중된 방향 일관성 R (0~1)
                       R = |sum(mag * e^(i*angle))| / sum(mag)
                       R=1 -> 모든 픽셀이 같은 방향 (정상적인 흐름)
                       R=0 -> 방향이 뒤섞임 (역주행/혼란)

판정 방식 (3가지 비교):
  - Magnitude-only : magnitude_p95 > mag_threshold
  - Direction-only : R < dir_threshold  (방향 일관성이 비정상적으로 낮음)
  - Combined       : 위 둘 중 하나라도 True (OR)

임계값은 Train(정상) 데이터의 평균±3σ로 산출합니다 (evaluate_ucsd.py와 동일한 방식).
"""

import re
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정 (Windows: 맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# ─── 설정 ────────────────────────────────────────────────────────────────────

BASE_DIR = r"C:\Users\IBYEO\PycharmProjects\grad_school\course_cv\src\project\final\data\UCSDped2"      # ← 실제 경로로 수정
OUTPUT_DIR = "ucsd_results_direction"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FARNEBACK_PARAMS = dict(
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)

# 그래프로 자세히 볼 영상: Test004(대조군, magnitude로 잘 됐던 영상) + Test009/010(문제 영상)
PLOT_TESTS = [4, 9, 10]


# ─── 1. 프레임별 (Magnitude P95, Direction R) 계산 ─────────────────────────────

def compute_motion_features(folder):
    """
    폴더 안의 .tif 프레임들로부터 프레임별 (magnitude_p95, direction_R)을 계산합니다.
    Farneback Optical Flow를 한 번만 계산해서 두 지표를 동시에 추출합니다.

    반환 길이는 프레임 수와 동일.
    (첫 프레임: magnitude=0.0, direction=1.0 으로 채움 - '변화 없음/일관됨' 의미)
    """
    files = sorted(glob.glob(os.path.join(folder, "*.tif")))
    mags, dirs = [0.0], [1.0]
    prev = None

    for f in files:
        gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, gray, None, **FARNEBACK_PARAMS)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # ── Magnitude: 상위 5% 픽셀 ──
            mags.append(float(np.percentile(mag, 95)))

            # ── Direction: magnitude로 가중된 방향 일관성 R ──
            total = float(np.sum(mag))
            if total > 1e-6:
                mx = float(np.sum(mag * np.cos(ang)))
                my = float(np.sum(mag * np.sin(ang)))
                R = float(np.sqrt(mx ** 2 + my ** 2) / total)
            else:
                R = 1.0
            dirs.append(R)

        prev = gray

    return mags, dirs


# ─── 2. .m 파일에서 Ground Truth 파싱 (evaluate_ucsd.py와 동일) ─────────────────

def parse_all_ground_truths(m_file_path):
    with open(m_file_path) as f:
        content = f.read()
    matches = re.findall(r"gt_frame\s*=\s*\[(\d+):(\d+)\]", content)
    return [(int(a), int(b)) for a, b in matches]


# ─── 3. Train 데이터로 두 지표의 임계값 산출 ─────────────────────────────────────

def compute_thresholds():
    train_folders = sorted(glob.glob(os.path.join(BASE_DIR, "Train", "Train*")))
    print(f"[Train] {len(train_folders)}개 영상에서 정상 통계 수집 중...")

    all_mags, all_dirs = [], []
    for folder in train_folders:
        mags, dirs = compute_motion_features(folder)
        all_mags += mags[1:]
        all_dirs += dirs[1:]
        print(f"  - {os.path.basename(folder)}: {len(mags)} 프레임 처리 완료")

    mag_mean, mag_std = np.mean(all_mags), np.std(all_mags)
    dir_mean, dir_std = np.mean(all_dirs), np.std(all_dirs)

    mag_threshold = mag_mean + 3 * mag_std
    dir_threshold = max(0.0, dir_mean - 3 * dir_std)  # R은 0 미만이 될 수 없음

    print(f"\n  [Magnitude] 평균={mag_mean:.4f}, 표준편차={mag_std:.4f}")
    print(f"    -> 임계값 (μ+3σ) = {mag_threshold:.4f}")
    print(f"  [Direction] 평균(R)={dir_mean:.4f}, 표준편차={dir_std:.4f}")
    print(f"    -> 임계값 (μ-3σ) = {dir_threshold:.4f}  (R이 이보다 작으면 '방향 혼란')\n")

    return mag_threshold, dir_threshold


# ─── 4. Test 영상 평가 (Magnitude / Direction / Combined 비교) ──────────────────

def evaluate_test(video_num, mag_th, dir_th, gt_ranges):
    folder = os.path.join(BASE_DIR, "Test", f"Test{video_num:03d}")
    mags, dirs = compute_motion_features(folder)
    n = len(mags)

    start, end = gt_ranges[video_num - 1]
    gt = np.zeros(n)
    gt[start - 1: min(end, n)] = 1

    mags_arr, dirs_arr = np.array(mags), np.array(dirs)

    pred_mag = (mags_arr > mag_th).astype(int)
    pred_dir = (dirs_arr < dir_th).astype(int)
    pred_combined = ((pred_mag == 1) | (pred_dir == 1)).astype(int)

    acc_mag = float(np.mean(gt == pred_mag))
    acc_dir = float(np.mean(gt == pred_dir))
    acc_combined = float(np.mean(gt == pred_combined))

    if video_num in PLOT_TESTS:
        plot_test(video_num, mags_arr, dirs_arr, gt, mag_th, dir_th,
                  acc_mag, acc_dir, acc_combined)

    return acc_mag, acc_dir, acc_combined


def plot_test(video_num, mags, dirs, gt, mag_th, dir_th, acc_mag, acc_dir, acc_combined):
    n = len(mags)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # ── 위: Magnitude ──
    axes[0].plot(mags, color="steelblue", label="Magnitude (P95)")
    axes[0].axhline(mag_th, color="orange", linestyle="--", label="임계값")
    ymax = max(mags) * 1.05 if max(mags) > 0 else 1
    axes[0].fill_between(range(n), 0, ymax,
                          where=gt == 1, color="red", alpha=0.12, label="GT 이상구간")
    axes[0].set_title(f"Test{video_num:03d} - Magnitude (정확도 {acc_mag*100:.1f}%)")
    axes[0].set_ylabel("Magnitude (P95)")
    axes[0].legend(loc="upper right", fontsize=8)

    # ── 아래: Direction ──
    axes[1].plot(dirs, color="seagreen", label="방향 일관성 R")
    axes[1].axhline(dir_th, color="orange", linestyle="--", label="임계값")
    axes[1].fill_between(range(n), 0, 1,
                          where=gt == 1, color="red", alpha=0.12, label="GT 이상구간")
    axes[1].set_title(
        f"Test{video_num:03d} - Direction (정확도 {acc_dir*100:.1f}%, "
        f"결합(Combined) {acc_combined*100:.1f}%)"
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("R (1=일관, 0=혼란)")
    axes[1].set_xlabel("Frame")
    axes[1].legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"test{video_num:03d}_direction.png")
    plt.savefig(save_path, dpi=150)
    plt.close()


# ─── 메인 실행 ──────────────────────────────────────────────────────────────

def main():
    mag_th, dir_th = compute_thresholds()

    m_file = os.path.join(BASE_DIR, "Test", "UCSDped2.m")
    gt_ranges = parse_all_ground_truths(m_file)

    print(f"{'영상':<10} {'Mag만':>8} {'Dir만':>8} {'결합':>8}   GT구간")
    print("-" * 50)

    results = []
    for i in range(1, 13):
        acc_mag, acc_dir, acc_combined = evaluate_test(i, mag_th, dir_th, gt_ranges)
        start, end = gt_ranges[i - 1]
        print(f"Test{i:03d}    {acc_mag*100:6.1f}% {acc_dir*100:6.1f}% {acc_combined*100:6.1f}%   {start}~{end}")
        results.append((acc_mag, acc_dir, acc_combined))

    results = np.array(results)
    print("-" * 50)
    print(f"전체 평균   {results[:,0].mean()*100:6.1f}% {results[:,1].mean()*100:6.1f}% {results[:,2].mean()*100:6.1f}%")
    print(f"\n비교 그래프는 '{OUTPUT_DIR}/' 폴더에 저장되었습니다 (Test{PLOT_TESTS}).")


if __name__ == "__main__":
    main()