"""
미세먼지 예측 전체 파이프라인
- 전처리 → 학습 → 테스트 → 평가 → 모델 비교
- 모델   : Linear Regression, Ridge, Lasso, RandomForest
- 타겟   : PM2.5, PM10
- 평가   : RMSE, MAE, R²
- 각 단계별 소요 시간 출력
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ────────────────────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────────────────────
PM25_PATH = "../../../data/raw/airkorea_pm2_5.csv"
PM10_PATH = "../../../data/raw/airkorea_pm1_0.csv"
OUTPUT_CSV = "../../../data/processed/preprocessed_airkorea.csv"
SPLIT_DATE  = "2025-07-01"
TARGET_COLS = ["pm25", "pm10"]

DOMAIN_BOUNDS = {
    "pm25": (0, 300),
    "pm10": (0, 500),
}

# 비교할 모델 목록 (이름 : 모델 객체)
MODELS = {
    # ── 선형 모델 ─────────────────────────────────────────────
    "Linear Regression" : LinearRegression(),
    "Ridge (α=0.01)"    : Ridge(alpha=0.01),
    "Ridge (α=1.0)"     : Ridge(alpha=1.0),
    "Ridge (α=100)"     : Ridge(alpha=100),
    "Lasso (α=0.01)"    : Lasso(alpha=0.01, max_iter=10000),
    "Lasso (α=0.1)"     : Lasso(alpha=0.1,  max_iter=10000),
    "Lasso (α=1.0)"     : Lasso(alpha=1.0,  max_iter=10000),

    # ── 앙상블 모델 ───────────────────────────────────────────
    "Random Forest"     : RandomForestRegressor(
                              n_estimators=100,   # 트리 개수
                              max_depth=10,        # 트리 최대 깊이
                              random_state=42,
                              n_jobs=-1            # 병렬 처리
                          ),
}


# ────────────────────────────────────────────────────────────────
# 유틸 · 타이머
# ────────────────────────────────────────────────────────────────
class StepTimer:
    """각 단계의 시작/종료 시각과 소요 시간을 기록합니다."""

    def __init__(self):
        self.records = []
        self._start  = None
        self._name   = None

    def start(self, name: str):
        self._name  = name
        self._start = time.time()
        ts = time.strftime("%H:%M:%S")
        print(f"\n{'='*60}")
        print(f"  [{ts}] {name} 시작")
        print(f"{'='*60}")

    def end(self):
        elapsed = time.time() - self._start
        ts = time.strftime("%H:%M:%S")
        print(f"  [{ts}] {self._name} 완료  ({elapsed:.2f}초)")
        self.records.append({"단계": self._name, "소요시간(초)": round(elapsed, 2)})

    def summary(self):
        print(f"\n{'='*60}")
        print("  전체 소요 시간 요약")
        print(f"{'='*60}")
        total = 0
        for r in self.records:
            print(f"  {r['단계']:<30} {r['소요시간(초)']:>6.2f}초")
            total += r["소요시간(초)"]
        print(f"  {'─'*40}")
        print(f"  {'총 합계':<30} {total:>6.2f}초")


timer = StepTimer()


# ────────────────────────────────────────────────────────────────
# 한글 폰트
# ────────────────────────────────────────────────────────────────
def set_korean_font():
    candidates = [
        "NanumGothic", "NanumBarunGothic", "AppleGothic",
        "Malgun Gothic", "Noto Sans KR", "DejaVu Sans"
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False


# ════════════════════════════════════════════════════════════════
# STEP 1 · 전처리
# ════════════════════════════════════════════════════════════════
def preprocess_raw(filepath: str, col_name: str) -> pd.DataFrame:
    """
    단일 CSV 파일을 읽어 시간 단위 Long 형식으로 변환합니다.

    [처리 순서]
    1. cp949 인코딩으로 파일 로드
    2. 불필요 컬럼 제거 (측정망, 측정소명, 지점ID, 지점)
    3. '-' 문자를 NaN으로 치환 후 숫자형 변환
    4. 같은 날 시간 방향 선형 보간 (axis=1)
    5. 전후 날짜로 나머지 결측 채움
    6. Wide(날짜×시간) → Long(datetime, 측정값) 변환
    """
    df = pd.read_csv(filepath, encoding="cp949")
    df["date"] = pd.to_datetime(df["date"])

    drop_cols = [c for c in ["측정망", "측정소명", "지점ID", "지점"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    hour_cols = [c for c in df.columns if "시" in c]
    df[hour_cols] = df[hour_cols].replace("-", np.nan)
    df[hour_cols] = df[hour_cols].apply(pd.to_numeric)
    df[hour_cols] = df[hour_cols].interpolate(method="linear", axis=1)
    df[hour_cols] = df[hour_cols].ffill().bfill()

    df_long = df.melt(id_vars="date", var_name="hour", value_name=col_name)
    df_long["hour"] = df_long["hour"].str.replace("시", "").astype(int)
    df_long["datetime"] = df_long["date"] + pd.to_timedelta(df_long["hour"] - 1, unit="h")

    return df_long.sort_values("datetime").reset_index(drop=True)[["datetime", col_name]]


def clip_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    2단계 이상치 처리
      1단계 : 도메인 클리핑 (센서 오류 범위 제거)
      2단계 : IQR(1.5) 클리핑 (통계적 극단값 처리)
    """
    df = df.copy()
    for col in cols:
        d_lower, d_upper = DOMAIN_BOUNDS[col]
        n_domain = ((df[col] < d_lower) | (df[col] > d_upper)).sum()
        df[col] = df[col].clip(lower=d_lower, upper=d_upper)
        print(f"  [{col}] 1단계 도메인 클리핑 : {n_domain}개  (범위 {d_lower}~{d_upper})")

        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        i_lower, i_upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_iqr = ((df[col] < i_lower) | (df[col] > i_upper)).sum()
        df[col] = df[col].clip(lower=i_lower, upper=i_upper)
        print(f"  [{col}] 2단계 IQR    클리핑 : {n_iqr}개  (범위 {i_lower:.1f}~{i_upper:.1f})")
    return df


def add_features(df: pd.DataFrame, target_cols: list) -> pd.DataFrame:
    """
    피처 엔지니어링
      - 시간 주기 : hour_sin/cos, month_sin/cos
      - 날짜 속성 : dayofweek, is_weekend, season
      - Lag       : lag1, lag3, lag24 (각 타겟별)
      - 이동평균  : roll3, roll24 (피처용, 타겟 직접 스무딩 제외)
    """
    df = df.copy()
    df["hour_sin"]   = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df.index.hour / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df.index.month / 12)
    df["dayofweek"]  = df.index.dayofweek
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["season"]     = df.index.month % 12 // 3

    for col in target_cols:
        df[f"{col}_lag1"]   = df[col].shift(1)
        df[f"{col}_lag3"]   = df[col].shift(3)
        df[f"{col}_lag24"]  = df[col].shift(24)
        df[f"{col}_roll3"]  = df[col].rolling(3).mean()
        df[f"{col}_roll24"] = df[col].rolling(24).mean()

    return df.dropna()


def run_preprocessing() -> pd.DataFrame:
    """전처리 전체 흐름을 실행하고 완성된 DataFrame을 반환합니다."""
    timer.start("STEP 1 · 전처리")

    print("  [1-1] 파일 로드 및 결측 처리")
    df_pm25 = preprocess_raw(PM25_PATH, "pm25")
    df_pm10 = preprocess_raw(PM10_PATH, "pm10")
    print(f"        PM2.5 : {len(df_pm25):,}개 레코드")
    print(f"        PM10  : {len(df_pm10):,}개 레코드")

    print("  [1-2] 병합 (inner join)")
    df = pd.merge(df_pm25, df_pm10, on="datetime", how="inner")
    df = df.set_index("datetime").sort_index()
    print(f"        병합 후 : {len(df):,}개  ({df.index.min().date()} ~ {df.index.max().date()})")

    print("  [1-3] 이상치 처리")
    df = clip_outliers(df, cols=["pm25", "pm10"])

    print("  [1-4] 피처 엔지니어링")
    df = add_features(df, target_cols=TARGET_COLS)
    print(f"        피처 수 : {len(df.columns) - len(TARGET_COLS)}개  /  전체 레코드 : {len(df):,}개")

    df.to_csv(OUTPUT_CSV)
    print(f"  [1-5] 전처리 완료 파일 저장 → {OUTPUT_CSV}")

    timer.end()
    return df


# ════════════════════════════════════════════════════════════════
# STEP 2 · 학습
# ════════════════════════════════════════════════════════════════
def run_training(df: pd.DataFrame):
    """
    Train 데이터로 모든 모델을 학습합니다.

    Returns
    -------
    trained_models : dict  { 모델명: { 타겟명: 학습된 모델 } }
    X_train, y_train, X_test, y_test, feature_cols, scaler
    """
    timer.start("STEP 2 · 학습")

    train = df[df.index < SPLIT_DATE]
    test  = df[df.index >= SPLIT_DATE]
    feature_cols = [c for c in df.columns if c not in TARGET_COLS]

    train_ratio = len(train) / len(df) * 100
    print(f"  Train : {len(train):,}개  ({train_ratio:.1f}%)  ~ {SPLIT_DATE} 이전")
    print(f"  Test  : {len(test):,}개  ({100-train_ratio:.1f}%)  ~ {SPLIT_DATE} 이후")

    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(train[feature_cols])
    X_test  = scaler.transform(test[feature_cols])
    y_train = train[TARGET_COLS].values
    y_test  = test[TARGET_COLS].values

    import copy
    trained_models  = {}
    model_train_time = {}

    for model_name, model in MODELS.items():
        trained_models[model_name]   = {}
        model_train_time[model_name] = {}
        for i, target in enumerate(TARGET_COLS):
            m = copy.deepcopy(model)
            t0 = time.time()
            m.fit(X_train, y_train[:, i])
            elapsed = time.time() - t0
            trained_models[model_name][target]   = m
            model_train_time[model_name][target] = elapsed
        total_t = sum(model_train_time[model_name].values())
        ts = time.strftime("%H:%M:%S")
        detail = "  ".join(
            f"{t.upper()}={model_train_time[model_name][t]*1000:.1f}ms"
            for t in TARGET_COLS
        )
        print(f"  [{ts}] [{model_name}] 학습 완료  ({total_t*1000:.1f}ms)  →  {detail}")

    timer.end()
    return (trained_models, model_train_time,
            X_train, y_train, X_test, y_test, feature_cols, scaler)


# ════════════════════════════════════════════════════════════════
# STEP 3 · 테스트 (예측)
# ════════════════════════════════════════════════════════════════
def run_testing(trained_models: dict, X_test: np.ndarray) -> dict:
    """
    Test 데이터로 모든 모델의 예측값을 생성합니다.

    Returns
    -------
    all_preds : dict  { 모델명: { 타겟명: 예측값 배열 } }
    """
    timer.start("STEP 3 · 테스트 (예측)")

    all_preds       = {}
    model_test_time = {}

    for model_name, target_models in trained_models.items():
        all_preds[model_name]       = {}
        model_test_time[model_name] = {}
        for target, model in target_models.items():
            t0 = time.time()
            all_preds[model_name][target] = model.predict(X_test)
            elapsed = time.time() - t0
            model_test_time[model_name][target] = elapsed
        total_t = sum(model_test_time[model_name].values())
        ts = time.strftime("%H:%M:%S")
        detail = "  ".join(
            f"{t.upper()}={model_test_time[model_name][t]*1000:.2f}ms"
            for t in TARGET_COLS
        )
        print(f"  [{ts}] [{model_name}] 예측 완료  ({total_t*1000:.2f}ms)  →  {detail}")

    timer.end()
    return all_preds, model_test_time


# ════════════════════════════════════════════════════════════════
# STEP 4 · 평가 및 모델 비교
# ════════════════════════════════════════════════════════════════
def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    RMSE, MAE, R² 계산

    RMSE : 예측 오차 제곱 평균의 제곱근  (단위: μg/m³, 큰 오차에 민감)
    MAE  : 예측 오차 절댓값 평균         (단위: μg/m³, 이상치에 덜 민감)
    R²   : 결정계수 0~1, 1에 가까울수록 우수
    """
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE" : mean_absolute_error(y_true, y_pred),
        "R2"  : r2_score(y_true, y_pred),
    }


def build_summary(all_preds: dict, y_test: np.ndarray,
                  model_train_time: dict, model_test_time: dict) -> pd.DataFrame:
    """모든 모델의 평가 결과와 소요 시간을 DataFrame으로 정리합니다."""
    rows = []
    for model_name, preds in all_preds.items():
        for i, target in enumerate(TARGET_COLS):
            m = evaluate(y_test[:, i], preds[target])
            rows.append({
                "모델"        : model_name,
                "타겟"        : target.upper(),
                "RMSE"       : round(m["RMSE"], 3),
                "MAE"        : round(m["MAE"],  3),
                "R²"         : round(m["R2"],   3),
                "학습(ms)"   : round(model_train_time[model_name][target] * 1000, 2),
                "테스트(ms)" : round(model_test_time[model_name][target]  * 1000, 2),
            })
    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame):
    """결과 요약 테이블과 최고 성능 모델을 출력합니다."""
    print("\n  ┌─ 전체 모델 비교 결과 ──────────────────────────────────────────────┐")
    for target in [t.upper() for t in TARGET_COLS]:
        sub = summary[summary["타겟"] == target]
        print(f"\n  [{target}]")
        print(f"  {'모델':<20} {'RMSE':>7} {'MAE':>7} {'R²':>7} {'학습(ms)':>10} {'테스트(ms)':>11} {'RMSE 차이':>10}")
        print(f"  {'─'*78}")
        base_rmse = sub[sub["모델"] == "Linear Regression"]["RMSE"].values[0]
        for _, row in sub.iterrows():
            diff = row["RMSE"] - base_rmse
            diff_str = "(기준)" if diff == 0 else f"({diff:+.3f})"
            print(f"  {row['모델']:<20} {row['RMSE']:>7.3f} {row['MAE']:>7.3f} {row['R²']:>7.3f}"
                  f" {row['학습(ms)']:>10.2f} {row['테스트(ms)']:>11.2f} {diff_str:>10}")
    print("\n  └───────────────────────────────────────────────────────────────────────┘")

    print("\n  [최고 성능 모델 (RMSE 기준)]")
    for target in [t.upper() for t in TARGET_COLS]:
        sub  = summary[summary["타겟"] == target]
        best = sub.loc[sub["RMSE"].idxmin()]
        print(f"  {target} → {best['모델']}  "
              f"(RMSE: {best['RMSE']:.3f} / MAE: {best['MAE']:.3f} / R²: {best['R²']:.3f} / "
              f"학습: {best['학습(ms)']:.2f}ms / 테스트: {best['테스트(ms)']:.2f}ms)")


def print_features(feature_cols: list, trained_models: dict):
    """
    피처 목록과 선형 모델의 회귀계수(절댓값 Top 10)를 출력합니다.
    Random Forest 등 비선형 모델은 Feature Importance를 대신 출력합니다.
    """
    # 피처 목록 전체 출력
    print(f"\n  피처 목록 (총 {len(feature_cols)}개)")
    print(f"  {'─'*40}")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")

    # 모델별 피처 중요도
    for model_name, target_models in trained_models.items():
        print(f"\n  [{model_name}] 피처 영향도 Top 10")
        print(f"  {'─'*40}")

        for target, model in target_models.items():
            if hasattr(model, "coef_"):
                # 선형 모델 : 회귀계수 절댓값 기준
                coef_df = (
                    pd.DataFrame({"피처": feature_cols, "값": model.coef_})
                    .assign(절댓값=lambda x: x["값"].abs())
                    .sort_values("절댓값", ascending=False)
                    .drop(columns="절댓값")
                    .reset_index(drop=True)
                )
                print(f"  [{target.upper()}] 회귀계수 (절댓값 내림차순)")
                print(coef_df.head(10).to_string(index=False))

            elif hasattr(model, "feature_importances_"):
                # 앙상블 모델 : Feature Importance 기준
                imp_df = (
                    pd.DataFrame({"피처": feature_cols, "중요도": model.feature_importances_})
                    .sort_values("중요도", ascending=False)
                    .reset_index(drop=True)
                )
                print(f"  [{target.upper()}] Feature Importance (내림차순)")
                print(imp_df.head(10).to_string(index=False))


def run_evaluation(all_preds: dict, y_test: np.ndarray,
                   trained_models: dict, feature_cols: list,
                   model_train_time: dict, model_test_time: dict) -> pd.DataFrame:
    """평가 지표 계산, 결과 비교, 시각화를 수행합니다."""
    timer.start("STEP 4 · 평가 및 모델 비교")

    print("  [4-1] 평가 지표 계산")
    summary = build_summary(all_preds, y_test, model_train_time, model_test_time)

    print("  [4-2] 모델 비교 결과")
    print_summary(summary)

    print("\n  [4-3] 피처 목록 및 피처 영향도")
    print_features(feature_cols, trained_models)

    print("\n  [4-4] 시각화")
    plot_metrics_comparison(summary)
    plot_predictions_comparison(all_preds, y_test)
    plot_scatter_comparison(all_preds, y_test)

    timer.end()
    return summary


# ────────────────────────────────────────────────────────────────
# 시각화
# ────────────────────────────────────────────────────────────────
# 모델별 색상 팔레트
MODEL_COLORS = {
    "Linear Regression" : "#2C2C2A",
    "Ridge (α=0.01)"    : "#185FA5",
    "Ridge (α=1.0)"     : "#378ADD",
    "Ridge (α=100)"     : "#85B7EB",
    "Lasso (α=0.01)"    : "#3B6D11",
    "Lasso (α=0.1)"     : "#639922",
    "Lasso (α=1.0)"     : "#97C459",
    "Random Forest"     : "#854F0B",
    "XGBoost"           : "#D85A30",
}


def plot_metrics_comparison(summary: pd.DataFrame,
                             save_path="metrics_comparison.png"):
    """RMSE / MAE / R² 를 모델별 · 타겟별로 비교합니다."""
    metric_list  = ["RMSE", "MAE", "R²"]
    model_names  = list(MODELS.keys())
    target_names = [t.upper() for t in TARGET_COLS]

    fig, axes = plt.subplots(len(target_names), len(metric_list),
                             figsize=(15, 5 * len(target_names)))
    fig.suptitle("모델별 평가 지표 비교 (Linear Regression vs Ridge vs Lasso)",
                 fontsize=13, fontweight="bold", y=1.01)

    for ti, target in enumerate(target_names):
        sub = summary[summary["타겟"] == target]
        for mi, metric in enumerate(metric_list):
            ax   = axes[ti][mi]
            vals = [sub[sub["모델"] == m][metric].values[0] for m in model_names]
            clrs = [MODEL_COLORS[m] for m in model_names]

            bars = ax.bar(range(len(model_names)), vals,
                          color=clrs, alpha=0.85, width=0.6)
            ax.set_title(f"{target} · {metric}", fontsize=11)
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"        저장 → {save_path}")
    plt.close()


def plot_predictions_comparison(all_preds: dict, y_test: np.ndarray,
                                 n_points: int = 200,
                                 save_path="predictions_comparison.png"):
    """실제값 vs 각 모델 예측값 시계열 비교 그래프."""
    fig, axes = plt.subplots(len(TARGET_COLS), 1,
                             figsize=(15, 5 * len(TARGET_COLS)))
    if len(TARGET_COLS) == 1:
        axes = [axes]

    for i, (ax, target) in enumerate(zip(axes, TARGET_COLS)):
        idx    = np.arange(n_points)
        y_true = y_test[:n_points, i]

        ax.plot(idx, y_true, color="black", linewidth=1.5,
                label="실제값", zorder=10)

        for model_name, preds in all_preds.items():
            y_pred = preds[target][:n_points]
            ax.plot(idx, y_pred,
                    color=MODEL_COLORS[model_name],
                    linewidth=0.9, linestyle="--",
                    label=model_name, alpha=0.8)

        ax.set_title(f"{target.upper()} 실제값 vs 예측값 비교 (처음 {n_points}시간)",
                     fontsize=12)
        ax.set_xlabel("시간 (h)")
        ax.set_ylabel("농도 (μg/m³)")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"        저장 → {save_path}")
    plt.close()


def plot_scatter_comparison(all_preds: dict, y_test: np.ndarray,
                             save_path="scatter_comparison.png"):
    """모델별 실제값 vs 예측값 산점도 (타겟별 행, 모델별 열)."""
    model_names = list(MODELS.keys())
    n_models    = len(model_names)
    n_targets   = len(TARGET_COLS)

    fig, axes = plt.subplots(n_targets, n_models,
                             figsize=(3.5 * n_models, 4 * n_targets))
    fig.suptitle("실제값 vs 예측값 산점도 비교",
                 fontsize=13, fontweight="bold")

    for ti, target in enumerate(TARGET_COLS):
        for mi, model_name in enumerate(model_names):
            ax     = axes[ti][mi]
            y_true = y_test[:, ti]
            y_pred = all_preds[model_name][target]

            ax.scatter(y_true, y_pred, alpha=0.15, s=3,
                       color=MODEL_COLORS[model_name])
            lim = [min(y_true.min(), y_pred.min()),
                   max(y_true.max(), y_pred.max())]
            ax.plot(lim, lim, "r--", linewidth=1.0)

            m = evaluate(y_true, y_pred)
            ax.set_title(f"{model_name}\n{target.upper()}  R²={m['R2']:.3f}",
                         fontsize=8)
            ax.set_xlabel("실제값", fontsize=7)
            ax.set_ylabel("예측값", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"        저장 → {save_path}")
    plt.close()


# ════════════════════════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════════════════════════
def main():
    set_korean_font()

    pipeline_start = time.time()
    print(f"\n{'='*60}")
    print(f"  미세먼지 예측 전체 파이프라인")
    print(f"  시작 시각 : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # STEP 1 · 전처리
    df = run_preprocessing()

    # STEP 2 · 학습
    (trained_models, model_train_time,
     X_train, y_train, X_test, y_test, feature_cols, scaler) = run_training(df)

    # STEP 3 · 테스트
    all_preds, model_test_time = run_testing(trained_models, X_test)

    # STEP 4 · 평가 및 모델 비교
    summary = run_evaluation(all_preds, y_test, trained_models, feature_cols,
                             model_train_time, model_test_time)

    # 전체 요약
    timer.summary()

    total_elapsed = time.time() - pipeline_start
    print(f"\n  종료 시각 : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  전체 소요 : {total_elapsed:.2f}초")
    print(f"\n{'='*60}")
    print("  파이프라인 완료!")
    print(f"{'='*60}\n")

    return trained_models, summary


if __name__ == "__main__":
    trained_models, summary = main()