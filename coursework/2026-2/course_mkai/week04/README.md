# 4주차: 지도학습 (Supervised Learning)

> 충북대학교 산업인공지능연구센터 | 강사: 김영진 (you359@cbnu.ac.kr)

---

## 목차

1. Introduction to Supervised Learning
2. Losses and Logits for Supervised Learning
3. Challenging Problems
4. Techniques

---

## 1. Introduction to Supervised Learning

### 지도학습이란?
Labeled Data(입력 + 정답 레이블)를 이용해 모델을 학습시키고, 새로운 데이터에 대한 예측을 수행하는 방법론.

**흐름**: `Labeled Data → Model Training → Prediction`

### 문제 유형에 따른 Output Layer 설계
신경망은 **문제에 따라 Output Layer와 Loss를 다르게 설계**해야 한다.

| Input (x) | Output (y) | Type |
|---|---|---|
| Home features | Price | Regression |
| Ad, user Info | Click on ad? (0/1) | Binary classification |
| Image | Object (1, …, 1000) | Multi-Class classification |
| Audio | Text transcript | Seq2seq (classification + alignment) |
| English | Korean | Seq2seq (classification + alignment) |
| Image, Radar info | Position of other cars | Multivariate (Multi-output) regression |
| Movie info. | Genres | Multi-Label classification |

---

## 2. Losses and Logits for Supervised Learning

### Loss란?
- 인공 신경망의 **예측 값(ŷ)**과 **실제 값(y)** 간의 차이(오류)
- **손실 함수(Cost/Loss Function)**: 이 차이(오류)를 정의하는 함수

### Logits란?
- 신경망의 **마지막 선형 계층에서 확률 변환(softmax 등)을 적용하기 전의 출력 값**
- 즉, 모델이 출력한 가공되지 않은 점수(raw score)
- → **문제에 따라 적합한 출력 활성화함수와 Loss를 설계**해야 함

---

### 2-1. Regression Problem

**출력 활성화 함수**: 일반적으로 없음(identity function). 출력 범위를 제한해야 하면 ReLU, sigmoid, tanh 사용 가능.

#### MAE (Mean Absolute Error, L1 Loss)

$$\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

- 이상치에 강건함 → 일부 노이즈가 있어도 안정적 학습 가능
- 작은 오차에는 민감하지 않아 정밀 예측에서 한계
- 모든 구간에서 gradient가 동일 → 학습 수렴이 어려움

#### MSE (Mean Squared Error, L2 Loss)

$$\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

- 작은 오차에 민감함 → 제곱 항으로 인해 작은 오차도 학습에 반영(유용함)
- 이상치에 취약함 → 제곱 항으로 인해 이상치가 손실을 크게 만듦
- 함수가 convex하여 gradient가 달라지므로 수렴에 용이

#### Huber Loss (MAE와 MSE의 절충안)

$$
L_\delta(y,\hat{y}) =
\begin{cases}
\frac{1}{2}(y-\hat{y})^2 & \text{if } |y-\hat{y}| \le \delta \\
\delta\left(|y-\hat{y}| - \frac{1}{2}\delta\right) & \text{if } |y-\hat{y}| > \delta
\end{cases}
$$

- MAE의 장점과 MSE의 장점을 결합
- 계산 비용이 다소 높음
- 하이퍼파라미터 δ 설정 필요

> 실습: `1.nn_regression_loss_comparison.ipynb`

---

### 2-2. Classification Problem

#### 이진 분류 (Binary Classification)
- 출력 차원: 1개 class
- 출력 활성화 함수: 일반적으로 **sigmoid**
- 손실 함수: **Binary Cross-Entropy (BCE)**, Log loss라고도 함

$$L_{BCE} = y\log\hat{y} + (1-y)\log(1-\hat{y})$$

#### 다중 클래스 분류 (Multi-Class Classification)
- 출력 차원: K개 class
- 출력 활성화 함수: 일반적으로 **softmax**
- 손실 함수: **Categorical Cross-Entropy (CE)**

$$softmax(y_i) = \frac{e^{y_i}}{\sum_{i=0}^{n}e^{y_i}} \qquad L_{CE} = -\sum_{k=1}^{K}y_k\log(\hat{y}_k)$$

- CE는 BCE의 일반화된 버전 (K=2 → K>2)

#### 다중 레이블 분류 (Multi-Label Classification)
- 출력 차원: K개 class
- 출력 활성화 함수: 일반적으로 **sigmoid** (각 클래스 독립 판단)
- 손실 함수: **Binary Cross-Entropy (BCE)** — 각 class별 BCE를 평균

$$L_{BCE} = -\frac{1}{K}\sum_{k=1}^{K}\left(y_k\log\hat{y}_k + (1-y_k)\log(1-\hat{y}_k)\right)$$

> 실습: `2.nn_classification_examples.ipynb`, `2-1.nn_multiclass_classification.ipynb`

---

## 3. Challenging Problems

### 3-1. Vanishing Gradient (기울기 소실)
- Sigmoid 함수는 입력 값이 커지거나 작아질수록 미분값(derivative)이 0에 가까워짐
- 역전파(Backpropagation) 시 Chain Rule에 의해 여러 층을 거치며 gradient가 계속 곱해지는데, 각 항이 작으면 **앞쪽 layer로 갈수록 gradient가 점점 작아져 소실**

$$\frac{\partial C_0}{\partial w^{(L-1)}} = \frac{\partial z^{(L-1)}}{\partial w^{(L-1)}}\frac{\partial a^{(L-1)}}{\partial z^{(L-1)}}\frac{\partial z^{(L)}}{\partial a^{(L-1)}}\frac{\partial a^{(L)}}{\partial z^{(L)}}\frac{\partial C_0}{\partial a^{(L)}}$$

### 3-2. Overfitting (과적합)
- 딥러닝 학습이란 `y = f(x)`를 만족하는 f()를 찾는 것
- 학습된 모델이 새로운(학습에 쓰이지 않은) 데이터로 **일반화되지 않는 경우** 과적합되었다고 함
- 발생 이유 1: 학습 데이터의 다양성 부족
- 발생 이유 2: 모델의 복잡도(파라미터 양) 과다

### 3-3. Class Imbalanced (클래스 불균형)
- 특정 클래스의 데이터 수가 다른 클래스보다 현저히 적은 경우
- Decision boundary가 다수 클래스 쪽으로 치우쳐 소수 클래스 인식률이 저하됨

### 3-4. Data Scarcity (데이터 부족)
- 항공/우주, 제조, 사이버보안, 재해/재난 등 분야에서 흔히 발생
- 데이터가 매우 적거나(1~10개 샘플) 아예 존재하지 않는 경우
- 또는 정상 데이터만 존재하는 경우 (normal: n, abnormal: 0)

> 실습: `3.vanishing_gradient_demo.ipynb`, `4-1.overfitting_dataset_size_demo.ipynb`, `4-2.overfitting_demo.ipynb`, `5.class_imbalanced_demo.ipynb`

---

## 4. Techniques (문제별 해결 기법)

### 4-1. Vanishing Gradient 해결
- **Leaky ReLU**: 음수 구간에서도 작은 기울기를 유지하여 gradient 소실 완화
- **Residual Connection**: 입력 x를 그대로 더해주는 skip connection (`x + F(x)`)으로 gradient가 직접 전달되는 경로 확보

### 4-2. Overfitting 해결
- **Early Stopping**: Validation error가 증가하기 시작하는 시점에 학습 중단
- **L1 Regularization**: 가중치의 절댓값 합을 손실에 추가 → 일부 가중치를 0으로 만들어 희소성(sparsity) 유도

$$J_{L1}(W) = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2 + \lambda\sum_{j=1}^{d}|w_j|$$

- **L2 Regularization**: 가중치의 제곱 합을 손실에 추가 → 가중치 크기를 전반적으로 작게 유지

$$J_{L2}(W) = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2 + \frac{\lambda}{2}\sum_{j=1}^{d}w_j^2$$

- **Dropout**: 학습 시 일부 뉴런을 확률적으로 비활성화하여 특정 뉴런에 대한 과의존 방지

### 4-3. Class Imbalanced 해결
- **Data Augmentation**: Flip, Rotation, Blur, Exposure, Contrast, Grayscale 등으로 소수 클래스 데이터 증강
- **Class Weighting**: 손실 함수 계산 시 소수 클래스에 더 큰 weight, 다수 클래스에는 작은 weight 부여

$$L = \frac{1}{N}\sum_{i=1}^{N}w_{y_i}\cdot\ell(\hat{y}_i, y_i)$$

### 4-4. Data Scarcity 해결

| 접근 | 세부 기법 |
|---|---|
| Data Expansion | Data Augmentation, Synthetic Data Generation, Active Learning |
| Knowledge Transfer | Transfer Learning, Domain Adaptation |
| Weakly Supervision | Semi-supervised, Self-supervised, Unsupervised Learning |
| Meta Learning | Few-Shot/One-Shot Learning, Meta Learning, Metric Learning |

> 실습: `6.residual_vs_plain_demo.ipynb`, `7.l1_l2_regularization_demo.ipynb`, `8.dropout_demo.ipynb`, `9.data_aug_class_weight_demo.ipynb`

---

## 다음 주 예고 (5주차)
**Learning Strategies: Unsupervised Learning** — 비지도학습의 동작 방식과 학습 방법, 응용 방안 등