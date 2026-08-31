# 🔥 YOLO & DeepSORT 기반 실시간 화재·연기 감지 및 추적 시스템
## Real-time Fire & Smoke Detection and Tracking System based on YOLO & DeepSORT

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Library](https://img.shields.io/badge/Library-Ultralytics-red.svg)](https://ultralytics.com/)
[![Library](https://img.shields.io/badge/Library-DeepSORT-green.svg)](https://github.com/levan92/deep_sort_realtime)

YOLO(You Only Look Once)와 DeepSORT 알고리즘을 결합하여 영상에서 **화재(fire)와 연기(smoke)를 실시간으로 감지하고 안정적으로 추적**하는 고성능 시스템입니다. 낮은 신뢰도의 미세한 객체까지 포착하고, 오탐지를 최소화하며, 안정적인 추적을 보장하는 데 중점을 두었습니다.

# 🚀 YOLOv8 표준 코드 가이드 (`ultralytics`)

이 문서는 `ultralytics` 라이브러리를 사용하여 YOLOv8 모델의 **추론(Inference)**, **학습(Training)**, **검증(Validation)**을 수행하는 표준 코드 예제를 제공합니다. 각 코드는 바로 실행할 수 있도록 구성되어 있으며, 필요한 부분만 수정하여 사용하시면 됩니다.

---

## 📖 목차
1.  [**추론 (Inference)**](#-1-추론-inference---이미지영상-예측)
2.  [**학습 (Training)**](#-2-학습-training---커스텀-데이터셋-학습)
3.  [**검증 (Validation)**](#-3-검증-validation---학습된-모델-성능-평가)

---

## 🚀 1. 추론 (Inference) - 이미지/영상 예측

학습된 모델을 사용하여 새로운 이미지, 동영상 또는 실시간 카메라 피드에서 객체를 탐지하는 기능입니다.

### 코드 예제
```python
from ultralytics import YOLO
import cv2

# 1. 사전 학습된 YOLOv8n 모델 로드
model = YOLO('yolov8n.pt')

# 2. 사용할 소스 설정 (아래 중 하나를 선택)
# source = 'path/to/your/image.jpg'    # 이미지 파일
source = 'path/to/your/video.mp4'      # 동영상 파일
# source = 0                             # 웹캠 (0번 카메라)

# 3. 모델을 사용하여 소스에 대한 예측 실행
# stream=True: 대용량 동영상 처리 시 메모리 효율성 증대
# show=True: 결과창을 실시간으로 화면에 표시
# save=True: 처리된 결과를 파일('runs/detect/predict*/...')로 저장
results = model(source, stream=True, show=True, save=True)

# 4. 결과 반복 처리 (스트림 모드)
# 'show=True' 옵션으로 인해 별도의 시각화 코드는 불필요합니다.
# 하지만 결과 데이터에 직접 접근하고 싶을 경우 아래와 같이 사용합니다.
for r in results:
    boxes = r.boxes         # 바운딩 박스 정보
    masks = r.masks         # 세그멘테이션 마스크 정보
    keypoints = r.keypoints # 키포인트(포즈) 정보

print("추론이 완료되었으며, 결과가 저장되었습니다.")
```
**실행 설명**: 위 코드는 `yolov8n.pt` 모델을 로드하여 지정된 `source`에 대해 객체 탐지를 수행합니다. `show=True` 옵션은 결과를 실시간으로 보여주며, `save=True`는 결과 영상/이미지를 자동으로 저장합니다.

---

## 🧠 2. 학습 (Training) - 커스텀 데이터셋 학습

사용자가 직접 구축한 데이터셋으로 YOLO 모델을 학습시켜 특정 객체를 탐지하는 모델을 생성합니다.

### 사전 준비 사항

학습을 시작하기 전, 데이터셋과 설정을 담은 `.yaml` 파일이 필요합니다.

1.  **데이터셋 폴더 구조**:
    ```
    /path/to/dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    ```

2.  **`data.yaml` 파일 작성**:
    ```yaml
    # 데이터셋 경로 설정
    train: /path/to/dataset/images/train
    val: /path/to/dataset/images/val

    # 클래스 정보
    nc: 2  # 클래스 개수
    names: ['fire', 'smoke'] # 클래스 이름 (순서 중요)
    ```

### 코드 예제
```python
from ultralytics import YOLO

# 1. 모델 로드
#    - 'yolov8n.pt': 처음부터 새로운 모델을 학습 (사전 학습된 가중치 활용)
#    - 'path/to/last.pt': 이전에 중단된 학습을 이어서 진행
model = YOLO('yolov8n.pt')

# 2. 모델 학습 실행
results = model.train(
    data='path/to/your/data.yaml',  # 위에서 작성한 yaml 파일 경로
    epochs=100,                     # 전체 데이터셋 반복 학습 횟수
    imgsz=640,                      # 입력 이미지 크기
    device=0,                       # 사용할 GPU 장치 번호 (CPU 사용 시 'cpu')
    name='custom_fire_smoke_model'  # 학습 결과가 저장될 폴더 이름
)

# 3. 학습 완료
# 결과는 'runs/detect/custom_fire_smoke_model' 폴더에 저장됩니다.
# 가장 성능이 좋은 모델은 해당 폴더 내 'weights/best.pt'로 저장됩니다.
print("학습이 완료되었습니다.")
```
**실행 설명**: `yolov8n.pt`를 기반으로 `data.yaml`에 정의된 커스텀 데이터셋을 `100 epochs` 동안 학습시킵니다. 학습 결과와 모델 가중치 파일(`best.pt`, `last.pt`)은 `runs/detect/custom_fire_smoke_model` 폴더에 저장됩니다.

---

## ✅ 3. 검증 (Validation) - 학습된 모델 성능 평가

학습된 커스텀 모델이 얼마나 정확하게 객체를 탐지하는지 성능 지표(mAP, Precision, Recall 등)를 통해 평가합니다.

### 코드 예제
```python
from ultralytics import YOLO

# 1. 평가할 모델 로드
#    - 학습 결과물 중 가장 성능이 좋았던 'best.pt' 모델을 로드합니다.
model = YOLO('runs/detect/custom_fire_smoke_model/weights/best.pt')

# 2. 모델 성능 검증 실행
metrics = model.val(
    data='path/to/your/data.yaml',  # 학습 시 사용했던 yaml 파일 경로
    split='val',                    # 'val' 또는 'test' 데이터셋 중 선택
    imgsz=640,
    device=0
)

# 3. 성능 지표 출력
print("--- 모델 성능 지표 ---")
print(f"mAP50-95: {metrics.box.map:.4f}")    # mAP@.5:.95
print(f"mAP50: {metrics.box.map50:.4f}")     # mAP@.5
print(f"mAP75: {metrics.box.map75:.4f}")     # mAP@.75
print("-----------------------")

# 상세 검증 결과(그래프, 이미지 등)는 'runs/detect/val*/' 폴더에 저장됩니다.
```
**실행 설명**: 학습을 통해 얻은 `best.pt` 모델을 로드하고, `data.yaml`에 명시된 검증 데이터셋(`val`)을 기준으로 성능을 평가합니다. 터미널에 주요 mAP 점수가 출력되며, 상세 리포트는 `runs/detect/val` 폴더에서 확인할 수 있습니다.
---

## ✨ 주요 기능 (Key Features)

* **고감도 감지**: 낮은 신뢰도(low-confidence)의 작은 불꽃이나 희미한 연기도 놓치지 않고 초기 감지
* **안정적인 추적**: DeepSORT를 활용하여 한 번 감지된 객체에 고유 ID를 부여하고, 가려지거나 빠르게 움직여도 끊김 없이 추적
* **오탐지 최소화**: 연속 프레임 검증 로직을 통해 낮은 임계값 설정으로 인해 발생할 수 있는 오탐지(False Positives)를 효과적으로 필터링
* **실시간 성능**: 추적기 업데이트 주기를 최적화하여 영상 끊김 없는 부드러운 실시간 분석 제공
* **결과 자동 저장**: 감지 및 추적 결과(바운딩 박스, ID)가 표시된 영상을 `output.mp4`와 같은 파일로 자동 저장

---

## 🛠️ 기술적 접근 (Technical Approach)

본 프로젝트는 두 가지 핵심적인 기술적 문제를 해결하는 데 집중했습니다.

### 1. 감지 민감도 향상 및 후처리 로직 적용

* **문제점**: 기본 YOLO 모델은 신뢰도가 낮은 객체(Low-confidence detections)를 필터링하여, 갑자기 발생하는 작은 불꽃이나 초기 연기를 놓치는 경향이 있었습니다.
* **해결 과정**:
    1.  **낮은 신뢰도 임계값 설정**: YOLO 모델의 `conf` 임계값을 기본값 `0.25`에서 `0.15`로 낮춰 더 민감한 감지가 가능하도록 설정했습니다.
    2.  **연속 프레임 검증**: 오탐지를 줄이기 위해, **동일 객체가 3프레임 이상 연속으로 감지될 때만 유효한 객체로 판단**하는 후처리 로직을 적용했습니다. 프레임 간 바운딩 박스의 **IoU (Intersection over Union)**를 계산하여 객체의 동일성을 판단합니다.

### 2. 성능 최적화 및 추적 안정성 확보

* **문제점**: 매 프레임마다 YOLO 추론과 복잡한 후처리 로직을 동시에 수행하자 CPU/GPU 부하가 증가하여 영상이 끊기는 성능 저하가 발생했습니다.
* **해결 과정**:
    1.  **DeepSORT 도입**: 강력한 객체 추적 라이브러리인 DeepSORT를 통합하여, YOLO가 감지한 객체에 **고유 ID를 부여**하고 프레임 간 움직임을 예측하며 안정적으로 추적하도록 구현했습니다. 이를 통해 직접 구현했던 복잡한 후처리 로직을 대체하고 더 높은 안정성을 확보했습니다.
    2.  **추적기 업데이트 최적화**: DeepSORT의 연산 부하를 줄이기 위해, 모든 프레임에서 추적기를 업데이트하는 대신 **2프레임마다 한 번씩 업데이트**하도록 코드를 수정했습니다. 이 최적화를 통해 실시간성을 확보하면서도 추적 정확도를 유지할 수 있었습니다.

---

## 🌱 향후 개선 계획 (Future Plans)

- [ ] **모델 추론 가속화**: TensorRT와 같은 하드웨어 가속 라이브러리를 적용하여 모델의 추론 속도를 최적화하고, 저사양 임베디드 장치에서도 원활하게 동작하도록 개선
- [ ] **실시간 알림 시스템 통합**: 화재 감지 시 SMS, 이메일, 또는 메신저 앱으로 실시간 알림을 전송하는 기능을 추가하여 실제 경보 시스템으로 확장
- [ ] **데이터셋 확장 및 모델 강건성 확보**: 야간, 우천, 안개 등 다양한 악조건 환경의 데이터를 추가로 수집하고 학습하여 모델의 **견고성(Robustness)**을 향상

---

## 📜 라이선스 (License)

이 프로젝트는 [MIT 라이선스](LICENSE)에 따라 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참고하십시오.
