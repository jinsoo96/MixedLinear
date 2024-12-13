MixedLinear: 효율적인 전력 수요 예측을 위한 DLinear와 TimeMixer 기반 앙상블 모델
MixedLinear는 전력 수요 예측을 위한 혁신적인 앙상블 모델로, DLinear와 TimeMixer 컴포넌트를 결합하여 높은 정확도와 계산 효율성을 달성합니다.
주요 특징
효율적인 계산: 복잡한 딥러닝 모델에 비해 적은 계산 자원으로 높은 예측 정확도 달성
시계열 데이터 처리: 계절성과 장기 트렌드를 효과적으로 포착
우수한 성능: 3년 입력 데이터로 MSE 0.1642, MAE 0.2474 달성 (DLinear와 TimeMixer 대비 개선)
모델 구조
DLinear 컴포넌트
시계열 데이터의 트렌드와 계절성을 독립적으로 학습
단일 선형 레이어를 사용하여 메모리와 계산 효율성 향상
입력 시퀀스 길이 L에 대해 선형 복잡도 O(L) 유지
TimeMixer 컴포넌트
다양한 스케일의 시계열 데이터를 트렌드와 계절 요소로 분해 및 혼합
복잡한 시계열 패턴 처리에 강점
응용 분야
전력 시스템 운영 안정화
재생 에너지 시스템 개발
탄소 중립 에너지 부문으로의 전환 지원
MixedLinear는 DLinear의 효율성과 TimeMixer의 복잡한 패턴 처리 능력을 결합하여, 다양한 시계열 데이터에 대해 우수한 예측 성능을 제공합니다.
