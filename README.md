# 주가 예측 프로그램

머신러닝과 기술적 지표를 활용한 주가 예측 도구입니다. Yahoo Finance 데이터를 기반으로 미래 주가를 예측합니다.

## 기능

- Yahoo Finance에서 OHLCV 데이터 자동 수집
- RSI, MACD, 볼린저밴드 등 14개 기술적 지표
- Random Forest / Gradient Boosting 모델
- 미국·한국 주식 지원 (`AAPL`, `005930.KS` 등)
- CLI 및 Streamlit 웹 UI

## 설치

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 사용법

### CLI

```bash
python predict.py AAPL
python predict.py 005930.KS --days 10
python predict.py TSLA --model gradient_boosting --chart chart.png
```

### 웹 UI

```bash
streamlit run app.py
```

## 프로젝트 구조

```
├── predict.py          # CLI 진입점
├── app.py              # Streamlit 웹 앱
├── requirements.txt
└── src/
    ├── data_fetcher.py # 데이터 수집
    ├── features.py     # 기술적 지표
    ├── model.py        # ML 모델
    └── visualizer.py   # 차트 시각화
```

## 주의사항

본 프로그램의 예측 결과는 참고용이며, 실제 투자 결정의 유일한 근거로 사용하지 마세요.