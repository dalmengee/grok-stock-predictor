# 주가 예측 프로그램

머신러닝과 기술적 지표를 활용한 주가 예측 도구입니다. Yahoo Finance 데이터를 기반으로 미래 주가를 예측합니다.

## 기능

- **한국 주식**: 키움 로컬 DB (`daily_quote.sqlite3`, `master.sqlite3`)에서 일봉 수집
- **해외 주식**: Yahoo Finance에서 OHLCV 데이터 수집
- RSI, MACD, 볼린저밴드 등 14개 기술적 지표
- Random Forest / Gradient Boosting 모델
- 미국·한국 주식 지원 (`AAPL`, `005930.KS` 등)
- **오늘의 한국 주식 스크리너** — 코스피/코스닥 종목을 분석해 매수 후보 순위 제공
- **투자 의사결정 도구** — 종목 분석, 수급/리스크, 손절·목표가, 포트폴리오, 일일 브리핑
- CLI 및 Streamlit 웹 UI

## 설치

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

한국 주식 데이터는 기본적으로 아래 경로의 DB를 **읽기 전용**으로 사용합니다 (DB 파일 수정 없음):

```
/Users/dalmengee/claude_kiwoom_20260521/data/db/
  ├── daily_quote.sqlite3   # 일봉
  └── master.sqlite3        # 종목 마스터
```

다른 경로를 쓰려면 환경변수를 설정하세요:

```bash
export KIWOOM_DB_DIR=/path/to/data/db
```

## 사용법

### CLI

```bash
python predict.py AAPL
python predict.py 005930 --days 10
python predict.py TSLA --model gradient_boosting --chart chart.png
```

### 투자 의사결정

```bash
python invest.py analyze 005930          # 종목 투자 분석 (진입/손절/목표가)
python invest.py brief                   # 오늘의 투자 브리핑
python invest.py portfolio show          # 포트폴리오 조회
python invest.py portfolio add 005930 10 320000  # 종목 추가
streamlit run invest_app.py              # 투자 대시보드
```

### 오늘의 한국 주식 (스크리너)

```bash
python kr_pick.py
python kr_pick.py --universe all --top 10
streamlit run kr_app.py
```

점수 기준 (100점): ML 예측 30 + 추세 25 + 모멘텀 20 + 거래량 15 + 단기 강도 10

### 웹 UI

```bash
streamlit run app.py      # 단일 종목 예측
streamlit run kr_app.py   # 한국 주식 스크리너
```

## 프로젝트 구조

```
├── predict.py          # 단일 종목 예측 CLI
├── invest.py           # 투자 의사결정 CLI
├── invest_app.py       # 투자 대시보드 웹 앱
├── kr_pick.py          # 한국 주식 스크리너 CLI
├── app.py              # 단일 종목 예측 웹 앱
├── kr_app.py           # 한국 주식 스크리너 웹 앱
├── data/               # 포트폴리오 (portfolio.json, gitignore)
├── requirements.txt
└── src/
    ├── data_fetcher.py # 데이터 수집 (한국=로컬DB, 해외=Yahoo)
    ├── local_db.py     # 키움 SQLite DB 리더
    ├── features.py     # 기술적 지표
    ├── model.py        # ML 모델
    ├── invest_analysis.py  # 투자 분석 (수급/리스크/손절)
    ├── portfolio.py        # 포트폴리오 관리
    ├── market_brief.py     # 일일 투자 브리핑
    ├── screener.py     # 매수 스코어링
    ├── korean_universe.py  # 한국 종목 리스트
    └── visualizer.py   # 차트 시각화
```

## 주의사항

본 프로그램의 예측 결과는 참고용이며, 실제 투자 결정의 유일한 근거로 사용하지 마세요.