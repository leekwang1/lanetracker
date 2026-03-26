# Lane Tracker V2

V1을 유지한 채 별도 패키지로 만든 1차 완성형 V2입니다.

포함 기능:
- profile 기반 + oriented strip 기반 차선 추적
- multi-hypothesis branching
- dashed / solid 분류
- gap bridging
- crosswalk stop
- CLI 실행
- GUI 디버거 골격 (PySide6 + pyqtgraph)

## CLI
```bash
python -m lane_tracker_v2.app.cli_v2 --las input.las --p0 X0 Y0 Z0 --p1 X1 Y1 Z1 --output out.csv
```

## GUI
```bash
python -m lane_tracker_v2.app.app_main
```

## 주의
이 버전은 통합 1차 완성본입니다. 구조는 완성형으로 잡혀 있지만, 데이터별 intensity/폭/점선 주기 차이 때문에 파라미터 튜닝은 추가로 필요할 수 있습니다.
