# RPPG_Health_Monitoring
RPPG기술을 이용한 건강 모니터링 SW
- RGB카메라로 촬영된 얼굴 영상으로부터 Remote PPG 신호추출
- Contact PPG 신호 수준의 RPPG / Spo2 / HRV 제공 SW
- 이를 통해 사용자의 건강상태를 모니터링

> ### Remote PPG 
> : 카메라를 이용해 비접촉식으로 PPG를 측정할 수 있는 기술
> - [참고논문](https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-26/issue-6/063003/Contactless-physiological-signals-extraction-based-on-skin-color-magnification/10.1117/1.JEI.26.6.063003.short?SSO=1)
> ### Resotred PPG
>  : CPPG의 수준으로 RPPG신호를 복원하는 기술
>  - [참고논문](https://www.mdpi.com/1424-8220/21/17/5910)
> ### Spo2
>  : RPPG를 이용해 산소포화도를 측정할 수 있는 기술
>  - [참고논문]()
> ### HRV
>  : RPPG를 이용해 심박변이도를 분석할 수 있는 기술
>  - [참고논문]()

## Monitoring SW UI
- [Usage Scenarios](https://github.com/sugyeong-yu/RPPG_Health_Monitoring/blob/main/%EB%B0%9C%ED%91%9C%20%EC%9E%90%EB%A3%8C.pptx)
- 환경구축
  - python
  - tensorflow 2.0 이상
  - scikit-learn 0.24.0
  - PyQt 5.15.4
  - opencv-python 4.5.1.48
  - heartpy 1.2.6
  - hrv-analysis 1.0.3 
- code
  - [analysis_functions.py](https://github.com/sugyeong-yu/RPPG_Health_Monitoring/blob/main/code/analysis_functions.py) : RPPG/resotred/spo2/hrv code
  - [functions.py](https://github.com/sugyeong-yu/RPPG_Health_Monitoring/blob/main/code/functions.py) : analysis_functions에 필요한 sub functions
  - [get_rgb_function.py](https://github.com/sugyeong-yu/RPPG_Health_Monitoring/blob/main/code/get_rgb_function.py) : RPPG 측정에 필요한 얼굴 영역으로 부터 rgb 성분 추출
  - [monitoring_5.ui](https://github.com/sugyeong-yu/RPPG_Health_Monitoring/blob/main/code/monitoring_5.ui) : SW ui 파일
  - [result_dialog.py](https://github.com/sugyeong-yu/RPPG_Health_Monitoring/blob/main/code/result_dialog.py) : health analysis 완료 후 분석결과 팝업 창 생성 ui
  - [UI_9.py](https://github.com/sugyeong-yu/RPPG_Health_Monitoring/blob/main/code/UI_9.py) : main file, UI 동작/thread/button event 
