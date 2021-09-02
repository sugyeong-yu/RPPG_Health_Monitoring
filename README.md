# RPPG_Health_Monitoring
RPPG기술을 이용한 건강 모니터링 SW
- RGB카메라로 촬영된 얼굴 영상으로부터 Remote PPG 신호추출
- Contact PPG 신호 수준의 RPPG / Spo2 / HRV 제공 SW
- 이를 통해 사용자의 건강상태를 모니터링

> ### Remote PPG 
> : 카메라를 이용해 비접촉식으로 PPG를 측정할 수 있는 기술
> - [참고논문]()
> ### Resotred PPG
>  : CPPG의 quality 수준으로 RPPG신호를 복원하는 기술
>  - [참고논문]()
> ### Spo2
>  : RPPG를 이용해 산소포화도를 측정할 수 있는 기술
>  - [참고논문]()
> ### HRV
>  : RPPG를 이용해 심박변이도를 분석할 수 있는 기술
>  - [참고논문]()

## Monitoring SW UI
![image](https://user-images.githubusercontent.com/70633080/131788175-367f8797-e886-4eab-9779-367b8d109cd5.png)
- [사용법]()
- 환경구축
  - tensorflow 2.0 이상
  - scikit-learn 0.24.0
  - PyQt 5.15.4
  - opencv-python 4.5.1.48
  - heartpy 1.2.6
  - hrv-analysis 1.0.3 
