import others
import rPPG_chrom
import numpy as np
import matplotlib.pyplot as plt

import pickle
import time

# RGB 신호 생성
video_path = 'test_sg.mp4'

fs = 30
minBPM = 42
maxBPM = 240

## 실시간 rPPG모니터링 ##
rPPG_chrom.real_time_cv2(video_path, fs, minBPM, maxBPM)




'''
## rPPG신호 수집 ##
rgb_signal = rPPG_chrom.run(video_path)

# 주파수 스펙트럼 & CHROM 검출
f, chrom_spectrum, rppg = rPPG_chrom.CHROM(rgb_signal, fs, minBPM, maxBPM)

# 맥박수 추정
pr = rPPG_chrom.estimate_average_pulserate(rppg, fs, minBPM, maxBPM)
print('평균 맥박수: %d'%pr)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(rppg, 'b', linewidth=2)
plt.xlim([0, len(rppg)])
plt.show()
others.ppg_save('rppg.csv', rppg)
'''



'''
## rPPG신호 복원 ##
model_sd = others.model_load('model_rppg/best_model_elu_sd_cubic.h5')
model_svr = pickle.load(open('model_rppg/SVR_cubic.pkl', 'rb'))

rppg = others.ppg_load('rppg.csv')

rppg_cycle = others.single_cycle_unit_rppg(rppg[17:-17])
rppg30 = others.feature(rppg_cycle)

train_rppg_svr = model_svr.predict(np.reshape(rppg30, (-1, 30)))
train_rppg_sd = model_sd.predict(np.reshape(train_rppg_svr, (-1, 30)))
restore_rppg = []

for i in train_rppg_sd:
    restore_rppg.extend(i)

# restore_rppg1 = []
# for i in range(len(train_rppg_sd)-1):
#     if restore_rppg1:
#         restore_rppg1.extend(train_rppg_sd[i] + restore_rppg1[-1])
#     else:
#         restore_rppg1.extend(train_rppg_sd[i])
'''


'''
## 실시간 신호 업데이트 자동화 ##
x = np.linspace(0, 60, 61)
y = restore_rppg[:61]

plt.ion() # 대화식 모드를 켠다.

figure, ax = plt.subplots(figsize=(8, 6))
line, = ax.plot(x, y)

for i in range(len(restore_rppg)-61):
    updated_y = restore_rppg[i:i+61]
    line.set_xdata(x)
    line.set_ydata(updated_y)

    figure.canvas.draw()    # 그림을 표시하는 JavaScript 기반 방법

    figure.canvas.flush_events()   # 그림을 지우는 JavaScript 기반 방법
    time.sleep(0.1)
'''


