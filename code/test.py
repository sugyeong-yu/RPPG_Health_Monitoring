import time

print(time.time())

past=time.time()
while(True):
    now=time.time()
    times=now-past
    print(times)
    if times>3:
        break