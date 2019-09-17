import time

s = time.strftime('%Y/%m/%d-%H:%M:%S.log')
print(s)

for _ in range(10):
        time.sleep(0.5)
        print(s)