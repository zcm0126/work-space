import time

a = time.time()
b = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
c = time.localtime(a)
print(a)
print(b)
d = f"{c[0]}-{c[1]}-{c[2]}-{c[3]}-{c[4]}"
print(d)