def distance(a, b):
    tmp = 0
    for i in range(len(a)):
        tmp += pow(a[i] - b[i], 2)
    return pow(tmp, 0.5)


p1 = (0, 0)
p2 = (1, 2)
p3 = (3, 1)
p4 = (8, 8)
p5 = (9, 10)
p6 = (10, 7)


c = (p3, p4, p5, p6)

for p in c:
    print('%.3f, %.3f' % (distance(p1, p), distance(p2, p)))




def f(x):
    return -2*pow(x, 2) + x

def F(x):
    return -2*pow(x, 2) + x +1600

def sel(x):
    tmp = sum(x)
    return [i/tmp for i in x]

xx = [19, 1, 7, 17]

fx = [f(x) for x in xx]
Fx = [F(x) for x in xx]
avg = sum(fx)/4
p = sel(Fx)
print(fx)
print(Fx)
print(avg)
print(p)
avg_F = sum(Fx)/4
print([i/avg_F for i in Fx])