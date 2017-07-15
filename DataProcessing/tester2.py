x = 0.4
for i in range(10000000)
    x = 1 - 0.95*(1-x)
    if x >= 0:
        print(i)
        break