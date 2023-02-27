import itertools


a = [i for i in range(10)]
b = [i**2 for i in range(11)]

for idx, batch in enumerate(itertools.zip_longest(a, b)):
    print(idx, batch)
