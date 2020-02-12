a = [(1,2),(2,3),(100,5)]
b = [(0,5),(1,1),(100,6)]
i = 0; j = 0
result = 0
while i < len(a) and j < len(b):
    if a[i][0] == b[j][0]:
        result += a[i][1] * b[j][1]
        i += 1
        j += 1
    elif a[i][0] < b[j][0]:
        i += 1
    else:
        j += 1
print(result)


a = [(1,2),(2,3),(100,5)]
b = [(0,5),(1,1),(100,6)]

result = 0
i = 0
j = 0

while i < len(a) and j < len(b):
    if a[i][0] == b[j][0]:
        result = result + a[i][1]*b[j][1]
        i = i + 1
        j = j + 1
    elif a[i][0] < b[j][0]:
        i = i + 1
    else:
        j = j + 1
print(result)