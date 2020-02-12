# BFS
def countComponents(n, edges):
    g = {x:[] for x in range(n)}
    for x, y in edges:
        g[x].append(y)
        g[y].append(x)
    ret = 0
    for i in range(n):
        queue = [i]

        ret += 1 if i in g else 0
        for j in queue:
            if j in g:
                queue += g[j]
                print(queue)
                del g[j]
    return ret
print(countComponents(5,[[0, 1], [1,2],[3,4]]))

