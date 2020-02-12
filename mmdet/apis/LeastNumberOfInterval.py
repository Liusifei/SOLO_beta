
[[0,3],[2,7],[3,4],[4,6]]
[0,6]

def find_min_intervals(intervals, target):
    intervals.sort()
    res = 0
    cur_target = target[0]
    i = 0
    max_step = 0
    while i < len(intervals) and cur_target < target[1]:
        while i < len(intervals) and intervals[i][0] <= cur_target:
            max_step = max(max_step, intervals[i][1])
            i += 1
        cur_target = max_step
        res += 1
    return res if cur_target >= target[1] else 0


print(find_min_intervals([[0, 3], [3, 4], [4, 6], [2, 7]], [0,6]))
