# Time:  O(nlogn)
# Space: O(1)
#
# Given a collection of intervals, merge all overlapping intervals.
#
# For example,
# Given [1,3],[2,6],[8,10],[15,18],
# return [1,6],[8,10],[15,18].

class Solution:
    def merge(self, intervals):
        # intervals.sort(key=lambda x: x.start)
        intervals.sort(key=lambda x: x[0])

        merged = []
        for interval in intervals:
            # if the list of merged intervals is empty or if the current
            # interval does not overlap with the previous, simply append it.
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
            # otherwise, there is overlap, so we merge the current and previous
            # intervals.
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged

# print(Solution().merge([Interval(1, 3), Interval(2, 6), Interval(8, 10), Interval(15,18)]))
print(Solution().merge([[1, 3], [2, 6], [8, 10], [15,18]]))

