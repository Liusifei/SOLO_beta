class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if len(s) == 0:
            return True
        if len(s) == 1:
            return False
        left = []
        right = []
        for i in range(len(s)):
            c = s[i]
            if c == '(' or c == '{' or c == '[':
                left.append(c)
            if c == ')':
                if len(left) == 0 or left[-1] != '(':
                    return False
                else:
                    left.pop()
            if c == '}':
                if len(left) == 0 or left[-1] != '{':
                    return False
                else:
                    left.pop()
            if c == ']' :
                if len(left) == 0 or left[-1] != '[':
                    return False
                else:
                    left.pop()
        if len(left) == 0:
            return True
        else:
            return False