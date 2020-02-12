#
#
def minRemoveToMakeValid(s):
    """
    :type s: str
    :rtype: str
    """
    opens, closes = [], []
    for i in range(len(s)):
        c = s[i]
        if c == '(':
            opens.append(i)
        elif c == ')':
            if len(opens) == 0:
                closes.append(i)
            else:
                opens.pop()
    hs = set(opens+closes)
    return ''.join(s[i] for i in range(len(s)) if i not in hs)

print(minRemoveToMakeValid("lee(t(c)o)de)"))
