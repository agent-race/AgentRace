from agentscope.service import ServiceResponse, ServiceExecStatus
from typing import List
from functools import wraps


def twoSum(nums: List[int], target: int) -> ServiceResponse:
    """
    Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
    Args:
        nums (List): an array of integers
        target (Int): an integer target
    Returns:
        List[int]: indices of the two numbers such that they add up to target.
    """
    try:
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if nums[i] + nums[j] == target:
                    return ServiceResponse(status=ServiceExecStatus.SUCCESS,content=str([i, j]))
    
        return ServiceResponse(status=ServiceExecStatus.SUCCESS,content=str([]))
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))



def lengthOfLongestSubstring(s: str) -> ServiceResponse:
    """
    Given a string s, find the length of the longest substring without duplicate characters.
    Arg:
        s (String): a string

    Returns:
        Int: the length of the longest substring without duplicate characters.
    """
    try:
        left = 0
        right = 0
        max_len = 0

        while right < len(s):
            if s[right] in s[left:right]:
                max_len = max(max_len, right-left)
                left = s.index(s[right], left, right)+1
            max_len = max(max_len, right-left+1)
            right += 1
        return ServiceResponse(ServiceExecStatus.SUCCESS, str(max_len))
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))



def findMedianSortedArrays(nums1: List[int], nums2: List[int]) -> ServiceResponse:
    """
    Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.
    Args: 
        nums1 (List[int]): sorted array 1
        nums2 (List[int]): sorted array 2
    Returns:
        float: the median of the two sorted arrays
    """
    try:
        m, n = len(nums1), len(nums2)

        def kth_small(k):
            i = j = 0
            while True:
                if i == m:
                    return ServiceResponse(ServiceExecStatus.SUCCESS, str(nums2[j + k - 1]))
                if j == n:
                    return ServiceResponse(ServiceExecStatus.SUCCESS, str(nums1[i + k - 1]))
                if k == 1:
                    return ServiceResponse(ServiceExecStatus.SUCCESS, str(min(nums1[i], nums2[j])))
                pivot_i = min(i + (k >> 1) - 1, m - 1)
                pivot_j = min(j + (k >> 1) - 1, n - 1)
                if nums1[pivot_i] < nums2[pivot_j]:
                    k -= pivot_i + 1 - i
                    i = pivot_i + 1
                else:
                    k -= pivot_j + 1 - j
                    j = pivot_j + 1
        return ServiceResponse(ServiceExecStatus.SUCCESS, str(
            kth_small((m + n + 1 >> 1))
            if m + n & 1
            else (kth_small((m + n >> 1) + 1) + kth_small((m + n >> 1)))
            * 0.5
        ))
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))


def longestPalindrome(s: str) -> ServiceResponse:   
    """
    Given a string s, return the longest palindromic substring in s.
    Args: 
        s (String): a string
    Returns:
        str: the longest palindromic substring in s.
    """
    try:
        n = len(s)
        max_len = 0
        for i in range(2 * n + 1):
            if i % 2 == 0:
                left, right = i // 2, i // 2
            else:
                left, right = i // 2, i // 2 + 1
            while left >= 0 and right < n and s[left] == s[right]:
                left -= 1
                right += 1
            if right - left - 1 > max_len:
                max_len = right - left - 1
                result = s[left + 1: left + max_len + 1]
        return ServiceResponse(ServiceExecStatus.SUCCESS, str(result))
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))
    
    


def convertZ(s: str, numRows: int) -> ServiceResponse:
    """
    The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)
    P   A   H   N
    A P L S I I G
    Y   I   R
    And then read line by line: "PAHNAPLSIIGYIR"
    Args: 
        s (Str): a string
        numRows (Int): the number of rows of zigzag pattern 
    Returns:
        Str: zigzag string of the string s 
    """
    try:
        n = numRows
        if n == 1:
            return ServiceResponse(ServiceExecStatus.SUCCESS, str(s))
        res = [''] * n 
        
        sign = -1
        i = 0 
        for chr in s:
            res[i] += chr
            if i == 0 or i == n-1:
                sign = -sign
            i += sign

        return ServiceResponse(ServiceExecStatus.SUCCESS, ''.join(res))
    
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))



def reverseX(x: int) -> ServiceResponse:
    """
    Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go outside the signed 32-bit integer range [-231, 231 - 1], then return 0.
    Args:
        x (Int): a signed 32-bit integer x
    Returns:
        Int: the reversed digits of x
    """
    try:
        y, res = abs(x), 0
        of = (1 << 31) - 1 if x > 0 else 1 << 31
        while y != 0:
            res = res * 10 + y % 10
            if res > of: return ServiceResponse(ServiceExecStatus.SUCCESS, str(0))
            y //= 10
        return ServiceResponse(ServiceExecStatus.SUCCESS, str(res)) if x > 0 else ServiceResponse(ServiceExecStatus.SUCCESS, str(-res))
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))



def myAtoi(str: str) -> ServiceResponse:
    """
    The algorithm for myAtoi(string s) is as follows:
    1. Whitespace: Ignore any leading whitespace (" ").
    2. Signedness: Determine the sign by checking if the next character is '-' or '+', assuming positivity if neither present.
    3. Conversion: Read the integer by skipping leading zeros until a non-digit character is encountered or the end of the string is reached. If no digits were read, then the result is 0.
    4. Rounding: If the integer is out of the 32-bit signed integer range [-231, 231 - 1], then round the integer to remain in the range. Specifically, integers less than -231 should be rounded to -231, and integers greater than 231 - 1 should be rounded to 231 - 1.
    Return the integer as the final result.
    Args: 
        s (String): a string
    Returns:
        Int: the final result of myAtoi(string s)
    """
    try:
        import re
        matches = re.match('[ ]*([+-]?\d+)', str)
        if matches:
            res = int(matches.group(1))
            if res > (MAX := 2 ** 31 - 1):
                return ServiceResponse(ServiceExecStatus.SUCCESS, str(MAX))
            elif res < (MIN := -2 ** 31):
                return ServiceResponse(ServiceExecStatus.SUCCESS, str(MIN))
            else:
                return ServiceResponse(ServiceExecStatus.SUCCESS, str(res))
        else:
            return ServiceResponse(ServiceExecStatus.SUCCESS, str(0))
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))
    


def isPalindrome(x: int) -> ServiceResponse:
    """
    Given an integer x, return true if x is a palindrome, and false otherwise. For example, 121 is a palindrome and 123 is not a palindrome.
    Args:
        x (Int): an integer x
    Returns:
        Bool: true if x is a palindrome, and false otherwise.
    """
    try:
        if x < 0 or x > 0 and x % 10 == 0:
            return ServiceResponse(ServiceExecStatus.SUCCESS, str(False))
        rev = 0
        while rev < x // 10:
            rev = rev * 10 + x % 10
            x //= 10
        return ServiceResponse(ServiceExecStatus.SUCCESS, str(rev == x or rev == x // 10))
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))



def isMatch(s: str, p: str) -> ServiceResponse:
    """
    Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where:
    '.' Matches any single character.​​​​
    '*' Matches zero or more of the preceding element.
    Args: 
        s (String): an input string
        p (String): a pattern p
    Returns:
        Bool: true if the string s is match the pattern p
    """
    try:
        m, n = len(s) + 1, len(p) + 1
        dp = [[False] * n for _ in range(m)]
        dp[0][0] = True
        for j in range(2, n, 2):
            dp[0][j] = dp[0][j - 2] and p[j - 1] == '*'
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i][j - 2] or dp[i - 1][j] and (s[i - 1] == p[j - 2] or p[j - 2] == '.') \
                            if p[j - 1] == '*' else \
                            dp[i - 1][j - 1] and (p[j - 1] == '.' or s[i - 1] == p[j - 1])
        return ServiceResponse(ServiceExecStatus.SUCCESS, str(dp[-1][-1]))
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))



def maxArea(height: List[int]) -> ServiceResponse:
    """
    Given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).
    Find two lines that together with the x-axis form a container, such that the container contains the most water.
    Return the maximum amount of water a container can store.
    Args: 
        height (List[int]): an integer array height.
    Returns:
        Int: the maximum amount of water a container can store.
    """
    try:
        i, j, res = 0, len(height) - 1, 0
        while i < j:
            if height[i] < height[j]:
                res = max(res, height[i] * (j - i))
                i += 1
            else:
                res = max(res, height[j] * (j - i))
                j -= 1
        return ServiceResponse(ServiceExecStatus.SUCCESS, str(res))
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))



def longestCommonPrefix(strs: List[str]) -> ServiceResponse:
    """
    A function that finds the longest common prefix string amongst an array of strings.
    Args:
        strs (List[str]): an array of strings.
    Returns: 
        str: the longest common prefix string amongst an array of strings.
    """
    try:
        s0 = strs[0]
        for j, c in enumerate(s0):
            for s in strs:
                if j == len(s) or s[j] != c:
                    return ServiceResponse(ServiceExecStatus.SUCCESS, str(s0[:j]))
        return ServiceResponse(ServiceExecStatus.SUCCESS, str(s0))
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))



def threeSum(nums: List[int]) -> ServiceResponse:
    """
    Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.
    Args:
        nums (List[int]): an integer array nums
    Returns:
        List[List[int]]: return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0
    """
    try:
        n=len(nums)
        res=[]
        if(not nums or n<3):
            return ServiceResponse(ServiceExecStatus.SUCCESS, str([]))
        nums.sort()
        res=[]
        for i in range(n):
            if(nums[i]>0):
                return ServiceResponse(ServiceExecStatus.SUCCESS, str(res))
            if(i>0 and nums[i]==nums[i-1]):
                continue
            L=i+1
            R=n-1
            while(L<R):
                if(nums[i]+nums[L]+nums[R]==0):
                    res.append([nums[i],nums[L],nums[R]])
                    while(L<R and nums[L]==nums[L+1]):
                        L=L+1
                    while(L<R and nums[R]==nums[R-1]):
                        R=R-1
                    L=L+1
                    R=R-1
                elif(nums[i]+nums[L]+nums[R]>0):
                    R=R-1
                else:
                    L=L+1
        return ServiceResponse(ServiceExecStatus.SUCCESS, str(res))
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))



def isValidBrackets(s: str) -> ServiceResponse:
    """
    Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
    An input string is valid if:
    1. Open brackets must be closed by the same type of brackets.
    2. Open brackets must be closed in the correct order.
    3. Every close bracket has a corresponding open bracket of the same type.
    Args:
        s (str): a string s containing just the characters '(', ')', '{', '}', '[' and ']'
    Returns:
        Bool: true if the input string is valid
    """
    try:
        dic = {')':'(',']':'[','}':'{'}
        stack = []
        for i in s:
            if stack and i in dic:
                if stack[-1] == dic[i]: stack.pop()
                else: return ServiceResponse(ServiceExecStatus.SUCCESS, str(False))
            else: stack.append(i)
            
        return ServiceResponse(ServiceExecStatus.SUCCESS, str(not stack))
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))



def generateParenthesis(n: int) -> ServiceResponse:
    """
    Given n pairs of parentheses, generate all combinations of well-formed parentheses.
    Args:
        n (int): n pairs of parentheses
    Returns:
        List[str]: all combinations of well-formed parentheses.
    """
    try:
        ans = []
        def backtrack(S = '', left = 0, right = 0):
            if len(S) == 2 * n:
                ans.append(S)
                return ServiceResponse(ServiceExecStatus.SUCCESS, str(None))
            if left < n:
                backtrack(S+'(', left+1, right)
            if right < left:
                backtrack(S+')', left, right+1)
        backtrack()
        return ServiceResponse(ServiceExecStatus.SUCCESS, str(ans))
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))



def groupAnagrams(strs: List[str]) -> ServiceResponse:
    """
    Given an array of strings strs, group the anagrams together. Return the answer in any order.
    Args:
        strs (List[str]): an array of strings strs
    Returns:
        List[List[str]]: anagrams
    """
    try:
        import collections
        mp = collections.defaultdict(list)
        for st in strs:
            key = "".join(sorted(st))
            mp[key].append(st)
        return ServiceResponse(ServiceExecStatus.SUCCESS, str(list(mp.values()) ))
    
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))




def lengthOfLastWord(s: str) -> ServiceResponse:
    """
    Given a string s consisting of words and spaces, return the length of the last word in the string.
    Args:
        s (Str): a string s consisting of words and spaces
    Returns:
        Int: the length of the last word in the string.
    """
    try:
        i = len(s) - 1
        while s[i] == ' ':
            i -= 1
        j = i - 1
        while j >= 0 and s[j] != ' ':
            j -= 1
        return ServiceResponse(ServiceExecStatus.SUCCESS, str(i - j))
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))



def addBinary(a: str, b: str) -> ServiceResponse:
    """
    Given two binary strings a and b, return their sum as a binary string.
    Args:
        a (Str): binary string 1
        b (Str): binary string 2
    Returns:
        Str: the sum of a and b (a binary string)
    """
    try:
        a_ = int(a,2)
        b_ = int(b,2)
        ans = bin(a_ + b_)
        return ServiceResponse(ServiceExecStatus.SUCCESS, str(ans[2:]))
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))


def minDistance(word1: str, word2: str) -> ServiceResponse:
    """
    Given two strings word1 and word2, return the minimum number of operations (Insert a character, Delete a character and Replace a character) required to convert word1 to word2.
    Args:
        word1 (Str): strings word 1
        word2 (Str): strings word 2
    Returns:
        Int: the minimum number of operations (Insert a character, Delete a character and Replace a character) required to convert word1 to word2.
    """
    try:
        length1, length2 = len(word1), len(word2)
        dp = [[0 for i in range(length2 + 1)] for j in range(length1 + 1)]
        i, j =0, 0
        word_1, word_2 = ' ' + word1, ' ' + word2
        for k in range(length1 + 1):
            dp[k][0] = k
        for k in range(length2 + 1):    
            dp[0][k] = k

        for i in range(1, length1+1):
            for j in range(1, length2+1):
                if word_1[i] == word_2[j]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min([dp[i-1][j], dp[i][j-1], dp[i-1][j-1]]) + 1
        return ServiceResponse(ServiceExecStatus.SUCCESS, str(dp[length1][length2]))
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))



def largestNumber(nums: List[int]) -> ServiceResponse:
    """
    Given a list of non-negative integers nums, arrange them such that they form the largest number and return it.
    Args:
        nums (List[int]): Given a list of non-negative integers nums
    Returns:
        str: Arranged nums such that form the largest number
    """
    try:
        import math
        def fun(x):
            if x==0:return ServiceResponse(ServiceExecStatus.SUCCESS, str(0))
            L=int(math.log10(x))+1
            return ServiceResponse(ServiceExecStatus.SUCCESS, str(x/(10**L-1)))
        nums.sort(key=fun,reverse=True)
        nums=list(map(str,nums))
        return ServiceResponse(ServiceExecStatus.SUCCESS, str(str(int("".join(nums)))))
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))



def reverseString(s: List[str]) -> ServiceResponse:
    """
    A function that reverses a string. The input string is given as an array of characters s.
    Args:
        s (List[str]): a string as an array of characters s
    Returns:
        Str: the reversed string
    """
    try:
        for i in range(len(s) // 2):
            s[i], s[-i - 1] = s[-i - 1], s[i]
        return ServiceResponse(ServiceExecStatus.SUCCESS, str(s))
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))
