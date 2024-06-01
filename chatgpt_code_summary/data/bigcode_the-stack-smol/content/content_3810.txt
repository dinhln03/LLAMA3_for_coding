#!/usr/bin/env python
#-*- coding: UTF-8 -*-
###########################################################################
#
# Copyright (c) 2018 www.codingchen.com, Inc. All Rights Reserved
#
##########################################################################
'''
 @brief leetcode algorithm
 @author chenhui(hui.chen6789@gmail.com)
 @date 2018/11/07 21:30:33
'''

class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        odd = (len(nums1) + len(nums2)) % 2
        if odd:
            half = (len(nums1) + len(nums2)) // 2
        else:
            half = (len(nums1) + len(nums2)) // 2 - 1
        for _ in range(half):
            __ = self.pop_num(nums1, nums2)
        if odd:
            return float(self.pop_num(nums1, nums2))
        else:
            t1 = self.pop_num(nums1, nums2)
            t2 = self.pop_num(nums1, nums2)
            return (t1 + t2) / 2

    def pop_num(self, nums1, nums2):
        if len(nums1) == 0:
            return nums2.pop(0)
        elif len(nums2) == 0:
            return nums1.pop(0)
        elif nums1[0] > nums2[0]:
            return nums2.pop(0)
        elif nums1[0] <= nums2[0]:
            return nums1.pop(0)

if __name__ == '__main__':
    s = Solution()
    nums1 = [1, 2]
    nums2 = [3, 4]
    print(s.findMedianSortedArrays(nums1, nums2))