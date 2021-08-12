# _*_ coding: utf-8 _*_
"""
Time:     2021/8/10 22:03
Author:   WANG Bingchen
Version:  V 0.1
File:     leetcode.py
Describe: 
"""
import numpy as np

a = np.ones((5, 3, 2))
b = np.ones((3, 5, 2))
c = np.dot(a[:, :, 0], b[:, :, 0])