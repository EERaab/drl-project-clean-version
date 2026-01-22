import numpy as np
wmap_base = np.zeros((49,49), dtype = np.int8)
wmap_base[3:33, 41:43] = 1
wmap_base[3:9, 31:43] = 1
wmap_base[6:8, 9:40] = 1
wmap_base[:15, 3:16] = 1
wmap_base[15:30, 9:10] = 1
wmap_base[28:35, 3:15] = 1
wmap_base[28:30, 14:29] = 1
wmap_base[22:33, 25:29] = 1
wmap_base[26:28, 29:45] = 1
wmap_base[30:46, 9:10] = 1
wmap_base[40:46, 2:12] = 1
wmap_base[33:40, 25:45] =1
wmap_base[23:28, 23:28] = 1

wmap_base -= 1
wmap_base *= -1