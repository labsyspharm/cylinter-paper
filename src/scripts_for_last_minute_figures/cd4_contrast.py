# original threshold settings
# CD4 = [2749.6, 21998.350000000002]
# CD45 = [1302.5, 20351.0]
# CD45RO = [1563.046, 46105.0]

# replace code for image contrast in CyLinter
# starting on line 6512 with this
cd4_dict = {0: [2049.0, 11998.0],
            1: [2749.0, 5998.0],
            2: [2749.0, 7098.0],
            3: [2049.0, 11998.0],
            4: [2749.0, 7998.0],
            5: [2749.0, 17998.0],
            6: [2749.0, 10098.0],
            7: [2749.0, 8998.0],
            8: [2749.0, 6998.0],
            9: [2749.0, 5998.0],
            10: [3549.0, 9098.0],
            11: [2849.0, 9998.0],
            12: [3549.0, 8098.0],
            13: [2749.0, 11998.0],
            14: [2749.0, 6998.0],
            15: [3549.0, 7598.0],
            16: [1449.0, 7998.0],
            17: [3149.0, 15998.0],
            18: [2249.0, 7998.0],
            19: [2049.0, 8998.0],
            20: [3049.0, 8998.0],
            21: [4049.0, 7998.0],
            22: [2049.0, 8998.0],
            23: [2049.0, 6998.0],
            24: [4049.0, 8998.0],
            }
cd45_dict = {0: [1302.0, 35535.0],
             1: [1302.0, 8351.0],
             2: [2802.0, 13351.0],
             3: [1302.0, 18351.0],
             4: [1802.0, 3051.0],
             5: [1302.0, 15351.0],
             6: [1302.0, 8351.0],
             7: [2802.0, 40351.0],
             8: [1502.0, 8351.0],
             9: [1302.0, 8351.0],
             10: [1502.0, 6351.0],
             11: [1302.0, 8051.0],
             12: [1202.0, 3551.0],
             13: [1302.0, 5351.0],
             14: [1002.0, 3351.0],
             15: [1202.0, 7051.0],
             16: [902.0, 1251.0],
             17: [1302.0, 20351.0],
             18: [1302.0, 3351.0],
             19: [1202.0, 4051.0],
             20: [1702.0, 5051.0],
             21: [3002.0, 4051.0],
             22: [3502.0, 7051.0],
             23: [2702.0, 10051.0],
             24: [2402.0, 21051.0],
             }
cd45ro_dict = {0: [1663.0, 46105.0],
               1: [463.0, 1570.0],
               2: [163.0, 15105.0],
               3: [1563.0, 55105.0],
               4: [1563.0, 9105.0],
               5: [1563.0, 46105.0],
               6: [1563.0, 43105.0],
               7: [1563.0, 30105.0],
               8: [1563.0, 30105.0],
               9: [1563.0, 19105.0],
               10: [1563.0, 44105.0],
               11: [1563.0, 46105.0],
               12: [963.0, 35105.0],
               13: [1563.0, 46105.0],
               14: [403.0, 13105.0],
               15: [1063.0, 25105.0],
               16: [403.0, 5105.0],
               17: [1563.0, 46105.0],
               18: [563.0, 7105.0],
               19: [703.0, 8105.0],
               20: [2003.0, 21105.0],
               21: [10203.0, 34105.0],
               22: [3803.0, 30105.0],
               23: [703.0, 15105.0],
               24: [1003.0, 50105.0],
               }

if marker == 'CD4_488':
    slice -= (cd4_dict[cell][0]/65535)
    slice /= (
        (cd4_dict[cell][1]/65535)
        - (cd4_dict[cell][0]/65535))
elif marker == 'CD45_PE':
    slice -= (cd45_dict[cell][0]/65535)
    slice /= (
        (cd45_dict[cell][1]/65535)
        - (cd45_dict[cell][0]/65535))
elif marker == 'anti_CD45RO':
    slice -= (cd45ro_dict[cell][0]/65535)
    slice /= (
        (cd45ro_dict[cell][1]/65535)
        - (cd45ro_dict[cell][0]/65535))