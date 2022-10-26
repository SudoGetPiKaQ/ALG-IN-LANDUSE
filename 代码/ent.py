import numpy as np
import pandas as pd
import math
from functools import reduce


def ent(x):
    '''
    :param x: 其中输入的x为一列数据
    :return: 该列数据的信息熵
    '''
    x = pd.Series(x)
    # 定义初始信息熵ent1
    ent1 = 0
    p = x.value_counts() / len(x)
    for i in range(len(p)):
        e = -p.iloc[i] * math.log(p.iloc[i])
        ent1 = ent1 + e
    return ent1


def cutIndex(x, y):
    '''
    :param x: 其中输入的x为一列数据
    :param y: 标签数据
    :return: 该列数据的切点以及划分后的信息熵
    '''
    n = len(x)
    x = pd.Series(x)
    # 初始化一个熵的值
    entropy = 9999
    cutD = None
    # 寻找最佳分裂点
    for i in range(n - 1):
        if (x.iloc[i + 1] != x.iloc[i]):
            # cutX=(x.iloc[i+1]+x.iloc[i])/2
            wCutX = x[x < x.iloc[i + 1]]
            wn = len(wCutX) / n
            # 左边权重wn
            e1 = wn * ent(y[:len(wCutX)])
            # 右边权重
            e2 = (1 - wn) * ent(y[len(wCutX):])
            # 权重总和的信息熵
            val = e1 + e2
            if val < entropy:
                entropy = val
                cutD = i
    if cutD == None:
        return (None)
    else:
        # 返回切点,最小信息熵
        return (cutD, entropy)


# 停止切分函数
def cutStop(cutD, y, entropy):
    '''
    :param cutD: 切点
    :param y: 标签列
    :param entropy:信息熵
    :return: 信息熵的变化
    '''
    n = len(y)
    es = ent(y)  # 总信息熵
    gain = es - entropy  # 信息熵的变化
    left = len(set(y[0:cutD]))
    right = len(set(y[cutD:]))
    lengthY = len(set(y))
    if (cutD == None or lengthY == 0):
        return (None)
    else:
        # math.log(3^2-2) 也就是math.log(7)
        delta = math.log(3 ** lengthY - 2) - (lengthY * ent(y) - left * ent(y[0:cutD]) - right * ent(y[cutD:]))
        cond = math.log(n - 1) / n + delta / n
        # 大约选择变化小于总体信息熵的20%(例如:总体熵为1,当切点且分后变化小于0.2的时候不需要进行切分)
        if (gain < cond):
            return (None)
        else:
            return (gain)


# 返回切点的位置
def cutPoints(x, y):
    '''
    :param x: 输入带切分的列数据
    :param y: 输入标签数据
    :return: 返回切点
    '''
    dx = x.sort_values()
    dy = pd.Series(y, index=dx.index)  # 按照X的排序更新y的值
    depth = 0

    def gr(low, upp, depth=depth):
        x = dx[low:upp]
        y = dy[low:upp]
        n = len(y)
        k = cutIndex(x, y)
        if k == None:
            return (None)  # 判断是否存在切点 加权求和
        else:
            cutD = k[0]  # 切点位置
            entropy = k[1]  # 信息熵
            gain = cutStop(cutD, y, entropy)
            if gain == None:
                return (None)  # 判断是否应该切分(基于熵的变化)
            else:
                return [cutD, depth + 1]

    # 递归函数返
    def part(low=0, upp=len(dx), cutTd1=[], depth=depth):
        x1 = dx[low:upp]
        y1 = dy[low:upp]
        n = len(x1)
        # 返回的是切点 与depth+1
        k = gr(low, upp, depth=depth)
        if (n < 2 or k == None):
            return cutTd1
        else:
            cutX = k[0]
            depth = depth + 1
            cutTd1.append(low + cutX)
            # cutTd1.extend([low+cutX])
            # 从小到大
            cutTd1.sort()
            return part(low, low + cutX, cutTd1, depth) + part(cutX + low, upp, cutTd1, depth)

    res1 = part(low=0, upp=len(dx), cutTd1=[], depth=depth)
    cutDx = []
    if res1 == []:
        return (None)
    # 去重
    func = lambda x, y: x if y in x else x + [y]
    res = reduce(func, [[], ] + res1)
    res = pd.Series(res)
    # 返回切点的label 对应的实际值
    for i in res.values:
        k = round((x.sort_values().values[i] + x.sort_values().values[i + 1]) / 2, 6)
        cutDx.append(k)
    return (cutDx)


def Mdlp(data):
    '''
    :param data: 输入的是df 需要带有label字段
    :return: [切点] [min,切点,max]
    '''
    p = data.shape[1] - 1
    y = data.iloc[:, p]
    xd = data
    cutP = []
    cutPs = []
    for i in range(0, p):
        print(i)
        x = data.iloc[:, i]
        cuts1 = cutPoints(x, y)
        if cuts1 == None:
            cuts1 = "ALL"
        cuts = [[min(x)], cuts1, [max(x)]]
        cutPs.append(cuts)
        cutP.append(cuts1)
        print(cuts1)
        # label=range(1,len(cuts))
        # xd.ix[:,i]=pd.cut(x,bins=cutpc,labels=label,include_lowest=True)
    return cutP, cutPs


def Mdlpx(data):
    '''
    :param data: 输入的是df 需要带有label字段
    :return:  {column_name:切点} {column_name:min,切点,max}
    '''
    cut, cutAll = Mdlp(data)
    zidian = {}
    zidian_all = {}
    for i in range(data.shape[1] - 1):
        zidian[data.columns[i]] = cut[i]
        zidian_all[data.columns[i]] = cutAll[i]
    return zidian, zidian_all


data = pd.read_csv(r'../数据集/landUse_normalize.csv', index_col=False)
# label=data['class'].values.flatten()
# dem=data.iloc[:, 1:2].values.flatten()

print(Mdlp(data))

'''
[[1.6e-05, 0.023404], [0.16327], [0.00327, 0.024938, 0.074824], [0.057448], [0.0127, 0.093862], [0.103024],
        [0.042742, 0.165894], [0.182247], [0.089934, 0.178186], [0.351203], [0.015394, 0.041381],
        [0.108468, 0.130336, 0.330738], [0.007911, 0.034607, 0.073349], [0.217429, 0.385486], [0.000292, 0.020352]]
'''
# arr = ([[0.0, 0.00075, 0.004179, 0.006108, 0.010746, 0.015032, 0.030442, 0.037048, 0.063229, 0.093484, 0.126862, 0.135796, 0.18918, 0.570629, 0.779928, 0.961777], [0.060914, 0.067708, 0.06952, 0.101222, 0.13972, 0.167799, 0.074048, 0.079484, 0.093976, 0.083106, 0.090354, 0.0976, 0.103488, 0.108922, 0.11798, 0.126132, 0.137002, 0.144248, 0.1524, 0.158288, 0.164176, 0.18048, 0.185462, 0.255208, 0.199049, 0.213994, 0.231658, 0.245697, 0.240262, 0.279212, 0.34307, 0.35711, 0.401494, 0.428668, 0.515625, 0.587183], [0.000957, 0.002312, 0.005902, 0.024304, 0.004057, 0.005582, 0.008012, 0.008346, 0.010041, 0.011882, 0.013476, 0.017502, 0.021543, 0.028085, 0.033258, 0.043278, 0.073998, 0.05004, 0.062096, 0.068862, 0.08977, 0.102067, 0.110182, 0.148412, 0.173624, 0.449716, 0.19138, 0.217301, 0.224044, 0.240586, 0.244721, 0.26169, 0.31967, 0.341207, 0.328684, 0.375338, 0.381497, 0.389228, 0.430032, 0.401062, 0.456562, 0.514936, 0.574522, 0.541618, 0.553924, 0.610812, 0.633932, 0.666576, 0.822267], [0.001423, 0.004858, 0.059932, 0.114648, 0.008293, 0.022856, 0.034209, 0.045578, 0.06833, 0.080164, 0.09475, 0.130924, 0.151272, 0.16033, 0.202184, 0.372912, 0.211416, 0.357755, 0.384318, 0.40934, 0.463333, 0.53436, 0.686703], [0.001159, 0.003249, 0.00611, 0.01138, 0.033878, 0.004848, 0.008032, 0.009282, 0.012574, 0.015272, 0.020299, 0.018229, 0.02263, 0.027046, 0.037934, 0.043614, 0.101188, 0.145392, 0.05187, 0.07006, 0.083696, 0.094188, 0.10794, 0.113362, 0.118074, 0.131582, 0.165937, 0.186679, 0.196806, 0.466483, 0.704742, 0.201628, 0.232669, 0.319996, 0.434117, 0.247392, 0.255381, 0.262231, 0.272657, 0.283294, 0.29201, 0.303387, 0.328241, 0.349558, 0.35626, 0.368652, 0.378738, 0.402612, 0.417436, 0.427036, 0.451803, 0.473016, 0.488319, 0.49375, 0.499086, 0.528812, 0.638666, 0.505684, 0.51632, 0.523013, 0.5433, 0.563834, 0.569347, 0.646349, 0.651789, 0.658294, 0.667266, 0.673186, 0.712978, 0.738218, 0.801376, 0.82856, 0.842146, 0.847012, 0.870186, 0.875826, 0.878951, 0.93728, 0.943917], [0.004434, 0.012431, 0.02279, 0.0474, 0.110238, 0.03296, 0.066175, 0.081235, 0.089871, 0.1198, 0.128601, 0.184064, 0.443354, 0.158305, 0.210245, 0.22098, 0.319864, 0.336838, 0.353006, 0.381892, 0.402796, 0.42649, 0.461182, 0.480838, 0.495938, 0.729562, 0.521187, 0.557457, 0.579662, 0.599084, 0.649412, 0.68426, 0.710429, 0.733325, 0.771742, 0.800306], [0.001922, 0.002718, 0.003372, 0.004638, 0.004904, 0.006053, 0.006465, 0.029134, 0.05121, 0.009416, 0.009874, 0.01091, 0.0112, 0.01769, 0.014281, 0.014545, 0.015838, 0.01631, 0.017364, 0.021296, 0.022228, 0.022426, 0.02377, 0.024166, 0.027639, 0.028564, 0.028674, 0.031776, 0.032014, 0.036972, 0.043272, 0.057909, 0.083032, 0.110138, 0.154547, 0.137232, 0.181188, 0.191802, 0.46721, 0.20123, 0.214871, 0.293662, 0.2337, 0.244972, 0.26988, 0.286676, 0.338871, 0.34979, 0.381094, 0.408221, 0.43222, 0.451308, 0.536904, 0.559891, 0.584797, 0.665251, 0.740004, 0.801896], [0.000806, 0.027796, 0.038396, 0.0776, 0.141538, 0.049771, 0.067361, 0.091353, 0.108401, 0.12704, 0.159264, 0.18498, 0.244561, 0.210061, 0.269131, 0.312551, 0.342522, 0.426835], [0.004642, 0.011046, 0.025729, 0.052278, 0.106762, 0.016872, 0.035092, 0.04444, 0.061238, 0.066004, 0.081703, 0.073204, 0.086197, 0.098529, 0.124051, 0.142023, 0.186223, 0.155724, 0.170756, 0.199623, 0.215232, 0.272816, 0.238441, 0.288118, 0.330432, 0.356863, 0.375546, 0.407474, 0.47887, 0.49297, 0.59444, 0.56678, 0.581082, 0.611612, 0.65559, 0.692796], [0.012156, 0.015932, 0.024589, 0.130288, 0.284948, 0.020678, 0.044125, 0.05686, 0.068128, 0.071577, 0.111878, 0.123321, 0.139354, 0.151996, 0.166439, 0.184566, 0.197304, 0.199706, 0.205071, 0.219128, 0.262702, 0.239043, 0.246992, 0.2574, 0.269284, 0.287423, 0.292375, 0.304368, 0.362362, 0.427498, 0.321279, 0.328071, 0.35591, 0.37086, 0.386119, 0.39955, 0.415939, 0.425276, 0.433096, 0.442792, 0.552702, 0.679738, 0.484001, 0.491442, 0.50177, 0.507159, 0.511494, 0.521173, 0.532848, 0.545706, 0.548317, 0.565902, 0.594164, 0.605536, 0.622438, 0.64063, 0.628456, 0.634198, 0.653343, 0.65799, 0.660942, 0.671644, 0.663658, 0.694128, 0.722815, 0.733916, 0.737474, 0.778878, 0.751, 0.776577, 0.789135, 0.798928, 0.809799, 0.821893, 0.835247, 0.903398, 0.853523, 0.880864, 0.899415], [0.0, 0.000459, 0.004672, 0.012318, 0.013648, 0.015037, 0.015646, 0.015713, 0.015856, 0.021308, 0.032862, 0.05831, 0.017868, 0.023756, 0.046494, 0.052358, 0.06494, 0.104394, 0.198499, 0.154464, 0.162486, 0.175519, 0.323766, 0.43776, 0.909006, 0.95275, 0.972694], [0.108468, 0.110042, 0.112142, 0.114766, 0.117827, 0.121414, 0.125613, 0.130336, 0.135497, 0.147218, 0.153954, 0.204951, 0.361966, 0.185882, 0.225945, 0.237229, 0.31578, 0.261371, 0.301435, 0.346134, 0.395382, 0.412789, 0.449178, 0.487666, 0.814294, 0.430721, 0.507698, 0.528254, 0.549248, 0.710811, 0.735917, 0.787702, 0.84141, 0.897306, 0.955214], [0.000302, 0.000768, 0.011244, 0.035734, 0.001515, 0.002736, 0.005196, 0.008298, 0.009624, 0.014284, 0.023404, 0.025024, 0.02585, 0.029944, 0.037216, 0.040158, 0.042432, 0.066016, 0.105984, 0.073752, 0.085346, 0.089308, 0.151088, 0.174513, 0.234898, 0.245127, 0.28618, 0.736646, 0.91084, 0.974558, 0.98887], [0.005375, 0.012756, 0.058683, 0.074144, 0.163752, 0.03982, 0.068448, 0.079522, 0.088244, 0.108463, 0.100601, 0.11251, 0.144546, 0.149048, 0.170262, 0.189548, 0.223053, 0.237308, 0.386579, 0.242822, 0.249314, 0.37028, 0.29448, 0.311357, 0.347804, 0.418741, 0.441906, 0.468539, 0.482103, 0.495842, 0.506309, 0.55784, 0.522972, 0.542364, 0.598121, 0.614026, 0.626985, 0.654519, 0.709834, 0.846607, 0.7323, 0.75906, 0.823158], [0.0, 0.001892, 0.007232, 0.01172, 0.017696, 0.020712, 0.021744, 0.022382, 0.053536, 0.029868, 0.041122, 0.06301, 0.100181, 0.155846, 0.14309, 0.196754, 0.256965, 0.352926, 0.624798, 0.965582]], [[[0.0], [0.0, 0.00075, 0.004179, 0.006108, 0.010746, 0.015032, 0.030442, 0.037048, 0.063229, 0.093484, 0.126862, 0.135796, 0.18918, 0.570629, 0.779928, 0.961777], [1.0]], [[0.0], [0.060914, 0.067708, 0.06952, 0.101222, 0.13972, 0.167799, 0.074048, 0.079484, 0.093976, 0.083106, 0.090354, 0.0976, 0.103488, 0.108922, 0.11798, 0.126132, 0.137002, 0.144248, 0.1524, 0.158288, 0.164176, 0.18048, 0.185462, 0.255208, 0.199049, 0.213994, 0.231658, 0.245697, 0.240262, 0.279212, 0.34307, 0.35711, 0.401494, 0.428668, 0.515625, 0.587183], [1.0]], [[0.0], [0.000957, 0.002312, 0.005902, 0.024304, 0.004057, 0.005582, 0.008012, 0.008346, 0.010041, 0.011882, 0.013476, 0.017502, 0.021543, 0.028085, 0.033258, 0.043278, 0.073998, 0.05004, 0.062096, 0.068862, 0.08977, 0.102067, 0.110182, 0.148412, 0.173624, 0.449716, 0.19138, 0.217301, 0.224044, 0.240586, 0.244721, 0.26169, 0.31967, 0.341207, 0.328684, 0.375338, 0.381497, 0.389228, 0.430032, 0.401062, 0.456562, 0.514936, 0.574522, 0.541618, 0.553924, 0.610812, 0.633932, 0.666576, 0.822267], [1.0]], [[0.0], [0.001423, 0.004858, 0.059932, 0.114648, 0.008293, 0.022856, 0.034209, 0.045578, 0.06833, 0.080164, 0.09475, 0.130924, 0.151272, 0.16033, 0.202184, 0.372912, 0.211416, 0.357755, 0.384318, 0.40934, 0.463333, 0.53436, 0.686703], [1.0]], [[0.0], [0.001159, 0.003249, 0.00611, 0.01138, 0.033878, 0.004848, 0.008032, 0.009282, 0.012574, 0.015272, 0.020299, 0.018229, 0.02263, 0.027046, 0.037934, 0.043614, 0.101188, 0.145392, 0.05187, 0.07006, 0.083696, 0.094188, 0.10794, 0.113362, 0.118074, 0.131582, 0.165937, 0.186679, 0.196806, 0.466483, 0.704742, 0.201628, 0.232669, 0.319996, 0.434117, 0.247392, 0.255381, 0.262231, 0.272657, 0.283294, 0.29201, 0.303387, 0.328241, 0.349558, 0.35626, 0.368652, 0.378738, 0.402612, 0.417436, 0.427036, 0.451803, 0.473016, 0.488319, 0.49375, 0.499086, 0.528812, 0.638666, 0.505684, 0.51632, 0.523013, 0.5433, 0.563834, 0.569347, 0.646349, 0.651789, 0.658294, 0.667266, 0.673186, 0.712978, 0.738218, 0.801376, 0.82856, 0.842146, 0.847012, 0.870186, 0.875826, 0.878951, 0.93728, 0.943917], [1.0]], [[0.0], [0.004434, 0.012431, 0.02279, 0.0474, 0.110238, 0.03296, 0.066175, 0.081235, 0.089871, 0.1198, 0.128601, 0.184064, 0.443354, 0.158305, 0.210245, 0.22098, 0.319864, 0.336838, 0.353006, 0.381892, 0.402796, 0.42649, 0.461182, 0.480838, 0.495938, 0.729562, 0.521187, 0.557457, 0.579662, 0.599084, 0.649412, 0.68426, 0.710429, 0.733325, 0.771742, 0.800306], [1.0]], [[0.0], [0.001922, 0.002718, 0.003372, 0.004638, 0.004904, 0.006053, 0.006465, 0.029134, 0.05121, 0.009416, 0.009874, 0.01091, 0.0112, 0.01769, 0.014281, 0.014545, 0.015838, 0.01631, 0.017364, 0.021296, 0.022228, 0.022426, 0.02377, 0.024166, 0.027639, 0.028564, 0.028674, 0.031776, 0.032014, 0.036972, 0.043272, 0.057909, 0.083032, 0.110138, 0.154547, 0.137232, 0.181188, 0.191802, 0.46721, 0.20123, 0.214871, 0.293662, 0.2337, 0.244972, 0.26988, 0.286676, 0.338871, 0.34979, 0.381094, 0.408221, 0.43222, 0.451308, 0.536904, 0.559891, 0.584797, 0.665251, 0.740004, 0.801896], [1.0]], [[0.0], [0.000806, 0.027796, 0.038396, 0.0776, 0.141538, 0.049771, 0.067361, 0.091353, 0.108401, 0.12704, 0.159264, 0.18498, 0.244561, 0.210061, 0.269131, 0.312551, 0.342522, 0.426835], [1.0]], [[0.0], [0.004642, 0.011046, 0.025729, 0.052278, 0.106762, 0.016872, 0.035092, 0.04444, 0.061238, 0.066004, 0.081703, 0.073204, 0.086197, 0.098529, 0.124051, 0.142023, 0.186223, 0.155724, 0.170756, 0.199623, 0.215232, 0.272816, 0.238441, 0.288118, 0.330432, 0.356863, 0.375546, 0.407474, 0.47887, 0.49297, 0.59444, 0.56678, 0.581082, 0.611612, 0.65559, 0.692796], [1.0]], [[0.0], [0.012156, 0.015932, 0.024589, 0.130288, 0.284948, 0.020678, 0.044125, 0.05686, 0.068128, 0.071577, 0.111878, 0.123321, 0.139354, 0.151996, 0.166439, 0.184566, 0.197304, 0.199706, 0.205071, 0.219128, 0.262702, 0.239043, 0.246992, 0.2574, 0.269284, 0.287423, 0.292375, 0.304368, 0.362362, 0.427498, 0.321279, 0.328071, 0.35591, 0.37086, 0.386119, 0.39955, 0.415939, 0.425276, 0.433096, 0.442792, 0.552702, 0.679738, 0.484001, 0.491442, 0.50177, 0.507159, 0.511494, 0.521173, 0.532848, 0.545706, 0.548317, 0.565902, 0.594164, 0.605536, 0.622438, 0.64063, 0.628456, 0.634198, 0.653343, 0.65799, 0.660942, 0.671644, 0.663658, 0.694128, 0.722815, 0.733916, 0.737474, 0.778878, 0.751, 0.776577, 0.789135, 0.798928, 0.809799, 0.821893, 0.835247, 0.903398, 0.853523, 0.880864, 0.899415], [1.0]], [[0.0], [0.0, 0.000459, 0.004672, 0.012318, 0.013648, 0.015037, 0.015646, 0.015713, 0.015856, 0.021308, 0.032862, 0.05831, 0.017868, 0.023756, 0.046494, 0.052358, 0.06494, 0.104394, 0.198499, 0.154464, 0.162486, 0.175519, 0.323766, 0.43776, 0.909006, 0.95275, 0.972694], [1.0]], [[0.0], [0.108468, 0.110042, 0.112142, 0.114766, 0.117827, 0.121414, 0.125613, 0.130336, 0.135497, 0.147218, 0.153954, 0.204951, 0.361966, 0.185882, 0.225945, 0.237229, 0.31578, 0.261371, 0.301435, 0.346134, 0.395382, 0.412789, 0.449178, 0.487666, 0.814294, 0.430721, 0.507698, 0.528254, 0.549248, 0.710811, 0.735917, 0.787702, 0.84141, 0.897306, 0.955214], [1.0]], [[0.0], [0.000302, 0.000768, 0.011244, 0.035734, 0.001515, 0.002736, 0.005196, 0.008298, 0.009624, 0.014284, 0.023404, 0.025024, 0.02585, 0.029944, 0.037216, 0.040158, 0.042432, 0.066016, 0.105984, 0.073752, 0.085346, 0.089308, 0.151088, 0.174513, 0.234898, 0.245127, 0.28618, 0.736646, 0.91084, 0.974558, 0.98887], [1.0]], [[0.0], [0.005375, 0.012756, 0.058683, 0.074144, 0.163752, 0.03982, 0.068448, 0.079522, 0.088244, 0.108463, 0.100601, 0.11251, 0.144546, 0.149048, 0.170262, 0.189548, 0.223053, 0.237308, 0.386579, 0.242822, 0.249314, 0.37028, 0.29448, 0.311357, 0.347804, 0.418741, 0.441906, 0.468539, 0.482103, 0.495842, 0.506309, 0.55784, 0.522972, 0.542364, 0.598121, 0.614026, 0.626985, 0.654519, 0.709834, 0.846607, 0.7323, 0.75906, 0.823158], [1.0]], [[0.0], [0.0, 0.001892, 0.007232, 0.01172, 0.017696, 0.020712, 0.021744, 0.022382, 0.053536, 0.029868, 0.041122, 0.06301, 0.100181, 0.155846, 0.14309, 0.196754, 0.256965, 0.352926, 0.624798, 0.965582], [1.0]]])

