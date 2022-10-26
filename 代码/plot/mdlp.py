import src

import matplotlib.pyplot as plt

print([len(i) for i in src.div])

plt.rcParams["font.sans-serif"]=["Songti SC"]

tag = sum([[index + 1 for j in i] for index, i in enumerate(src.div)], [])
data = sum(src.div, [])

#
fig, ax = plt.subplots(dpi=300)
ax.scatter(data, tag, c=tag, linewidths=0.1)
#
ax.set_xlabel(r'数据范围')

index_ls = ['商场点密度', '高程', '到道路距离', '到水体距离', '到地铁站距离', '到高速出入口距离', '到铁路距离', '坡度', '到区中心距离', '到机场距离', '学校点密度',
            '夜灯强度', '景点点密度', '到火车站距离', '医院点密度']

index_ls = [f"{i[0]}({i[1]})" for i in zip(index_ls, [len(i) for i in src.div])]

plt.yticks(range(1, 16), index_ls, )
# ax.set_ylabel('驱动因子')
# ax.set_title('src.divide Points')
#
ax.grid(True)
fig.tight_layout()
# plt.show()
plt.savefig('./mdlp.tiff')
