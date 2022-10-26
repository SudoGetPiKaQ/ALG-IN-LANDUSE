import src
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

my_chinese_dict=dict(fontsize=7,
              color='black',
              family='Songti SC',
              weight='light',
              )

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 7
NUM= len(src.log)
x=np.arange(1,1+NUM,1)
fig, (ax1,ax2,ax3) = plt.subplots(3, 1,dpi=150)


[i.set_xticks(np.arange(1, NUM+1)) and i.set_xlim(1, NUM) for i in [ax1,ax2,ax3]]
args={}

ax1.plot(x, src.lr_oa,label='LR',**args)
ax1.plot(x, src.tan_oa,label='TAN',**args)
ax1.plot(x, src.ann_oa,label='ANN',**args)
ax1.plot(x, src.rf_oa,label='RF',**args)



ax1.set_xlabel('训练次数',fontdict=my_chinese_dict)
ax1.set_ylabel('OA')
# ax1.grid(True)
ax2.plot(x, src.lr_f1, x, src.tan_f1,x, src.ann_f1,x, src.rf_f1,**args)
ax2.set_xlabel('训练次数',fontdict=my_chinese_dict)
ax2.set_ylabel('F1')
# ax2.grid(True)

ax3.plot(x, src.lr_kappa, x, src.tan_kappa,x, src.ann_kappa,x, src.rf_kappa,**args)
ax3.set_xlabel('训练次数',fontdict=my_chinese_dict)
ax3.set_ylabel('Kappa')
# ax3.grid(True)
# fig.tight_layout()
fig.legend()
# plt.show()
plt.savefig('./accu.tiff',dpi=300)
