import os

desc=['ncols         2438',
       'nrows         2290',
       'xllcorner     -59881.25498585',
       'yllcorner     3151836.6306969',
       'cellsize      100',
       'NODATA_value  -9999']

list=os.listdir('../数据集/data')

for i in list:
    with open('../数据集/data/'+i,'r+') as f:
        content=f.read()
        f.seek(0,0)
        for j in desc:
            f.write(j+'\n')
        f.write(content)

