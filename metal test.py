
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.tsa.stattools import adfuller
# import math
# import logging
import os
os.chdir('C:/Users/35003/OneDrive/Desktop/dadi')
print(os.getcwd())
import pandas as pd
import numpy as np
from auto_corr_test import AutoCorr
from auto_corr_test import DataClean
from auto_corr_test import DataNormalizer
import seaborn as sns
import matplotlib.pyplot as plt

#%% 字体
#中文字体
#import matplotlib.font_manager

# 获取系统中所有可用的字体文件路径
#available_fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')



# 打印可能支持中文的字体的文件路径
# print("\nFonts that might support Chinese: ")
# for font_path in available_fonts:
#     font_name = font_path.split("\\")[-1].lower()  # 适用于Windows路径，对于其他OS，可能需要适当修改
#     if "hei" in font_name or "song" in font_name or "fang" in font_name or "kai" in font_name or "ming" in font_name:
#         print(font_path)
# # #font setting
# plt.rcParams[ 'font.sans-serif' ] = [ 'SimHei ' ]
# plt.rcParams[ 'axes.unicode_minus' ] = False


#%%
''' read data '''
path = os.getcwd()
raw_data=pd.read_excel(path+'\\'+'螺纹钢基础数据.xlsx')
raw_data.columns = raw_data.iloc[0]
raw_data.columns = ['date'] + raw_data.columns[1:].tolist()

# choose data for test

df=raw_data.iloc[1008:,:]
df=df.reset_index(drop=True)
df1 = pd.to_datetime(df.iloc[:,0], format='%Y/%m/%d-%H:%M')
df1 = df1.to_frame()
df['date'] = pd.to_datetime(df1.iloc[:,0])
df['date'] = df['date'].dt.date
test_data=df.iloc[:,1:]
#test_data.columns

#%%
'''data cleaning'''
obj1=DataClean()
test_data=obj1.clean_na(test_data)
print(test_data.isna().any().any())
print(test_data.head(10))
# test_data=obj1.clean_zero(test_data)
print((test_data == 0).any().any())
#test_data=obj1.clean_abnormal(test_data,0.7)
#%%
'''decompose'''

'''trytry'''
trytry= test_data.iloc[:,0]
trytry=obj1.decomposition(trytry)
obj = AutoCorr(trytry.iloc[:,2])
obj.ADF_test()
obj.differencing(periods=14)
obj2=DataNormalizer(trytry.iloc[:,0])
df_normalized=obj2.normalize()



#先根据stl后的season的acf判断季节的周期和trend是否要做差分
#做差分后trend也可以是平稳的
#0列的显著lag是14，周期为7


test_data=obj1.decomposition(test_data)
print(test_data.head(10))
 #%%
'''stationary test'''
obj = AutoCorr(test_data)
obj.ADF_test()
print(obj.data_non_stationary_original)
#%%
#type(obj.data_non_stationary_original)
#defferencing non-stationary data

#提取出non-stationary series的column name
column_names_list = obj.data_non_stationary_original.columns.tolist()
print(column_names_list)
obj = AutoCorr(test_data[column_names_list])
obj.differencing()
df_stationary=obj.data_stationary_diff
print(df_stationary.head(10))
test_data[column_names_list]=df_stationary[column_names_list]
#%%
'''normalize'''
obj2=DataNormalizer(test_data)
df_normalized=obj2.normalize()

#polt
sns.set(style="whitegrid")


# for column in df_normalized.columns:
#     plt.figure(figsize=(10, 6))
#     sns.kdeplot(df_normalized[column], fill=True)
#     plt.title(f'PDF of {column}')
#     plt.xlabel(column)
#     plt.ylabel('Density')
#     plt.show()
# from matplotlib.font_manager import FontProperties
#
# myfont = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf')
#
#
# for column in df_normalized.columns:
#     plt.figure(figsize=(10, 6))
#     sns.kdeplot(df_normalized[column], fill=True)
#     plt.title(f'PDF of {column}', fontproperties=myfont)  # 使用字体属性
#     plt.xlabel(column, fontproperties=myfont)  # 如果xlabel也包含中文也需要指定字体属性
#     plt.ylabel('Density', fontproperties=myfont)  # 如果ylabel也包含中文也需要指定字体属性
#     plt.show()
#%%
'''final data set'''
dataset_normalized = pd.concat([df.iloc[:,0], df_normalized.iloc[:, :]], axis=1)
print(dataset_normalized.head(10))
dataset_normalized.to_excel("标准化螺纹钢基础数据.xlsx", index=False)

#%%
#清除字体缓存
import matplotlib
import os
#
font_cache_path = os.path.join(matplotlib.get_cachedir(), 'fontlist-v310.json')
if os.path.exists(font_cache_path):
    os.remove(font_cache_path)

