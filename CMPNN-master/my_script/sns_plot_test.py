import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

tips = sns.load_dataset("tips")
print(tips.head())
fig, axes =plt.subplots(1,6)
# sns.boxplot(y=tips['total_bill'])
# sns.boxplot(x=tips["total_bill"],ax=axes[0])
# sns.boxplot(y=tips["total_bill"],ax=axes[1])
# sns.boxplot(x="day", y="total_bill", data=tips,ax=axes[2])
sns.boxplot(x="day", y="total_bill", hue="smoker", data=tips, palette="Set3",ax=axes[3])
sns.boxplot(x="day", y="total_bill", hue="time",data=tips, linewidth=2.5,ax=axes[4])
sns.boxplot(data=tips, orient="h", palette="Set2",ax=axes[5])
plt.show()
print('Done')

