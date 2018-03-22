import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/beh/'

df1 = pd.read_excel(datadir + 'familiarity_scores.xlsx',"Day 1")
df2 = pd.read_excel(datadir + 'familiarity_scores.xlsx',"Day 2")


data1 = df1.as_matrix(columns=df1.columns[0:])
data2 = df2.as_matrix(columns=df2.columns[0:])

mean_data1 = np.mean(data1)
mean_data2 = np.mean(data2)
sem_data1 = stats.sem(np.reshape(data1,(16*23)))
sem_data2 = stats.sem(np.reshape(data2,(16*23)))

all_data_means = np.array([mean_data1,mean_data2])
all_data_sems = np.array([sem_data1,sem_data2])

N = 2
ind = np.arange(N)
width = 0.35

plt.bar(ind, all_data_means, width, color='k')
#plt.bar(ind, all_data_means, width, color='k', yerr = all_data_sems, error_kw=dict(ecolor='lightseagreen',lw=3,capsize=0,capthick=0))
plt.ylabel('Familiarity Ratings', fontsize=15)
plt.title('Familiarity Judgements',fontweight='bold',fontsize=18)
labels = ['Day 1', 'Day 2']
plt.xticks(ind + width / 4.5, labels, fontsize = 15)
axes = plt.gca()
axes.set_ylim([0,5])

plt.savefig('familiarity_fig.svg')


