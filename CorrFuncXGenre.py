import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys
from scipy import stats

subj = sys.argv[1]
roi = sys.argv[2]

datadir = '/Users/jamalw/Desktop/PNI/music_event_structures/subjects/'

def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))

# take average and correlation for each subject pair
# load in subject data (16 lists of #VxTRs for each song)
subjData1 = pickle.load(open(datadir + subj + '/reorder1' + roi + '.p','rb'))
subjData2 = pickle.load(open(datadir + subj + '/reorder2' + roi + '.p','rb'))
# append average of each list to new array
avgsubjData1 = []
avgsubjData2 = []
for j in range(len(subjData1)):
    avgsubjData1.append(np.mean(subjData1[j],1))
    avgsubjData2.append(np.mean(subjData2[j],1))
# stack full and average data horizontally and vertically (respectively) to form #VxallTRs
subjData1Horz    = np.hstack(subjData1)
subjData2Horz    = np.hstack(subjData2)
avgsubjData1Horz = np.vstack(avgsubjData1)
avgsubjData2Horz = np.vstack(avgsubjData2)

# perform correlation on full data and average data
corrD    = corr2_coeff(subjData1Horz,subjData2Horz)
avgCorrD = corr2_coeff(avgsubjData1Horz,avgsubjData2Horz)

np.save(datadir + str(subj) + '/' + 'corrD' + roi, corrD)
np.save(datadir + str(subj) + '/' + 'avgCorrD' + roi, avgCorrD)

# compute average section of avgCorrD
classical_within  = avgCorrD[0:8,0:8]
jazz_within       = avgCorrD[8:16,8:16]
classJazz_between = avgCorrD[8:16,0:8]
jazzClass_between = avgCorrD[0:8,8:16]

plt.figure(1)
plt.imshow(corrD,interpolation='none')
plt.colorbar()
plt.axis('tight')

plt.figure(2)
plt.imshow(avgCorrD,interpolation='none')
plt.plot((-.5, 15.5), (7.5, 7.5), 'k-')
plt.plot((7.5, 7.5), (-.5, 15.5), 'k-')
plt.colorbar()
plt.axis('tight')

plt.figure(3)
allComparisonsAvg = np.array([np.mean(classical_within),np.mean(jazz_within),np.mean(classJazz_between),np.mean(jazzClass_between)])
allComparisonsSem = np.array([stats.sem(np.mean(classical_within,0)),stats.sem(np.mean(jazz_within,0)),stats.sem(np.mean(classJazz_between,0)),stats.sem(np.mean(jazzClass_between,0))])
N = 4
ind = np.arange(N)
width = 0.35
labels = ['Classical vs Classical', 'Jazz vs Jazz', 'Jazz vs Classical', 'Classical vs Jazz']
plt.xticks(ind + width / 2, labels)
plt.bar(ind, allComparisonsAvg, width, color='k',yerr = allComparisonsSem,error_kw=dict(ecolor='lightseagreen',lw=3,capsize=0,capthick=0))
plt.plot((0,3.5),(0,0),'k-')
#allComparisonsStd = np.array([np.std(classical_within),np.std(jazz_within),np.std(classJazz_between),np.std(jazzClass_between)])
plt.show()

