import numpy as np
from brainiak.funcalign.srm import SRM
from scipy.stats import zscore, stats

# train one run at a time (run # needs to be specified as input)
def SRM_V1(train,test,srm_k,n_iter):
     # initialize model
    print('Building Models')
    srm_train_data = SRM(n_iter=n_iter, features=srm_k)
   
    # fit model to training data
    print('Training Models')
    srm_train_data.fit(train)

    print('Testing Models')
    shared_data = srm_train_data.transform(test)


    return shared_data


# train on run 1, train on run 2, test on run 1, test on run 2
def SRM_V2(run1,run2,srm_k,n_iter):
    # initialize model
    print('Building Models')
    n_iter= n_iter
    srm_k = srm_k
    srm_train_run1 = SRM(n_iter=n_iter, features=srm_k)
    srm_train_run2 = SRM(n_iter=n_iter, features=srm_k)

    # fit model to training data
    print('Training Models')
    srm_train_run1.fit(run1)
    srm_train_run2.fit(run2)

    print('Testing Models')
    shared_data_run1 = srm_train_run2.transform(run1)
    shared_data_run2 = srm_train_run1.transform(run2)

    # average test data across subjects
    run1 = sum(shared_data_run1)/len(shared_data_run1)
    run2 = sum(shared_data_run2)/len(shared_data_run2)

    return run1, run2


# train on concatenated run1 and run2, then test on run1, and test on run 2
def SRM_V3(run1,run2,srm_k,n_iter):
    # initialize model
    print('Building Models')
    n_iter= n_iter
    srm_k = srm_k
    srm_train = SRM(n_iter=n_iter, features=srm_k)

    # concatenate run1 and run2 within subject before fitting SRM
    runs = []
    for i in range(len(run1)):
        runs.append(np.concatenate((run1[i],run2[i]),axis=1))

    # fit model to training data
    print('Training Models')
    srm_train.fit(runs)

    print('Testing Models')
    shared_data_run1 = srm_train.transform(run1)
    shared_data_run2 = srm_train.transform(run2)

    # average test data across subjects
    run1 = sum(shared_data_run1)/len(shared_data_run1)
    run2 = sum(shared_data_run2)/len(shared_data_run2)

    return run1, run2


