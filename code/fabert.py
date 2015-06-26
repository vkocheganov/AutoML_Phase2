__author__ = 'vmkocheg'
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
import time
import numpy as np
from sets import Set
from time import strftime

from preprocess import GBT_params
from utils import make_classification

def fabert_predict(train_data,labels,valid_data,test_data,output_dir,time_budget,target_num, is_sparse):
    print(strftime("%Y-%m-%d %H:%M:%S"))
    print("make multiclass prediction\n")
    np_seed = int(time.time())
    np.random.seed(np_seed)
    print ("np seed = " , np_seed)
    print(train_data.shape)

    print("train_data.shape == (%d,%d)\n"%train_data.shape)
    n_features = train_data.shape[1]
    n_samples = train_data.shape[0]
    start_time = time.time()
    if is_sparse:
        print("no FS, it is sparse data\n")
        train_data=train_data.toarray()
        valid_data=valid_data.toarray()
        test_data=test_data.toarray()
        # train_data = select_clf.transform(train_data,threshold=my_mean )
        # valid_data = select_clf.transform(valid_data,threshold=my_mean )
        # test_data = select_clf.transform(test_data,threshold=my_mean)
        print("sparse converting time = ", time.time() - start_time)
        start_time = time.time()


    FS_iterations = max(1,int(5000/target_num * (5000./n_samples)*2000./n_features))
    print ("FS_iterations = %d\n" % FS_iterations)
    select_clf = ExtraTreesClassifier(n_estimators=FS_iterations,max_depth=3)
    select_clf.fit(train_data, labels)
    print("FS time = ", time.time() - start_time)

    my_mean =1./(10*n_features)
    train_data = select_clf.transform(train_data,threshold=my_mean )
    valid_data = select_clf.transform(valid_data,threshold=my_mean )
    test_data = select_clf.transform(test_data,threshold=my_mean)
    print(my_mean)
    print(train_data.shape)

    ######################### Make validation/test predictions
    n_features=train_data.shape[1]
    if n_features < 100:
        gbt_features=n_features
    else:
        gbt_features=int(n_features**0.5)
    gbt_iterations= int((time_budget / 3000.) * 3000000/(gbt_features * target_num) * (7000./n_samples))
    gbt_params=GBT_params(n_iterations=gbt_iterations,depth=int(10 * np.log2(gbt_iterations)/14.3), learning_rate=0.01,subsample_part=0.6,n_max_features=gbt_features,min_samples_split=5, min_samples_leaf=3)
    gbt_params.print_params()
    (y_valid, y_test) = make_classification(gbt_params, train_data, labels, valid_data, test_data)
    print("y_valid.shape = ",y_valid.shape )
    print("y_test.shape = ",y_test.shape )
    return (y_valid, y_test)
