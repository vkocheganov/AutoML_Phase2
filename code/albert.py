__author__ = 'vmkocheg'

from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
import time
import numpy as np
from sets import Set
from time import strftime

from preprocess import GBT_params
from utils import make_classification

def albert_predict(train_data,labels,valid_data,test_data,output_dir,time_budget,target_num, is_sparse):
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


#    FS_iterations =max(1,int(5000/target_num * (5000./n_samples)*2000./n_features))
#     FS_iterations = 1000
#     print ("FS_iterations = %d\n" % FS_iterations)
#    select_clf = ExtraTreesClassifier(n_estimators=FS_iterations,max_depth=3)

    # select_clf = ExtraTreesClassifier(n_estimators=FS_iterations,max_depth=4)
    # select_clf.fit(train_data, labels)
    # print("FS time = ", time.time() - start_time)
    # my_mean =1./(10*n_features)
    # print(my_mean)
    # print("feature importances: ", np.sort(select_clf.feature_importances_))
    #
    # train_data = select_clf.transform(train_data,threshold=my_mean )
    # valid_data = select_clf.transform(valid_data,threshold=my_mean )
    # test_data = select_clf.transform(test_data,threshold=my_mean)
    # print(my_mean)
    # print(train_data.shape)

#    exit(1)
    ######################### Make validation/test predictions
    n_features=train_data.shape[1]
    # if n_features < 100:
    #     gbt_features=n_features
    # else:
    gbt_features=int(n_features**0.5)
#    gbt_iterations= int((time_budget / 3000.) * 3000000/(gbt_features * target_num) * (7000./n_samples))
    gbt_iterations = 3000
#    gbt_params=GBT_params(n_iterations=gbt_iterations,depth=int(10 * np.log2(gbt_iterations)/14.3), learning_rate=0.01,subsample_part=0.6,n_max_features=gbt_features,min_samples_split=5, min_samples_leaf=3)
    gbt_params=GBT_params(n_iterations=gbt_iterations,depth=5, learning_rate=0.01,subsample_part=0.6,n_max_features=gbt_features,min_samples_split=10, min_samples_leaf=5)
    gbt_params.print_params()
    (y_valid, y_test) = make_classification(gbt_params, train_data, labels, valid_data, test_data)
    print("y_valid.shape = ",y_valid.shape )
    print("y_test.shape = ",y_test.shape )
    return (y_valid, y_test)


from sys import argv, path
import sys
import datetime
zipme = False # use this flag to enable zipping of your code submission
the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
submission_filename = 'automl_sample_submission_' + the_date
debug_mode = 0
verbose = True
import os
import numpy as np
import time
overall_start = time.time()
run_dir = os.path.abspath(".")
lib_dir = os.path.join(run_dir, "lib")
res_dir = os.path.join(run_dir, "res")

# Our libraries
path.append (run_dir)
path.append (lib_dir)

import data_io                       # general purpose input/output functions
from data_io import vprint           # print only in verbose mode
from data_manager import DataManager # load/save data and get info about them

sys.path.append("libs")

default_input_dir="C:\\Users\\vmkocheg\\Documents\\MLContest\\Phase2\\input"
default_output_dir="C:\\Users\\vmkocheg\\Documents\\MLContest\\Phase2\\output"
if len(argv)==1: # Use the default input and output directories if no arguments are provided
    input_dir = default_input_dir
    output_dir = default_output_dir
else:
    input_dir = argv[1]
    output_dir = os.path.abspath(argv[2]);

#### INVENTORY DATA (and sort dataset names alphabetically)
datanames = data_io.inventory_data(input_dir)
#### DEBUG MODE: Show dataset list and STOP
if debug_mode>=3:
    data_io.show_io(input_dir, output_dir)
    data_io.write_list(datanames)
    datanames = [] # Do not proceed with learning and testing


for basename in datanames: # Loop over datasets
    if basename not in ["albert"]:
        continue

    vprint( verbose,  "************************************************")
    vprint( verbose,  "******** Processing dataset " + basename.capitalize() + " ********")
    vprint( verbose,  "************************************************")

    # ======== Learning on a time budget:
    # Keep track of time not to exceed your time budget. Time spent to inventory data neglected.
    start = time.time()

    # ======== Creating a data object with data, informations about it
    vprint( verbose,  "======== Reading and converting data ==========")
    D = DataManager(basename, input_dir, replace_missing=True, filter_features=True, verbose=verbose)
    print D

    # ======== Keeping track of time
    time_spent = time.time() - start
    vprint( verbose,  "time spent %5.2f sec" %time_spent)

    vprint( verbose,  "======== Creating model ==========")
    train_data = D.data['X_train']
    labels = D.data['Y_train']
    valid_data = D.data['X_valid']
    test_data = D.data['X_test']
    # print (train_data.shape)
    # print (valid_data.shape)
    # print (test_data.shape)
    # print (labels.shape)
    time_spent = 0                   # Initialize time spent learning
    (Y_valid, Y_test) = locals()[basename+"_predict"](train_data,labels, valid_data, test_data,output_dir, D.info['time_budget'],D.info['target_num'],D.info['is_sparse'])
    Y_valid=Y_valid[:,1]
    Y_test=Y_test[:,1]
    time_spent = time.time() - start

    vprint( verbose,  "[+] Prediction success, time spent so far %5.2f sec" % (time.time() - start))
    # Write results
    filename_valid = basename + '_valid_' + '.predict'
    data_io.write(os.path.join(output_dir,filename_valid), Y_valid)
    filename_test = basename + '_test_' + '.predict'
    data_io.write(os.path.join(output_dir,filename_test), Y_test)

    vprint( verbose,  "[+] Results saved, time spent so far %5.2f sec" % (time.time() - start))
    time_spent = time.time() - start

overall_time_spent = time.time() - overall_start
vprint( verbose,  "[+] Done")
vprint( verbose,  "[+] Overall time spent %5.2f sec " % overall_time_spent)
