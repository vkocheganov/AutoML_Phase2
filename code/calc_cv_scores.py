__author__ = 'vmkocheg'
from sklearn.cross_validation import KFold
import time
from libs.libscores import *
from preprocess import GBT_params
from sklearn import ensemble

def Calc_CV_ERROR(classifier, data, solution, cv_folds):
    cv_scores = np.zeros(cv_folds)

    kf=KFold(len(solution), cv_folds)
    cv_iter = 0
    for cv_train, cv_test in kf:
        print("cv iteration %d" % cv_iter)
        start_time = time.time()
        classifier.fit(data[cv_train], solution[cv_train])
        cv_test_pred = classifier.predict_proba(data[cv_test])[:,1]
        print("time = %d"%(time.time() - start_time))
        cv_scores[cv_iter] = bac_metric(cv_test_pred, solution[cv_test], task='binary.classification')
        print "cv_score = %1.5f"%cv_scores[cv_iter]
        cv_iter += 1;

    return cv_scores.mean()


def make_cross_validation(data, solution, cv_folds, params_begin, params_mult_factor, params_add_factor, params_num_iter):
    #    params = GBT_params(params_begin.n_iterations,params_begin.depth, params_begin.learning_rate, params_begin.subsample_part, params_begin.n_max_features)
    params = GBT_params()
    cv_iterations =params_num_iter.n_iterations *params_num_iter.depth * params_num_iter.learning_rate * params_num_iter.subsample_part*params_num_iter.n_max_features
    cv_res=np.zeros(cv_iterations)
    cv_times=np.zeros(cv_iterations)

    cur_iter = 0
    params.n_iterations = params_begin.n_iterations
    for n_iterations in range(params_num_iter.n_iterations):
        params.learning_rate = params_begin.learning_rate
        for n_learning_rate in range(params_num_iter.learning_rate):
            params.depth = params_begin.depth
            for n_max_depth in range(params_num_iter.depth):
                params.subsample_part = params_begin.subsample_part
                for subsample_part in range(params_num_iter.subsample_part):
                    params.n_max_features = params_begin.n_max_features
                    for n_max_features in range(params_num_iter.n_max_features):
                        start_time = time.time()
                        params.print_params()
                        clf = ensemble.GradientBoostingClassifier(n_estimators=params.n_iterations,learning_rate=params.learning_rate, max_depth=params.depth, subsample=params.subsample_part, max_features=int(params.n_max_features))
                        cv_res[cur_iter] = Calc_CV_ERROR(clf,data, solution, cv_folds)
                        print ("CV score = %1.5",cv_res[cur_iter])
                        cv_times[cur_iter]=  time.time() - start_time
                        print ("CV time = %d",cv_times[cur_iter])
                        params.n_max_features *= params_mult_factor.n_max_features
                        params.n_max_features += params_add_factor.n_max_features
                        cur_iter +=1
                    params.subsample_part *= params_mult_factor.subsample_part
                    params.subsample_part += params_add_factor.subsample_part
                params.depth *= params_mult_factor.depth
                params.depth += params_add_factor.depth
            params.learning_rate *= params_mult_factor.learning_rate
            params.learning_rate += params_add_factor.learning_rate
        params.n_iterations*= params_mult_factor.n_iterations
        params.n_iterations += params_add_factor.n_iterations
    return (cv_res, cv_times)