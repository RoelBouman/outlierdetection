import pickle
import os
import numpy as np
import pandas as pd

from pyod.models.knn import KNN 

result_dir = "D:\\Promotie\\outlier_detection\\results"

data_names = os.listdir(result_dir)
method_names = [x.replace(".pickle", "") for x in os.listdir(os.path.join(result_dir, data_names[0]))]

method_names = os.listdir(os.path.join(result_dir, data_names[0]))

for picklefile_name in picklefile_names:
    
    #check if data path exists, and make it if it doesn't
    target_dir = os.path.join('result_dir', picklefile_name.replace(".pickle", ""))
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    #print name for reporting purpose
    print(picklefile_name)
    
    full_path_filename = os.path.join(pickle_dir, picklefile_name)
    
    data = pickle.load(open(full_path_filename, 'rb'))
    X, y = data["X"], np.squeeze(data["y"])
    
    #loop over all methods:
    CV_results = {}
    for method, settings in methods_params.items():
        
        print("______"+method)
        
        target_file_name = os.path.join(target_dir, method+".pickle")
        #check if file exists and is non-empty
        if os.path.exists(target_file_name) and os.path.getsize(target_file_name) > 0:
            print("results already calculated, skipping recalculation")
        else:
            clf = ODWrapper(settings["method"]())
            pipeline = make_pipeline(RobustScaler(), clf)
            
            clf_settings = dict()
            for key in settings["params"].keys():
                clf_settings["odwrapper__"+key] = settings["params"][key]
            
            gridsearch = GridSearchCV(pipeline, clf_settings, scoring=scorer, cv = StratifiedKFold(n_splits=5,shuffle=True), return_train_score=False)
            gridsearch.fit(X, y)
            
            #CV_results[method] = pd.DataFrame(gridsearch.cv_results_)
            with open(target_file_name, 'wb') as handle:
                pickle.dump(pd.DataFrame(gridsearch.cv_results_), handle, protocol=pickle.HIGHEST_PROTOCOL)

