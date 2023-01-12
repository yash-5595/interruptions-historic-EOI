


'''
Training classifier seperating Events of Interest from other events.
'''


import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helpers import return_det_waveform
import random
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix

def geteventsmatrix(dfevents):
    index = [10,20,30,40,50,60,70,80,90,100]
    columns = [300,600,900,1200,1500]
    dfmatrix = pd.DataFrame(index=index, columns=columns)
    dfmatrix = dfmatrix.fillna(0)
    for index, row in dfevents.iterrows():
        mdip = int(round(row['avg_reduction'],-1))
        ldip = int(round(row['time_elapsed'],-2))
        if(ldip>1500):
            ldip = 1500
        dfmatrix.loc[mdip,ldip ] += 1
    return dfmatrix


def _do_for_one_event(one_event, trig_threshold):
    FINAL_ARR = []
    
    RECENT_VOLUME_THRESHOLD = 10
    PREV_COUNT_THRESHOLD = 1

    reduce_index = np.where(np.array(one_event.reductions)>trig_threshold)
    if(reduce_index[0].shape[0]== 0):
        return False, []
    trigger_start_ts = one_event.time_start  + pd.Timedelta(seconds = 300*reduce_index[0][0])
    end_ts = trigger_start_ts + pd.Timedelta(seconds = look_forward)
    curr_arrivals = return_det_waveform(one_event.DetectorID, trigger_start_ts,end_ts )
    FINAL_ARR.extend(list(np.cumsum(curr_arrivals.values)))
    recent_cycle_arrivals = return_det_waveform(one_event.DetectorID, one_event.time_start- pd.Timedelta(seconds = look_forward), one_event.time_start)
    prev_count = {}
    all_prev = [7,14, 21]
    for i in  all_prev:
        time_prev_start =  trigger_start_ts - pd.Timedelta(days=i)
        time_prev_end =   end_ts - pd.Timedelta(days=i)


        count_df_prev = return_det_waveform(one_event.DetectorID, time_prev_start,time_prev_end )

        if(count_df_prev.shape[0]<10):
            pass
        else:

            prev_count[i]  = count_df_prev  
    print(f"Detector {one_event.DetectorID} ts {trigger_start_ts} curr_arrivals {curr_arrivals.sum()[0]} recent cycles {recent_cycle_arrivals.sum()[0]}") 

    
    if(recent_cycle_arrivals.sum()[0] < RECENT_VOLUME_THRESHOLD ):
        return False,  []
    else:
        FINAL_ARR.extend(list(np.cumsum(recent_cycle_arrivals.values)))
    
    count = 0
    for each_prev in all_prev:
        if each_prev in prev_count:
#             print(f"Yash: {each_prev} sum {prev_count[each_prev].sum()[0]}")
            if(prev_count[each_prev].sum()[0]>RECENT_VOLUME_THRESHOLD):
                
                FINAL_ARR.extend(list(np.cumsum(prev_count[each_prev].values)))
                count+=1
                if(count ==PREV_COUNT_THRESHOLD ):
                    return True, FINAL_ARR
                
    return False, []
            

# mappings file 
map_file = pd.read_csv('MAPPINGS_DET_INFO_OCT_2022.csv')
all_corridors = os.listdir('corridor_label_data/')

# corridor = 'sr436-09-2022'
final_df = []
for corridor in all_corridors:
    all_files = glob.glob(f"corridor_label_data/{corridor}/*.pickle")
    
    for each_file in all_files:
        final_df.append(pd.read_pickle(each_file))
EOI_all = pd.concat(final_df)


EOI_all_filter = EOI_all[(EOI_all.start_volume>25) & (EOI_all.avg_reduction>=10)]

Ev_of_In = EOI_all_filter[(EOI_all_filter.avg_reduction>=20) & ((EOI_all_filter.time_elapsed>300) )].copy()


look_forward = 90 # Look forward 90 sec for the trigger 
Trigger_threshold=0.4 # instant reduction threshold
reduction_threshold = 60 # threshold for volume reduction 
duration_threshold=600  # threshold for duration


trigger_exist = []
for i, each_row in Ev_of_In.iterrows():
    all_reductions = each_row.reductions
    trigger_exist.append(any([x>Trigger_threshold for x in all_reductions]))
Ev_of_In['trigger_exist'] = trigger_exist  
triggers = Ev_of_In[Ev_of_In.trigger_exist == True]

print(geteventsmatrix(triggers))


p_events =  triggers[(triggers.avg_reduction>=reduction_threshold)&((triggers.time_elapsed>=duration_threshold))]

other_events =triggers[~((triggers.avg_reduction>=reduction_threshold)&((triggers.time_elapsed>=duration_threshold)))]

final_arr= []
for i in range(p_events.shape[0]):
    one_event = p_events.iloc[i]
    flag,val_list = _do_for_one_event(one_event, 0.5)
    print(flag)
    if(flag):
        final_arr.append(val_list)

final_arr_other_class = []
i =0
NO_EVENTS = 200
while (i<NO_EVENTS):
    one_event = other_events.iloc[random.choice(range(other_events.shape[0]))]
    flag,val_list = _do_for_one_event(one_event, 0.4)
    if(flag):
        final_arr_other_class.append(val_list)
        i+=1


pos_class = np.array(final_arr)
pos_lables = np.ones(pos_class.shape[0])
neg_class = np.array(final_arr_other_class)
neg_lables = np.zeros(neg_class.shape[0])

input_data = np.concatenate([pos_class,neg_class ], axis = 0)
input_lables = np.concatenate([pos_lables,neg_lables ], axis = 0)

# standardize and apply PCA
scaler = preprocessing.StandardScaler().fit(input_data)
inputdata_scaled = scaler.transform(input_data)

pca = PCA()
pca = pca.fit(inputdata_scaled)

inputdata_pca = pca.transform(inputdata_scaled )
X_train, X_test, y_train, y_test = train_test_split(inputdata_pca, input_lables, test_size=0.4, )


# Initialize SVM classifier
clf = svm.SVC(kernel='rbf', class_weight={1: 100})

clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

matrix = plot_confusion_matrix(clf, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Confusion matrix for our classifier')
plt.show()


matrix = plot_confusion_matrix(clf, X_train, y_train,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Confusion matrix for our classifier')
plt.show()