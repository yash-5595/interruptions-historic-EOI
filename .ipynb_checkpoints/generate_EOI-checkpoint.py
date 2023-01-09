from detector_label_EOI import LabelEOI
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from multiprocessing import Pool, freeze_support
import os
import logging
logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
from datetime import timedelta
from config import corridor_isc_map


map_file = pd.read_csv('MAPPINGS_DET_INFO_OCT_2022.csv')



curr_corridor = 'RedBugLakeRd'
time_start ="2022-11-15 00:00:00"
time_end ="2022-12-15 00:00:00"
store_path = f"corridor_label_data/{curr_corridor}-12-2022/"

all_isc = corridor_isc_map[curr_corridor]
all_dets = map_file[map_file.ATSPM_ID.isin(all_isc)]
dets_of_interest = all_dets[(all_dets.phase.isin(['2','6'])) & (all_dets.distanceToStopbar >150) ]
detector_ids = []
for i, each_row in dets_of_interest.iterrows():
    each_signal = each_row['ATSPM_ID']
    channel = each_row['channel']
    detector_ids.append(int(str(each_signal)+ str(channel)))
    
    

if not os.path.exists(store_path):
    os.makedirs(store_path)
    
    

for det_id in detector_ids:
    logging.info(f"**** DOING FOR DET {det_id} ****** ")
    try:
        eoi_gen = LabelEOI(det_id,time_start,time_end )
        final_res = eoi_gen.parallel_process_days()
        if(final_res is not None):
            final_res['DetectorID'] = det_id
            final_res.to_csv(f'{store_path}/{det_id}_{time_start[:10]}_{time_end[:10]}.csv')
            final_res.to_pickle(f'{store_path}/{det_id}_{time_start[:10]}_{time_end[:10]}.pickle')
        else:
            logging.info(f" FAILURE FOR {det_id} due to NO DATA")
    except Exception as e:
        logging.info(f"TOTAL FAILURE FOR {det_id} due to {e}")