import pandas as pd
import matplotlib.pyplot as plt
# import mysql.connector as sql
import time
import copy
import datetime
from datetime import timedelta
import psycopg2
import pandas.io.sql as psql
import os
from datetime import datetime
from multiprocessing import Pool, freeze_support
import numpy as np
from helpers import return_orlando_data_detector, return_bit_mask
import logging
logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
import multiprocessing as mp



class LabelEOI():

    """
    A class to label events for a single detector for a given time period.

    ...

    Attributes
    ----------
    detectord : str
        Detector ID to process
    start_time : str
        start timestamp
    end_time : str
        end timestamp
    window_size: int
        Number of recent cycles to use for computing moving average
    min_prev_data: int
        Minuimun number of previous weeks data to use (2 is previous two weeks)
    no_cores: int
        Number of CPU cores to use
    aggregation_level: int
        Time in seconds for aggrehation
        
    """


    def __init__(self, detector_id, start_time, end_time):
        super().__init__()
        self.no_of_cores = 30
        self.aggregation_level = 300
        self.min_prev_data = 2
        self.window_size = 4
        self.detector_id =detector_id
        self.start_time= start_time
        self.end_time = end_time
        start = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime( end_time, "%Y-%m-%d %H:%M:%S")

        date_generated = [start + timedelta(days=x) for x in range(0, (end-start).days+1)]
        testarray1 = np.array([])
        for date in date_generated:
            testarray1 = np.append(testarray1, date.strftime("%Y-%m-%d %H:%M:%S"))
        self.all_dates = testarray1


        # ALL THRESHOLDS  AND CONSTANTS
        self.DET_HISTORY_VOLUME_THRESHOLD = 10
        self.time_start = '07:00'
        self.time_end = '21:00'
        self.prev_days_to_consider = [7,14,21]
        self.current_weight = 2
        self.history_weight = 1



    def compute_historic_baseline(self, row):

        """
        Computes historic baseline traffic for a given time interval
        Parameters
        ----------
        row : dataframe row
            More info to be displayed (default is None)

        Returns
        -------
        value: float
            traffic volume baseline
        """
        all_prev = ['MA_prev_7','MA_prev_14','MA_prev_21' ]
        value = 0
        count = self.current_weight
        value += self.current_weight*row['MA_curr']
        for prev in all_prev:
            if(prev in row and row[prev]>self.DET_HISTORY_VOLUME_THRESHOLD):
                value+=self.history_weight*row[prev]
                count+=self.history_weight
                
        return value/count


    def return_historic_df(self, time):

        """
        Computes events given historical data
        Parameters
        ----------
        time : str
            timestamp for which historical data to be retrieved
        Returns
        -------
        Flag: boolean
            True if historical data exists
        hist_df: dataframe
            Dataframe with historical data
        """
        count_df = return_orlando_data_detector(self.detector_id,time)

        count_df.drop(columns = ['SignalID','EventParam'], inplace = True)
        count_df.rename(columns={'EventCode': 'Count_curr'}, inplace=True)
        if(count_df.shape[0]<10):
            return False, None
        ll = f"{self.aggregation_level}s"
        count_df= count_df.resample(ll).count()
        count_df['MA_curr'] = count_df.rolling(window=self.window_size ).mean()
        count_df.fillna(method='bfill', inplace = True)
        count_df['time'] = count_df.index.time
#         print(f"Yash DEBUG  {count_df.head(5)}" )
        # prev week 
        prev_count ={}
        all_prev = [7,14, 21]
        for i in  self.prev_days_to_consider:
            time_prev =  (pd.to_datetime(time) - pd.Timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")
            # print(f"time prev {time_prev}")

            count_df_prev = return_orlando_data_detector(self.detector_id,time_prev)
            # print(f"count_df_prev shape {count_df_prev.shape}")
            if(count_df_prev.shape[0]<10):
                pass
            else:
                count_df_prev.drop(columns = ['SignalID','EventParam'], inplace = True)
                count_df_prev.rename(columns={'EventCode': 'Count'}, inplace=True)
                ll = f"{self.aggregation_level}s"
                count_df_prev= count_df_prev.resample(ll).count()
                count_df_prev[f'MA_prev_{i}'] = count_df_prev.rolling(window=self.window_size ).mean()
                count_df_prev[f'time'] =  count_df_prev.index.time
                count_df_prev.fillna(method='bfill', inplace = True)
                prev_count[i]  = count_df_prev  

        all_merged_df =   count_df.copy() 
        all_merged_df['timestamp'] = all_merged_df.index
#         print(f"Yash DEBUG   all_merged_df {all_merged_df.head(5)}" )
        if(len(prev_count) > self.min_prev_data):
            for k,each_prev in   prev_count.items():
#                 all_merged_df = pd.merge(all_merged_df,each_prev, on = 'time', right_index= True)
                all_merged_df = pd.merge(all_merged_df,each_prev, on = 'time')
#                 print(f"Yash DEBUG   all_merged_df  last {all_merged_df.head(5)}" )
            all_merged_df['baseline'] = all_merged_df.apply(lambda x: self.compute_historic_baseline(x), axis = 1)
            all_merged_df['DetectorID'] = self.detector_id
            all_merged_df.set_index('timestamp', inplace = True)
            return True, all_merged_df
        else:
            return False, None

    
    def compute_EOI(self, hist_df):

        """
        Computes events given historical data
        Parameters
        ----------
        hist_df : dataframe
            dataframe with historical data from previous weeks
        Returns
        -------
        eoi_df: dataframe
            Dataframe with events
        """

        final_res = []
    
        prev_dip = 0
        start_volume = 0
        time_start = None
        dip_store = []
        for time_idx, row in hist_df.iterrows():
            if(prev_dip>0):#still in dip
                if(row.Count_curr<row.baseline):
                    curr_volume = row.Count_curr
                    curr_dip = (row.baseline- curr_volume)/(row.baseline)
                    dip_store.append(curr_dip)
                    prev_dip = curr_dip
                else:# dip ended, save EOI
                    time_elapsed = (time_idx - time_start).total_seconds()
                    final_res.append({'avg_reduction':np.mean(dip_store)*100,'time_elapsed': time_elapsed, 'time_start':time_start, 'time_end': time_idx, 'start_volume':start_volume, 'reductions':dip_store})
                    
                    prev_dip = 0
                    dip_store = []

            else:
                if(row.Count_curr<row.baseline):# first dip
                    time_start = time_idx
                    start_volume = row.Count_curr
                    prev_dip = (row.baseline- start_volume)/(row.baseline)
                    dip_store.append(prev_dip)

                    pass
                else:#  no prev dip and curr count high than baseline
                    pass
                
        return pd.DataFrame(final_res)
 


    def return_EOI_df(self, time):
        """
        Computes events for a single day
        Parameters
        ----------
        time : str
            day for which events to compute
        Returns
        -------
        Flah: Boolean
            True if events for this day exist
        eoi_df: dataframe
            Dataframe with events
        """

        try:
            flag, hist_df  = self.return_historic_df(time)
            if(flag):
                hist_df = hist_df.between_time(self.time_start, self.time_end)
                self.history = hist_df
                eoi_df = self.compute_EOI(hist_df)
                
#                check if mask exists
                df_mask  =  return_bit_mask(self.detector_id, time, no_days=1)
                if(df_mask.shape[0]>0):
                    missing = df_mask[df_mask.flag == 0]
                    eoi_df_upd = self.check_if_missing(eoi_df,missing )
                    return True, eoi_df_upd[eoi_df_upd.flag_missing==1]
#                     return True, eoi_df_upd
                else:
                    return False, None
                
                

            else:
                logging.info(f"CANT COMPUTE EOI HISTORIC DATA NOT FOUND or BAD DATA {self.detector_id}  {time} ")
                return False, None

        except Exception as e:
            logging.info(f"FAILED DUE TO ERROR for {self.detector_id}  {time} error--< {e} ")
            return False, None

    def parallel_process_days(self):

        """
        Computes events for all the dates in parallel
        Parameters
        ----------
        Returns
        -------
        eoi_df: dataframe
            events dataframe for all the days
        """


        p = mp.Pool(self.no_of_cores)
        results = p.map(self.return_EOI_df, self.all_dates)
        p.close()
        det_final_df = []
        for each_res in results:
            if(each_res[0]):
                det_final_df.append(each_res[1])
        if(det_final_df):
            return pd.concat(det_final_df)
        else:
            return None
        
        
    def check_if_missing(self, eoi_df, missing_df):
        
        """
        Filters events with missing data
        Parameters
        ----------
        eoi_df : dataframe
            dataframe with events of interest

        missing_df: dataframe
            missing data bit mask dataframe

        Returns
        -------
        eoi_df: dataframe
            events dataframe where events with missing data is filtered out
        """

        flag_missing = []
        for i, each_row in eoi_df.iterrows():
            start_ts = each_row.time_start - pd.Timedelta(seconds = 60)
            end_ts = each_row.time_end + pd.Timedelta(seconds = 60)
            miss_count = missing_df[(missing_df.index>=start_ts) & (missing_df.index<=end_ts)].shape[0]
            if(miss_count>0):
                flag_missing.append(0)
            else:
                flag_missing.append(1)

        eoi_df['flag_missing'] = flag_missing
        return eoi_df
    
    
    
        

