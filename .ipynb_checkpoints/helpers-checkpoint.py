import psycopg2
import pandas.io.sql as psql
import pandas as pd

def return_orlando_data_detector(DetectorID, time1, no_days=1):
    """ 
    Returns Dataframe describing the behaviour of signals at a Intersection SignalID for a Day between 8AM to 8PM .
    """
    conn=psycopg2.connect(host="localhost", database="postgres", user="postgres", password="postgres")
#     testtime = date1

    time1 = (pd.to_datetime(time1)).strftime("%Y-%m-%d %H:%M:%S")
#     time2 = (pd.to_datetime(time2)).strftime("%Y-%m-%d %H:%M:%S")
    time2 = (pd.to_datetime(time1) + pd.Timedelta(days=no_days)).strftime("%Y-%m-%d %H:%M:%S")
    
    signalID1 = str(DetectorID)[:4]
    event_param = int(str(DetectorID)[4:])
#     print(f" signalid {signalID1} eventparam {event_param}")
    sql_command = "SELECT * FROM atspmrawv2 where signalid = '{}'  AND timestamp >= '{}' AND timestamp <= '{}' AND eventcode = {} AND eventparam={}".format(
        signalID1, time1, time2, 82,event_param)

#     print(f"sql_command {sql_command}")
    df = psql.read_sql(sql_command, conn)
    df.columns = ['SignalID', 'Timestamp', 'EventCode', 'EventParam']
    df['Timestamp'] = pd.to_datetime(df['Timestamp'].values, infer_datetime_format=True)
    df = df.set_index('Timestamp')
    df['EventParam'] = df['EventParam'].astype(int)
    
    start_time = time1[:10] + ' ' +'00' + ':00:01'
    end_time = time1[:10] + ' ' + '23' + ':59:59'


    new_row = pd.Series(data={'SignalID':signalID1, 'EventCode':81, 'EventParam':event_param}, name=pd.to_datetime(start_time))
    df = df.append(new_row, ignore_index=False)


    new_row = pd.Series(data={'SignalID':signalID1, 'EventCode':81, 'EventParam':event_param }, name=pd.to_datetime(end_time))
    df = df.append(new_row, ignore_index=False)
    
    
    df.sort_index(inplace = True)
    return df



def return_bit_mask(DetectorID, time1, no_days=1):
    """ 
    Returns Dataframe describing the behaviour of signals at a Intersection SignalID for a Day between 8AM to 8PM .
    """
    conn=psycopg2.connect(host="localhost", database="postgres", user="postgres", password="postgres")
#     testtime = date1

    time1 = (pd.to_datetime(time1)).strftime("%Y-%m-%d %H:%M:%S")
#     time2 = (pd.to_datetime(time2)).strftime("%Y-%m-%d %H:%M:%S")
    time2 = (pd.to_datetime(time1) + pd.Timedelta(days=no_days)).strftime("%Y-%m-%d %H:%M:%S")
    
    signalID1 = str(DetectorID)[:4]
#     event_param = int(str(DetectorID)[4:])
#     print(f" signalid {signalID1} eventparam {event_param}")
    sql_command = "SELECT * FROM atspm_bit_mask where signalid = '{}'  AND timestamp >= '{}' AND timestamp <= '{}'".format(
        signalID1, time1, time2)

#     print(f"sql_command {sql_command}")
    df = psql.read_sql(sql_command, conn)
    df.columns = ['SignalID', 'Timestamp', 'flag']
    df['Timestamp'] = pd.to_datetime(df['Timestamp'].values, infer_datetime_format=True)
    df = df.set_index('Timestamp')
    
    
    start_time = time1[:10] + ' ' +'00' + ':00:01'
    end_time = time1[:10] + ' ' + '23' + ':59:59'


    new_row = pd.Series(data={'SignalID':signalID1,'flag':0}, name=pd.to_datetime(start_time))
    df = df.append(new_row, ignore_index=False)


    new_row = pd.Series(data={'SignalID':signalID1, 'flag':0 }, name=pd.to_datetime(end_time))
    df = df.append(new_row, ignore_index=False)
    
    
    df.sort_index(inplace = True)
    
    return df.resample('1min').count()


def return_det_waveform(DetectorID, start_ts,end_ts ):
    """ 
    Returns Dataframe describing the behaviour of signals at a Intersection SignalID for a Day between 8AM to 8PM .
    """
    conn = psycopg2.connect(host="localhost", database="postgres", user="postgres", password="postgres")
#     testtime = date1

    time1 = (pd.to_datetime(start_ts)).strftime("%Y-%m-%d %H:%M:%S")
#     time2 = (pd.to_datetime(time2)).strftime("%Y-%m-%d %H:%M:%S")
    time2 = (pd.to_datetime(end_ts)).strftime("%Y-%m-%d %H:%M:%S")
    
    signalID1 = str(DetectorID)[:4]
    event_param = int(str(DetectorID)[4:])
#     print(f" signalid {signalID1} eventparam {event_param}")
    sql_command = "SELECT * FROM atspmrawv2 where signalid = '{}'  AND timestamp >= '{}' AND timestamp <= '{}' AND eventcode = {} AND eventparam={}".format(
        signalID1, time1, time2, 82,event_param)

#     print(f"sql_command {sql_command}")
    df = psql.read_sql(sql_command, conn)
    df.columns = ['SignalID', 'Timestamp', 'EventCode', 'EventParam']
    df['Timestamp'] = pd.to_datetime(df['Timestamp'].values, infer_datetime_format=True)
    df = df.set_index('Timestamp')
    df['EventParam'] = df['EventParam'].astype(int)
    
#     start_time = time1[:10] + ' ' +'00' + ':00:01'
#     end_time = time1[:10] + ' ' + '23' + ':59:59'


    new_row = pd.Series(data={'SignalID':signalID1, 'EventCode':81, 'EventParam':event_param}, name=pd.to_datetime(start_ts))
    df = df.append(new_row, ignore_index=False)


    new_row = pd.Series(data={'SignalID':signalID1, 'EventCode':81, 'EventParam':event_param }, name=pd.to_datetime(end_ts))
    df = df.append(new_row, ignore_index=False)
    
    
    df.sort_index(inplace = True)
    df.drop(columns = ['SignalID','EventParam'], inplace = True)
    df.rename(columns={'EventCode': 'Count_curr'}, inplace=True)
    df= df.resample('5s').count()
    
    df.iloc[0]['Count_curr'] -=1
    df.iloc[-1]['Count_curr'] -=1
    return df
