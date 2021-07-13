import pandas as pd
import numpy as np
# import sys
# import scipy.io as sio

from datetime import date as dt # replace the obspy version of UTCDateTime with this. 
#from obspy import UTCDateTime 
# a timestamp is defined (POSIX time) as the number of seconds elapsed since 1st of January 1970 


# take in a CATALOG with clustering data,
# add TIME to it-- in year frac 

def calc_YearFloat_HourFloat(cat):
    n_evts = len(cat)
    hourFrac_vec = []
    yearFrac_vec = []
    for ind in range(n_evts):
        dtobj = cat.datetime.iloc[ind] 
        # print(dtobj)
        tt = dtobj.timetuple()
        dayofyear = tt.tm_yday
        # print(dayofyear)

        year = dtobj.year
        hour = dtobj.hour
        minu = dtobj.minute
        sec = dtobj.second
        microsec = dtobj.microsecond

        # print(hour,minu,sec,microsec)

        dayDec = (hour*60*60 + minu*60 + sec + microsec/1e6)/(24*60*60) 
        # calculate in seconds/seconds_in_day
        hourDec = (minu*60 + sec + microsec/1e6 )/ (60*60) # seconds/seconds_in_hour
        yearDec = (dayofyear+dayDec)/365.25
        #delta = dayofyear/365.25 - yearFrac

        hourFrac = hour+hourDec
        yearFrac = year+yearDec
        
        hourFrac_vec.append(hourFrac)
        yearFrac_vec.append(yearFrac)
        
#         print(yearFrac)
#         print(delta)
#         print(dayFrac)
#         print(yearFloat)
#         print(hourFrac)
#         print(hourFloat)


    return np.asarray(yearFrac_vec),np.asarray(hourFrac_vec)


# def calc_YearFrac_UTCts(cat):
#     n_evts = len(cat)
#     UTCtimestamp = np.zeros([n_evts])
#     YearFrac = np.zeros([n_evts])

#     for ind in np.arange(n_evts):
#         yr=int(cat.year.iloc[ind]); mo=int(cat.month.iloc[ind]); dy=int(cat.day.iloc[ind]) 
#         hr=int(cat.hour.iloc[ind]); mn=int(cat.mn.iloc[ind]); sc=int(cat.sec.iloc[ind]) 
#         # split sec at '.' to get microsec.. later
        
#         stamp = int(float(UTCDateTime(year=yr, month=mo, day=dy, hour=hr, minute=mn, second=sc)))
#         UTCtimestamp[ind] = stamp
        
#         yearfrac = yr + (float(UTCDateTime(year=yr, month=mo,day=dy,hour=hr,minute=mn, second=sc).julday/365.25))
#         YearFrac[ind] = yearfrac
        
#     return YearFrac, UTCtimestamp

                                  
# print('----------------')  
# print(TimeStamp[1])
# print(type(TimeStamp[1]))
# print(len(TimeStamp))
# print(TimeStamp[0],TimeStamp[-1])


