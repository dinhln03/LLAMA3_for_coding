# import datetime, time
from typing import List
from datetime import datetime, timedelta
import pytz
import os
from puffmarker.domain.datapoint import DataPoint
from puffmarker.input.import_stream_processor_inputs import load_data, load_data_offset

data_dir = '/home/nsaleheen/data/rice_ema_puffmarker_activity_loc/'
# data_dir = '/home/nsaleheen/data/RICE_data/without_raw_data/'

smoking_self_report_file = 'SMOKING+SELF_REPORT+PHONE.csv'
activity_type_file = 'ACTIVITY_TYPE+PHONE.csv'
puffmarker_smoking_epi_cloud_file = 'PUFFMARKER_SMOKING_EPISODE+PHONE.csv'

# streamprocessor_puffmarker_smoking_epi_file = 'streamprocessor.puffMarker.smoking.episode.rip.wrist.combine.csv'
streamprocessor_puffmarker_smoking_epi_file = 'org.md2k.streamprocessor+PUFFMARKER_SMOKING_EPISODE+PHONE.csv'
# streamprocessor_puffmarker_smoking_epi_file = 'puffmarker_streamprocessor.csv'

ema_random_file = 'EMA+RANDOM_EMA+PHONE.csv'
ema_smoking_file = 'EMA+SMOKING_EMA+PHONE.csv'
ema_end_of_day_file = 'EMA+END_OF_DAY_EMA+PHONE.csv'
ema_stressed_file = 'EMA+STRESS_EMA+PHONE.csv'

tz = pytz.timezone('US/Central')
print(tz)


# unix time to '2017-11-01 15:52:00'
def unixtime_to_datetime_pre(timestamp):
    timestamp = timestamp / 1000
    dt = datetime.fromtimestamp(timestamp, tz).strftime('%m-%d-%Y %H:%M:%S')
    return dt


def unixtime_to_datetime(timestamp):
    timestamp = timestamp / 1000
    dt = datetime.fromtimestamp(timestamp, tz).strftime('%m/%d %H:%M:%S')
    return dt


# unix time to  '2017-11-01 15:52:00' -> '2017-11-01'
def unixtime_to_date(timestamp):
    dt = unixtime_to_datetime(timestamp)
    return dt.split(' ')[0]


# unix time to  '2017-11-01 15:52:00' -> '15:52:00'
def unixtime_to_time(timestamp):
    dt = unixtime_to_datetime(timestamp)
    return dt.split(' ')[1]


# unix time to '15*52' in minutes
def unixtime_to_timeOfDay(timestamp):
    tm = unixtime_to_time(timestamp)
    toks = tm.split(':')
    h = int(toks[0])
    m = int(toks[1])
    timeOfday = h * 60 + m
    return timeOfday


ut = 1512506705814  # 1386181800

print(unixtime_to_datetime(ut))
print(unixtime_to_date(ut))
print(unixtime_to_time(ut))
print(unixtime_to_timeOfDay(ut))

# timezone = datetime.timezone(datetime.timedelta(milliseconds=offset))
# ts = datetime.datetime.fromtimestamp(ts, timezone)

import json


def get_fileName(cur_dir, file_sufix):
    filenames = [name for name in os.listdir(cur_dir) if name.endswith(file_sufix)]
    # print(file_sufix + ':' + str(filenames))
    if len(filenames) > 0:
        return filenames[0]
    else:
        return None


def get_EMA_data(cur_dir, filename):
    if filename is None:
        return []

    fp = open(cur_dir + filename)
    file_content = fp.read()
    fp.close()

    lines = file_content.splitlines()
    data = []
    for line in lines:
        if len(line) > 1:
            ts, offset, sample = line.split(',', 2)
            # start_time = int(ts)
            start_time = int(float(ts)) / 1000.0
            start_time = datetime.fromtimestamp(start_time)
            offset = int(offset)
            sample = sample[1:-1]
            data.append([start_time, offset, sample])
    return data


# random ema + stressed EMA
# sample = (#smoked, from_time, to_time); eg: "2 hrs - 4 hrs" one cig smoked (1, 2*60*60*1000, 4*60*60*1000)
def get_random_EMA(cur_dir, filename) -> List[DataPoint]:
    emas = get_EMA_data(cur_dir, filename)
    data = []
    for ema in emas:
        d = ema[2]
        jsn_file = json.loads(d)
        status = jsn_file['status']
        #         print(jsn_file['status'])
        if status == 'COMPLETED':
            is_smoked = jsn_file['question_answers'][32]['response'][0]

            if is_smoked == 'Yes':
                nSmoked = jsn_file['question_answers'][33]['response'][0]
                if int(nSmoked) == 1:
                    nQI = 34
                else:
                    nQI = 35
                # options: ["0 - 2 hrs", "2 hrs - 4 hrs", "4 hrs - 6 hrs", "6 hrs - 8 hrs", "8 hrs - 10 hrs", "10 hrs - 12 hrs", "More than 12 hrs"]
                howlong_ago = jsn_file['question_answers'][nQI]['response']
                sample = [int(nSmoked)]
                for hla in howlong_ago:
                    hla = str(hla)
                    if hla in ["More than 12 hrs"]:
                        sample.extend([12 * 60 * 60 * 1000, 24 * 60 * 60 * 1000])
                        continue
                    st = hla.split('-')[0]
                    et = hla.split('-')[1]
                    st = st.split(' ')[0]
                    st = int(st.strip()) * 60 * 60 * 1000
                    et = et.strip().split(' ')[0]
                    et = int(et.strip()) * 60 * 60 * 1000
                    sample.extend([st, et])

                # print([ema[0], ema[1], nSmoked, howlong_ago, sample])
                # data.append([ema[0], ema[1], int(nSmoked)])
                data.append(DataPoint(start_time=ema[0], offset=ema[1], sample=sample))
    return data

# Confirm refute
def get_smoking_EMA(cur_dir, filename) -> List[DataPoint]:

    emas = get_EMA_data(cur_dir, filename)
    data = []
    for ema in emas:
        d = ema[2]
        jsn_file = json.loads(d)
        status = jsn_file['status']
        if status == 'COMPLETED':
            is_smoked = jsn_file['question_answers'][0]['question_answer'][0:3]
            #             print(is_smoked)
            if is_smoked.lower() == 'yes':
                data.append(DataPoint(start_time=ema[0], offset=ema[1], sample=1))
                # data.append([ema[0], ema[1], 1])
            else:
                data.append(DataPoint(start_time=ema[0], offset=ema[1], sample=0))
                # data.append([ema[0], ema[1], 0])
    return data


def get_smoking_self_report(cur_dir, filename) -> List[DataPoint]:
    emas = get_EMA_data(cur_dir, filename)
    data = []
    for ema in emas:
        d = ema[2]
        jsn_file = json.loads(d)
        status = jsn_file['message']
        if 'YES' in status:
            #             print(status)
            data.append(DataPoint(start_time=ema[0], offset=ema[1], sample=1))
            # print(ema)
            # data.append([ema[0], ema[1], status])
    return data


cur_dir = data_dir + '2007/'


# emas = get_smoking_self_report(cur_dir, get_fileName(cur_dir, smoking_self_report_file))
# print(emas)

# emas = get_smoking_EMA(cur_dir, get_fileName(cur_dir, ema_smoking_file))
# print(emas)
# emas = get_random_EMA(cur_dir, get_fileName(cur_dir, ema_stressed_file))
# print(emas)
# emas = get_random_EMA(cur_dir, get_fileName(cur_dir, ema_random_file))
# print(emas)


def get_RICE_PILOT_EMAs(pid):
    cur_dir = data_dir + pid + '/'

    # smoking_epis = load_data(cur_dir + get_fileName(cur_dir, streamprocessor_puffmarker_smoking_epi_file))
    smoking_epis = load_data_offset(cur_dir + get_fileName(cur_dir, streamprocessor_puffmarker_smoking_epi_file))
    smoking_selfreport = get_smoking_self_report(cur_dir, get_fileName(cur_dir, smoking_self_report_file))

    smoking_emas = get_smoking_EMA(cur_dir, get_fileName(cur_dir, ema_smoking_file))
    random_emas = get_random_EMA(cur_dir, get_fileName(cur_dir, ema_random_file))
    stressed_emas = get_random_EMA(cur_dir, get_fileName(cur_dir, ema_stressed_file))

    sup_sr = [0] * len(smoking_epis)
    sup_cr = [0] * len(smoking_epis)
    sup_ema = [0] * len(smoking_epis)

    for i, epi in enumerate(smoking_epis):
        for sr in smoking_selfreport:
            time_diff = (sr.start_time - epi.start_time).total_seconds()
            if (time_diff > -1800 and time_diff < 1800):
                sup_sr[i] = 1
                break
        for sr in smoking_emas:
            time_diff = (sr.start_time - epi.start_time).total_seconds()
            if (time_diff > -600 and time_diff < 1800):
                sup_cr[i] = 1
                break

        for re in random_emas:
            st = re.start_time - timedelta(milliseconds=re.sample[2])
            et = re.start_time - timedelta(milliseconds=re.sample[1])
            if (epi.start_time >= st and epi.start_time <= et):
                sup_ema[i] = 1
                break
        for re in stressed_emas:
            st = re.start_time - timedelta(milliseconds=re.sample[2])
            et = re.start_time - timedelta(milliseconds=re.sample[1])
            if (epi.start_time >= st and epi.start_time <= et):
                sup_ema[i] = 1
                break

    sup = [sup_sr[i] * 100 + sup_cr[i] * 10 + sup_ema[i] for i in range(len(sup_ema))]

    print('se=' + str(len(smoking_epis)) + ' : sup sr = ' + str(sum(sup_sr)) + ' : sup cr = ' + str(
        sum(sup_cr)) + ' : sup ema = ' + str(sum(sup_ema)))
    non_sup = len([v for v in sup if v == 0])
    print('Supported : Not supported = ' + str(len(sup) - non_sup) + ' : ' + str(non_sup))
    # print(sup)
    # print(len(smoking_selfreport))
    # print(len(smoking_emas))
    # print(len(random_emas))
    # print(len(stressed_emas))

    # print(smoking_epis)
    # print(smoking_emas)
    # print(smoking_selfreport)
    # print(random_emas)
    # print(stressed_emas)
    #
# , "2008", "2010", "2011", "2012"

pids = ["2006", "2007", "2009", "2013", "2014", "2015", "2016", "2017"]
# for pid in pids:
#     print('-----------' + pid + '---------------------------')
#     get_RICE_PILOT_EMAs(pid)
get_RICE_PILOT_EMAs('2006')

# -----------2006---------------------------
# se=25 : sup sr = 19 : sup cr = 18 : sup ema = 4
# Supported : Not supported = 21 : 4
# -----------2007---------------------------
# se=6 : sup sr = 5 : sup cr = 6 : sup ema = 0
# Supported : Not supported = 6 : 0
# -----------2009---------------------------
# se=32 : sup sr = 14 : sup cr = 30 : sup ema = 10
# Supported : Not supported = 30 : 2
# -----------2013---------------------------
# se=113 : sup sr = 72 : sup cr = 108 : sup ema = 49
# Supported : Not supported = 113 : 0
# -----------2014---------------------------
# se=44 : sup sr = 6 : sup cr = 43 : sup ema = 23
# Supported : Not supported = 44 : 0
# -----------2015---------------------------
# se=0 : sup sr = 0 : sup cr = 0 : sup ema = 0
# Supported : Not supported = 0 : 0
# -----------2016---------------------------
# se=0 : sup sr = 0 : sup cr = 0 : sup ema = 0
# Supported : Not supported = 0 : 0
# -----------2017---------------------------
# se=8 : sup sr = 0 : sup cr = 5 : sup ema = 2
# Supported : Not supported = 5 : 3