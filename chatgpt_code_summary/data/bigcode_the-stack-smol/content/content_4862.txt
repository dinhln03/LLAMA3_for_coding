import pandas as pd
import read_mta_turnstile as t

# This function generally generates a schedule for all stations in the df_top.csv file in a pivot table format.
def find_schedule():
    # Read the stations with highest Toucan scores and select columns relavant
    # to our schedule algorithm
    top_stations = pd.read_csv('df_top.csv')
    top_stations.rename(columns={'name':'STATION'}, inplace = True)
    top_stations1 = top_stations.loc[:,['STATION','toucan_score']]    
    
    # Read the turnstile data and select the columns relavant to schedule algorithm
    turnstile_data = t.read_mta_turnstile(start='20180501', end='20180531')
    turnstile_data1 = turnstile_data.loc[:,['STATION','DATE','TIME','hourly_entries','hourly_exits']]
    
    # Merge the two DataFrames to have hourly entries and exits of stations with top Toucan scores
    turnstile_data2 = turnstile_data1.merge(top_stations1, on = 'STATION')
    
    # Format dataframe and give it "day of week" and "hour of day" values and
    # aggergate hourly entries of each station by date
    schedule = pd.DataFrame(columns = ['STATION', 'hour_of_day', 'day_name', 'hourly_entries'])
    agg = turnstile_data1.groupby(['STATION','DATE','TIME'])[['hourly_entries']].sum().reset_index()
    agg.DATE = pd.to_datetime(agg.DATE, format='%m/%d/%Y')
    agg.TIME = pd.to_datetime(agg.TIME, format='%H:%M:%S')
    agg['day_name'] = agg.DATE.dt.day_name()
    agg['hour_of_day'] = agg.TIME.dt.hour
    
    # Remove 0, 4, and 20 hours of day. Only want 8:00am, 12:00pm, and 4:00pm
    agg = agg[(agg['hour_of_day'] > 5) & (agg['hour_of_day'] < 19 )]
    
    # Segment hours of day into three different shifts: Morning, Afternoon and Evening
    l_times = []
    for h in agg.hour_of_day:
        if int(h) <= 11:
            l_times.append('Morning')
        elif int(h) >= 15:
            l_times.append('Evening')
        else:
            l_times.append('Afternoon')
    agg.hour_of_day = l_times
    
    # For each station in the top station list, this for loop generates a schedule, which identifies 
    # three shifts with the highest number of entries during the week. Volunteers should be at the station
    # at these three shifts.
    for station_name in top_stations1.STATION.unique():
        # Aggergate each station's hourly entries by day of the week, shifts of the day and 
        # pivot the DataFrame as shift vs. day
        hm = agg.loc[agg.STATION == station_name,['hour_of_day','day_name','hourly_entries']]
        hm = hm.groupby(['hour_of_day','day_name'])['hourly_entries'].mean().reset_index()
        hm = hm.pivot(index='hour_of_day',columns='day_name',values='hourly_entries')

        # Calculate three shifts with highest throughput
        sc = hm.stack().nlargest(3).reset_index() 
        sc.rename(columns={0:'hourly_entries'}, inplace=True)
        sc['STATION'] = [station_name]*3
        schedule = schedule.append(sc) # This is a schedule for all stations in the top station list.

    # Make a pivot table of the schedule
    schedule['p'] = [1]*schedule.shape[0]
    schedule_pivot = schedule.pivot_table(index=['STATION'],columns=['day_name','hour_of_day'],values='p')    
    
    return schedule_pivot