import os
import time
import requests
from bs4 import BeautifulSoup
import datetime
from twilio.rest import Client
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math

class Data():
    def __init__(self,link): # Automatically stores the data from parsed link as the object's attribute 
        self.data = self.update(link)
        self.prev = None
        
    def update(self,link): # Parses the site's HTML code
        result = requests.get(link)
        soup = BeautifulSoup(result.content,'lxml')
        return soup


class Hawaii(Data):
    def __init__(self,link): # Automatically stores the data from parsed link as the object's attribute (Same constructor as Class Data)
        super().__init__(link)
        

    def do_sms(self,numbers): # Creates SMS notification with the COVID data for each island
        for number in numbers:
                smsNotification.notify(number,self.get_data()) 
                smsNotification.save_number(numbers) 


    def get_data(self): # Returns the data from today:
        # Gathering all the data
        today = self.get_dataframe()
        order = ["Total cases","Hawai’i","Oahu","Kaua’i","Maui","Pending","Residents diagnosed outside of Hawai‘i","Required Hospitalization","Hawaii deaths","Lanai","Molokai"]
        data = today.to_numpy()[0]

        message = ""
        for index in range(len(order)):
            diff = int(data[index+1]) - int(self.prev[index])
            if diff >= 0:
                diff = "+" + str(diff)

            else:
                diff = "-" + str(diff)
                
            line = order[index] + ": " + str(data[index+1]) + " (" + diff + ") \n"
            message = message + line
        return message
            

    def get_dataframe(self): # Returns the data structure for today's data
        date = self.get_date()
        
        names = self.data.find_all('span',{'class': 'label'})
        values = self.data.find_all('span',{'class': 'value'})

        df = pd.DataFrame()
        # Formats the names and values
        for i in range(len(names)):
            names[i] = names[i].text.replace(":","")
            values[i] = int(values[i].text.replace("§","").replace("†","").replace("‡","").replace("*","").split(" ")[0])

        # Orders the names and values in the order of the .csv
        order = ["Total cases","Hawai’i","Oahu","Kaua’i","Maui","Pending","Residents diagnosed outside of Hawai‘i","Required Hospitalization","Hawaii deaths","Lanai","Molokai"]
        namesOrdered = ["","","","","","","","","","",""]
        valuesOrdered = ["","","","","","","","","","",""]
        for i in range(len(order)):
            for j in range(len(names)):
                if order[i] == names[j]:
                    namesOrdered[i] = names[j]
                    valuesOrdered[i] = values[j]
                    
        dfNew = pd.DataFrame({
            "Date": date, 
            namesOrdered[0]: valuesOrdered[0],
            namesOrdered[1]: valuesOrdered[1],
            namesOrdered[2]: valuesOrdered[2],
            namesOrdered[3]: valuesOrdered[3],
            namesOrdered[4]: valuesOrdered[4],
            namesOrdered[5]: valuesOrdered[5],
            namesOrdered[6]: valuesOrdered[6],
            namesOrdered[7]: valuesOrdered[7],
            namesOrdered[8]: valuesOrdered[8],
            namesOrdered[9]: valuesOrdered[9],
            namesOrdered[10]: valuesOrdered[10],
            }, index = [0])

        return dfNew

            
    def get_date(self): # Returns the update date of the data in the datetime format
        # Formatting
        date = self.data.find_all('dd',{'class': 'small'})
        date = date[0].text[33:]
        date = datetime.datetime.strptime(date, '%B %d, %Y')
        date = str(date.date())
        return date


    def do_update(self): # Does an update if the history.txt is not updated
        # If the history.txt is not updated relevant to the available data, the update proceeds 
        if self.check_update() == False:
            # Checks if the data on the website is updated; Loops the program until the data is updated
            if self.get_date() != str(datetime.date.today()):
                print("Data not updated. Sleeping for 1 minute.\n")
                time.sleep(60)
                print("Rechecking.\n")
                self.do_update()
                return

            dfOld = pd.read_csv('data.csv', index_col = False)
            dfOld = dfOld.append(self.get_dataframe())
            dfOld.to_csv('data.csv', index=False)
            
            file = "phoneNumbers.txt"
            numbers = open(file,"r")
            # Checks if there are any recently saved numbers
            if(os.stat(file).st_size) == 0:
                print("No recent phone numbers found. Please enter your phone numbers including area code and no dashes into the phoneNumbers.txt file, with each phone number tabbed.")
                return
            
            else:
                paste=[]
                for line in numbers:
                    paste.append(line.replace("\n",""))
                self.do_sms(paste)
            
    
    def check_update(self): # Checks when the data.csv was last updated; Returns True if already updated today; Returns False if not
        file = "data.csv"
        history = open(file,'r')

        # Checks if the file is empty ahead of time to prevent crash and formats the document if it is empty
        if(os.stat(file).st_size) == 0:
            File.append_file(file, "Date,Total cases,Hawai’i,Oahu,Kaua’i,Maui,Pending,Residents diagnosed outside of Hawai‘i,Required Hospitalization,Hawaii deaths,Lanai,Molokai")
            return False
        
        # Finds the last line in the .txt
        for line in history:
            pass
        lastLine = line
        history.close()
        
        # Checks if the last updated date was today
        if self.get_date() in lastLine:
            return True
        
        # Formats the data from .csv to a Python list
        lastLine = lastLine.split(",")
        lastLine.pop(0)
        self.prev = lastLine
        return False


class smsNotification:
    @staticmethod
    def notify(toNumber,message): # Creates SMS notifications; (IMPORTANT) List your Twilio account sid, auth token, and phone number in the token.txt file by tabbing each token
        f = open('token.txt','r')
        accountSid, authToken, fromNumber = f.readlines()
        
        accountSid = accountSid.replace("\n","")
        authToken = authToken.replace("\n","")
        fromNumber = fromNumber.replace("\n","")
        
        client = Client(accountSid, authToken)
        client.messages.create(to=toNumber,from_=fromNumber,body=message)
        print("SMS sent")


    @staticmethod
    def save_number(paste): # Saves the recently used phone number on file
        numbers = open("phoneNumbers.txt","w")
        for number in paste:
            numbers.write(str(number) + "\n")

class Graph:
    @staticmethod
    def display_graph(islands,scope=[],graphType='Cases'): # Displays data in a graph format where islands is a list containing the statistics that should be included, the scope is the time period, and the graph type differentiates between cases vs change in cases
        if graphType == 'Cases': # For graphing cases
            df = pd.read_csv('data.csv', index_col = False)

        else: # For graphing the change in cases
            df = App.get_df_change()
            if scope[0] == 0: # Adjust the scope to not include the first entry since there is no change observerd on that day
                scope[0] = 1

        plt.figure(figsize=(8,8))
        min_ = -1
        max_ = -1
        for island in islands: # Plots data for each island on the same plot
            plt.plot(df["Date"], df[island], label = island)
            if graphType == 'Cases': 
                if scope != []:
                    if min_ == - 1 and max_ == -1:
                        min_ = df[island].get(scope[0])
                        max_ = df[island].get(scope[1])

                    else:
                        minNow = df[island].get(scope[0])
                        maxNow = df[island].get(scope[1])
                        if minNow < min_:
                            min_ = minNow

                        elif maxNow > max_:
                            max_ = maxNow
                    
                    plt.ylim(min_,max_)        
        
        title = "COVID Cases vs Time"
        if scope != []: # Scales the interval to the scope
            intervals = (scope[1]-scope[0])/4
            if intervals < 1:
                intervals = 1
                    
            plt.gca().xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=math.floor(intervals)))
            plt.xlim(scope[0],scope[1])  
            title = title + " (" + df["Date"].get(scope[0]) + " to " + df["Date"].get(scope[1]) + ")" # Title formatting

        else:
            plt.gca().xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=30)) # Automatically sets the scale if there is no scale

        
        plt.xlabel("Date")

        if graphType == 'Cases':
            plt.ylabel("# of Cases")

        else:
            plt.ylabel("Change in Cases")
            title = title.replace("COVID Cases","Change in COVID Cases")

        plt.title(title)
        plt.grid()
        plt.legend()
        plt.show()

            
class File:
    @staticmethod # Appends the passed file with the passed text
    def append_file(file,text):
        history = open(file,'a')
        history.write(text)
        history.close()

class App:
    @staticmethod
    def get_df(): # Returns the dataframe
        return pd.read_csv('data.csv', index_col = False)

    def format_date(date): # Receives the data and returns the index of the date
        df = pd.read_csv('data.csv', index_col = False)
        for x in range(len(df["Date"])):
            if df["Date"][x] == date:
                return x

    def get_last_index(): # Returns the index of the last element in the dataframe
        df = pd.read_csv('data.csv', index_col = False)

        for index in range(len(df["Date"])):
            pass
        return index

    def get_df_change(): # Returns the change over time dataframe
        df = pd.read_csv('data.csv', index_col = False)

        dates = df['Date']
        dates = pd.DataFrame(dates) # Save datafrmae
        df = df.drop(columns=['Date']) # Must drop the dates since the dataframe diff() function will produce an unideal dataframe otherwise
        dfDiff = df.diff()
        dfDiff = dates.join(dfDiff) # Rejoin dataframes
        dfDiff = dfDiff.iloc[1:] # Get rid of bad data from first row

        return dfDiff
                    


if __name__ == "__main__":
    data=Hawaii("https://health.hawaii.gov/coronavirusdisease2019/")
    data.do_update()
    lastIndex = App.get_last_index()
    firstIndex = lastIndex - 6 # The scope is automatically set to the past 7 days
    
    Graph.display_graph(["Total cases"],[firstIndex,lastIndex],"Change") # Displays total cases over the past seven days



        













