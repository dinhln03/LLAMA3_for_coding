import pandas as pd
data=pd.read_csv("C:/Users/user/Documents/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_1068945.csv")  #your raw data obtained from world bank
import pandas as pd
import matplotlib.pyplot as plt

fulldataonly=data.dropna()
listofcountry=fulldataonly['Country Name']
listofcountry=list(listofcountry) 


def findcountryrow(country):
    for i in range(len(data['Country Name'])):
        if data['Country Name'][i]==country:
            return i
    else:
        print("error, country not found")  # find which row is the country

listyear=list(range(1960,2018))
x=[]
y=[]
mydata=[]

#for country in range(len(listofcountry)):
#    for year in listyear:
#        y0=data.loc[findcountryrow(listofcountry[country]),str(year)]
#        y1=data.loc[findcountryrow(listofcountry[country]),str(year+1)]
#        delta=(y1-y0)/y0
#        x.append(y0)
#        y.append(delta)
#        mydata.append([y0,delta])


fulllistofcountry=data['Country Name']
fulllistofcountry=list(fulllistofcountry) 

for country in range(len(fulllistofcountry)):
    for year in listyear: 
        if (pd.notnull(data.loc[country,str(year)]))&(pd.notnull(data.loc[country,str(year+1)])):
            y0=data.loc[country,str(year)]
            y1=data.loc[country,str(year+1)]
            delta=((y1-y0)/y0)*100
            x.append(y0)
            y.append(delta)
            mydata.append([y0,delta])
        
mydata.sort(key=lambda x: x[0])
count=0
GDP, myGDP=[],[]
Growth, myGrowth=[],[]
mysd=[]
naverage=500
averagedatax,averagedatay=[],[]
import statistics as s
for i in range(len(mydata)):
    if count<naverage:
        GDP.append(mydata[i][0])
        Growth.append(mydata[i][1])
        count+=1
    if count==naverage:
        myGDP=s.mean(GDP)
        myGrowth=s.mean(Growth)
        mysd.append(s.stdev(Growth))
        averagedatax.append(myGDP)
        averagedatay.append(myGrowth)
        count=0
        GDP=[]
        Growth=[]
    if i==len(mydata)-1:
        myGDP=s.mean(GDP)
        myGrowth=s.mean(Growth)
        mysd.append(s.stdev(Growth))
        averagedatax.append(myGDP)
        averagedatay.append(myGrowth)
      
    
plt.xscale('log')
plt.xlim(100,200000)
plt.xlabel(' GDP per capita in US dollar',size=15) 
plt.ylabel('GDP growth rate %',size=15)    
plt.title('Dependence of Economic Growth Rate with GDP per capita',size=15)      
plt.scatter(averagedatax,averagedatay)


# histogram=mydata[0:1800]
# per=[]
# for gdp, percentage in histogram:
#     per.append(percentage)
# plt.xlim(-50,60)
# plt.xlabel('GDP per capita Growth %',size=15)
# plt.ylabel('Density Function',size=15)
# plt.title('Economic Growth for different countries for 1960-2018', size=15)
# plt.hist(x=per, bins='auto', density=True)
