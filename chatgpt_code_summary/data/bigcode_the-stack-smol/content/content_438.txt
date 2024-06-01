import csv
source_file = "Resources/budget_data.csv"
output_file = "Resources/budget_data_analysis.txt"
#initialize months counter, total income, decrease and increase in revenue amounts
number_of_months = 0 # to track the total number of months
income_total = 0    #variable to hold total income as we iterate through the csv
previous_income = 0 #variable to hold previously eveluated value from csv 
greatest_profit_increase = ["",0] #list to hold the greatest profit increase, inaitialized to lowest value 0
greatest_loss_decrease = ["",1000000000000] #list to hold the greatest loss decrease, inaitialized to highest value
change_in_pl = [] #list to hold change in profit/loss as we iterate through the csv
change_in_income = 0 

#print (revenue_decrease)

with open(source_file) as budget_data:
    csv_reader = csv.DictReader(budget_data)

    for row in csv_reader:

        number_of_months = number_of_months + 1
        #print(row["Profit/Losses"])
        income_total = income_total + int(row["Profit/Losses"])
        #print(row)
        
        #trace the changes in amount
        change_in_income = int(row["Profit/Losses"]) - previous_income
        #print(change_in_income)
        
        #reinitiate the value to the record we completed evaluating
        previous_income = int(row["Profit/Losses"])
        #print(previous_income)
        
        #greatest increase
        if(change_in_income > greatest_profit_increase[1]):
            greatest_profit_increase[0] = row["Date"]
            greatest_profit_increase[1] = change_in_income
        #greatest decrease
        if(change_in_income < greatest_loss_decrease[1]):
            greatest_loss_decrease[0] = row["Date"]
            greatest_loss_decrease[1] = change_in_income
        
        #append to the change_in_pl for sum calculations
        #print(int(row['Profit/Losses']))
        change_in_pl.append(int(row['Profit/Losses']))
    #calculate net profit or loss
    net_profit = sum(change_in_pl)
    #print(net_profit)
print()
print('Financial Anlysis')
print('--------------------------')
print("Total Months: " + str(number_of_months))
print("Total Income: " + "$" + str(net_profit))
print("Greatest Increase in Profits: " + str(greatest_profit_increase[0]) + " $" + str(greatest_profit_increase[1]))
print("Greatest Decrease in Profits: " + str(greatest_loss_decrease[0]) + " $" + str(greatest_loss_decrease[1]))

#write outup to text file
with open(output_file,"w") as results:
    results.write("Total Months: " + str(number_of_months))
    results.write("\n")
    results.write("Total Income: " + "$" + str(net_profit))
    results.write("\n")
    results.write("Greatest Increase in Profits: " + str(greatest_profit_increase[0]) + " $" + str(greatest_profit_increase[1]))
    results.write("\n")
    results.write("Greatest Decrease in Profits: " + str(greatest_loss_decrease[0]) + " $" + str(greatest_loss_decrease[1]))    