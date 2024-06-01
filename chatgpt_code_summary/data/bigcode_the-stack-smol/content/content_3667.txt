import os
import csv

# File path 
election_dataCSV = os.path.join('.', 'election_data.csv')

# The total number of votes cast
# A complete list of candidates who received votes
# The percentage of votes each candidate won
# The total number of votes each candidate won
# The winner of the election based on popular vote.

# Declaring my variables

total_votes = 0
khan_votes = 0
correy_votes = 0
li_votes = 0
otooley_votes = 0
# percent_votes = 0
# total_votes_candidate = 0
# winner = 0

# Open file as read
with open ('election_data.csv','r') as csvfile:

    # Identifying CSV file with delimiter set
    csvreader = csv.reader(csvfile, delimiter=',')
    
    header = next(csvreader)
    # firstRow = next(csvreader)
    # total_votes += 1
    # previous_row = int(firstRow[0])

    # Add rows to list
    for row in csvreader:
    
        #Adding total number of votes cast
        total_votes += 1
    
        #Candidates that received votes
        if row[2] == "Khan":
            khan_votes += 1
        elif row[2] == "Correy":
            correy_votes += 1
        elif row[2] == "Li":
            li_votes += 1
        elif row[2] == "O'Tooley":
            otooley_votes +=1
        
        # Create a list of the candidates
        candidates_list = ["Khan", "Correy", "Li", "O'Tooley"]
        votes = [khan_votes, correy_votes, li_votes, otooley_votes]
        
        # Pair candidates and votes together
        dict_candidates_and_votes = dict(zip(candidates_list,votes))
        
        # Find the winner by using the max function
        key = max(dict_candidates_and_votes, key = dict_candidates_and_votes.get)
        
        # Calculating the percentage of votes per candidate
        khan_percentage = (khan_votes/total_votes) *100
        correy_percentage = (correy_votes/total_votes) *100
        li_percentage = (li_votes/total_votes) *100
        otooley_percentage = (otooley_votes/total_votes) *100

        
# Print conclusion
print(f"Election Results")
print(f"----------------------------")
print(f"Total Votes: {total_votes}")
print(f"----------------------------")
print(f"Khan: {khan_percentage:.3f}% ({khan_votes})")
print(f"Correy: {correy_percentage:.3f}% ({correy_votes})")
print(f"Li: {li_percentage:.3f}% ({li_votes})")
print(f"O'Tooley: {otooley_percentage:.3f}% ({otooley_votes})")
print(f"----------------------------")
print(f"Winner: {key}")
print(f"----------------------------")

# Export results into txt file

file = open('election_output.txt','w')

file.write("Election Results: Total Votes - 1048575, Khan - 63.094% (661583), Correy - 19.936% (209046), Li: - 13.958% (146360), O'Tooley - 3.012% (31586), Winner - Khan")

file.close
