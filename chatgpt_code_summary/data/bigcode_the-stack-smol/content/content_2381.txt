'''Ask two student's grade, inform 3 possible averages.
average :
> 7 = Approved
< 7 & > 5 = Recovery
< 5 = Failed
'''
g1 = float(input("Inform the student's first grade: "))
g2 = float(input("Inform the student's second grade: "))
average = (g1 + g2)/2 # how to calculate the avarege grade between two values

if average >= 7:
    print(f"Student with avarege \033[35m{average}\033[m: \033[32mAPPROVED\033[m")
elif average >= 5 and average < 7:
    print(f"Student with avarege \033[35m{average}\033[m: \033[33mRECOVERY\033[m")
else:
    print(f"Student with avarege \033[35m{average}\033[m: \033[31mFAILED\033[m")