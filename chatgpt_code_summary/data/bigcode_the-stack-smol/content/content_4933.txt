# console converter - USD to BGN
# Write a program for converting US dollars (USD) into Bulgarian levs (BGN).
# Round the result to 2 digits after the decimal point. Use a fixed exchange rate between the dollar and the lev: 1 USD = 1.79549 BGN.

USD = float(input())
BGN = round(USD * 1.79549, 2)
print(BGN)
