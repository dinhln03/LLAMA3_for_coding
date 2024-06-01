dist = int(input())
clubs = int(input())

clublist = []
strokes = [0]*(dist+1)

for i in range(clubs):
    clublist.append(int(input()))

for i in range(1, dist+1):
    min_club = 1000000

    for j in range(clubs):
        if i - clublist[j] >= 0:
            min_club = min(min_club, strokes[i-clublist[j]]+1)

    strokes[i] = min_club

if strokes[dist] != 1000000:
    print(f"Roberta wins in {strokes[dist]} strokes.")
else:
    print("Roberta acknowledges defeat.")
