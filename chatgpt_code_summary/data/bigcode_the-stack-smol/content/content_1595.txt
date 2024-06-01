# Charlie Conneely
# Score Keeper

from player import Player

ranks_file = "rankings.txt"

class ScoreKeeper:

    def __init__(self):
        self.ranks = []

    """ 
    Check if player score ranks against scores in rankings.txt
    """
    def check_ranking(self, p):
        self.populate_ranks_array(ranks_file)
        # check score against rankings
        top5 = self.compare_score(p)
        if top5:
            print("Well Done! You ranked Top 5!")
            print("\nNew Rankings:")
            for i in self.ranks:
                print(i.name + " - " + str(i.score))
            self.append_file(ranks_file)
        else:
            print("Sorry, your score didn't rank top 5!")
            print("\nCurrent Rankings:")
            for i in self.ranks:
                print(i.name + " - " + str(i.score))
        # Clear ranks array
        self.ranks = []

    """
    Append ranks file with new score
    """
    def append_file(self, rfile):
        with open(rfile, 'w') as file:
            for p in self.ranks:
                file.write(str(p.name) + " " + str(p.score) + "\n")

    """
    Check if score beats that of any currently ranked players
    If true - Add player to rankings, resort array, pop last item from the end. 

    returns Boolean 
    """
    def compare_score(self, player):
        does_rank = False
        for p in self.ranks:
            if (int(player.score) > int(p.score)):
                does_rank = True
        if does_rank:
            self.ranks.append(player)
            # sort ranks array by scores
            self.ranks.sort(key=lambda p: int(p.score), reverse=True)
            # remove the last item
            self.ranks.pop()
        return does_rank

    """ 
    Populate local array with scores from txt file
    """
    def populate_ranks_array(self, scores_file):
        with open(scores_file) as f:
            for line in f:
                (n, s) = line.split()
                self.ranks.append(Player(n,s))

