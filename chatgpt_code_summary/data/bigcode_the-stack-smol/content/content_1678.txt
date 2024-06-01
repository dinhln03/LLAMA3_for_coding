import random
import time
class unit :
    def __init__(self, HP, Atk) :
        self.HP = HP
        self.Atk = Atk
    def DoAtk(self, OtherUnit) :
        self.DoDmg(self.Atk, OtherUnit)
    def DoDmg(self, dmg, OtherUnit) :
        OtherUnit.HP-=dmg
class hero(unit) :
    def __init__(self, HP, Atk) :
        super(hero, self).__init__(HP, Atk)
    def Heal(self) :
        self.HP+=30
class moster(unit) :
    def __init__(self, HP, Atk) :
        super(moster, self).__init__(HP, Atk)
    def HardAtk(self, OtherUnit) :
        self.DoDmg(self.Atk + 10, OtherUnit)
while True :
    print("initializing hero...")
    time.sleep(1)
    Hero = hero(random.randint(30,50), random.randint(10,15))
    print("A hero is here now")
    print("HP:" + str(Hero.HP))
    print("Atk:" + str(Hero.Atk) + "\n")
    time.sleep(0.5)
    print("initializing moster...")
    time.sleep(1)
    Moster = moster(random.randint(20,30), random.randint(5,10))
    print("A moster is here now")
    print("HP:" + str(Moster.HP))
    print("Atk:" + str(Moster.Atk) + "\n")
    ###
    time.sleep(1.5)
    while Hero.HP > 0 and Moster.HP > 0 :
        print("Hero turn")
        time.sleep(1.5)
        if Hero.HP < 10 :
            Hero.Heal()
            print("Hero use heal")
        else :
            Hero.DoAtk(Moster)
            print("Hero atk")
        print("Hero HP:" + str(Hero.HP))
        print("Moster HP:" + str(Moster.HP) + "\n")
        time.sleep(1.5)
        if Moster.HP <= 0:
            print("Hero Win")
            break
        ###
        print("Moster turn")
        time.sleep(1.5)
        if Moster.HP < 5 :
            Moster.HardAtk(Hero)
            print("Moster use Hard Atk")
        else :
            Moster.DoAtk(Hero)
            print("Moster atk")
        print("Hero HP:" + str(Hero.HP))
        print("Moster HP:" + str(Moster.HP) + "\n")
        if Moster.HP <= 0:
            print("Hero Win")
            break
        if Hero.HP <= 0:
            print("Moster Win")
            break
        time.sleep(1.5)