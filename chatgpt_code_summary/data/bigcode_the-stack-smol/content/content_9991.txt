import pygame
import os
from pygame.locals import *
import config
import game
import engine
import menu
from random import randint
import _fighter
from pygame_functions import *



class Scenario:
    
    def __init__(self, game, scenario):
        self.game = game
        self.scenario = scenario
        pygame.mixer.music.stop()
        music = engine.Music("mkt")
        music.play()
        music.volume(0.5)

    def setScenario(self, scenario):
        if scenario == 9:
            scenario = randint(1, 8)
        #self.scene = pygame.image.load('../res/Background/Scenario'+str(scenario)+'.png')
        #self.game.getDisplay().blit(self.scene, (0, 0))
        #pygame.display.update()
        #screenSize(800, 500,"pyKombat",None,None,True) # FullScreen
        screenSize(800, 500,"pyKombat") # Minimized
        setBackgroundImage('../res/Background/Scenario'+str(scenario)+'.png')
        self.judge(scenario)
    
    def judge(self,scenario):
        [player1,player2] = self.addFigther(scenario) 
        player1.act()
        player2.act()
        nextFrame1 = clock()
        nextFrame2 = clock()
        hitCounter = 0
        while True:
            aux1 = player1.fight(clock(),nextFrame1)
            nextFrame1 = aux1
            aux2 = player2.fight(clock(),nextFrame2)
            nextFrame2 = aux2
            x1 = player1.getX()
            x2 = player2.getX()
            #print(x1, x2, x2-x1)
            # caso encostem na tela
            if player1.getX() < 20:
                player1.setX(20) 

            if player2.getX() < 20:
                player2.setX(20)  
            
            if player1.getX() > (800-20):
                player1.setX(800-20) 

            if player2.getX() > (800-20):
                player2.setX(800-20)    
                
            if(collide(player1.currentSprite(),player2.currentSprite())):
                # caso só encostem
                if ( (player1.isWalking() or player1.isJumping()) and (player2.isDancing() or player2.isCrouching() or player2.isWalking()) ) or ((player2.isWalking() or player2.isJumping()) and (player1.isDancing() or player1.isCrouching() or player2.isWalking()) ) or (player1.isWalking() and player2.isWalking()) or (player1.isJumping() and player2.isJumping()) or (player1.isDancing() and player2.isDancing()) or (player2.isSpecialMove() and player1.ishitSpecial()):
                    player1.setX(x1-15)
                    if not player2.isSpecialMove() :player2.setX(x2+15) 
                    else: player1.setX(x1-25)
                # caso houve soco fraco:
                if ( player1.isApunching() and (player2.isWalking() or player2.isDancing() or player2.isApunching() or player2.ishitSpecial()) ) or ( player2.isApunching() and (player1.isWalking() or player1.isDancing() or player1.isApunching()) ):
                    if player1.isApunching():                        
                        player2.takeHit("Apunching")
                    if player2.isApunching():    
                        player1.takeHit("Apunching")
                    print("socofraco")
                    engine.Sound("Hit0").play()
                    if hitCounter == 0: engine.Sound().roundHit()
                    hitCounter = (hitCounter+1) % 5 
                # caso houve soco forte:
                if ( player1.isBpunching() and (player2.isWalking() or player2.isDancing() or player2.isBpunching()) ) or ( player2.isBpunching() and (player1.isWalking() or player1.isDancing() or player1.isBpunching()) ):
                    if player1.isBpunching():                        
                        player2.takeHit("Bpunching")
                    if player2.isBpunching():    
                        player1.takeHit("Bpunching")
                    print("socoforte")
                    engine.Sound("Hit0").play()
                    if hitCounter == 0: engine.Sound().roundHit()
                    hitCounter = (hitCounter+1) % 5 
                # caso houve chute fraco:
                if ( player1.isAkicking() and (player2.isWalking() or player2.isDancing() or player2.isAkicking() or player2.isCrouching()) and not player2.isBblocking() ) or ( player2.isAkicking() and (player1.isWalking() or player1.isDancing() or player1.isAkicking() or player1.isCrouching() and not player1.isBblocking()) ):
                    if player1.isAkicking():                        
                        player2.takeHit("Akicking")
                    if player2.isAkicking():                        
                        player1.takeHit("Akicking")
                    print("chutefraco")
                    engine.Sound("Hit0").play()
                    if hitCounter == 0: engine.Sound().roundHit()
                    hitCounter = (hitCounter+1) % 5 
                # caso houve chute forte:
                if ( player1.isBkicking() and (player2.isWalking() or player2.isDancing() or player2.isBkicking()) ) or ( player2.isBkicking() and (player1.isWalking() or player1.isDancing() or player1.isBkicking()) ):
                    if player1.isBkicking():                        
                        player2.takeHit("Bkicking")
                    if player2.isBkicking():                        
                        player1.takeHit("Bkicking")
                    print("chuteforte")
                    engine.Sound("Hit0").play()
                    if hitCounter == 0: engine.Sound().roundHit()
                    hitCounter = (hitCounter+1) % 5 
                # caso houve bloqueio em pé:
                if ( (player1.isApunching() or player1.isBpunching() or player1.isDpunching() or player1.isAkicking() or player1.isBkicking() ) and player2.isAblocking() ) or ( (player2.isApunching() or player2.isBpunching() or player1.isDpunching() or player2.isAkicking() or player2.isBkicking() ) and player1.isAblocking() ):
                    if player1.isAblocking():                        
                        player1.takeHit("Ablocking")
                    if player2.isAblocking():                        
                        player2.takeHit("Ablocking")
                    engine.Sound("block").play()
                    player1.setX(x1-12)
                    player2.setX(x2+12) 
                    print("ablock")
                # caso houve soco ou chute agachado fraco em alguém em pé:
                if ( ((player1.isCpunching() or player1.isCkicking() ) and not player2.isCrouching() and not player2.isBblocking() ) or ((player2.isCpunching() or player2.isCkicking() ) and not player1.isCrouching() and not player1.isBblocking() ) ): # falta adicionar o Bblock
                    if player1.isCpunching() or player1.isCkicking():                        
                        player2.takeHit("Cpunching")
                    if player2.isCpunching() or player2.isCkicking():    
                        player1.takeHit("Cpunching")
                    print("socofraco!!!!!!!")
                    engine.Sound("Hit0").play()
                    if hitCounter == 0: engine.Sound().roundHit()
                    hitCounter = (hitCounter+1) % 5
                # caso houve soco agachado forte em alguém em pé:
                if ( (player1.isDpunching() and (not player2.isAblocking() and not player2.isBblocking())  )  or player2.isDpunching() and (not player1.isAblocking() and not player1.isBblocking()) ): 
                    if player1.isDpunching():                        
                        player2.takeHit("Bkicking")
                    if player2.isDpunching():    
                        player1.takeHit("Bkicking")
                    print("socofraco$#$")
                    engine.Sound("Hit0").play()
                    if hitCounter == 0: engine.Sound().roundHit()
                    hitCounter = (hitCounter+1) % 5 
                # caso houve chute agachado forte em alguém em pé:
                if ( player1.isDkicking()  or player2.isDkicking() ): 
                    if player1.isDkicking():                        
                        player2.takeHit("Dkicking")
                    if player2.isDkicking():    
                        player1.takeHit("Dkicking")
                    print("socofraco")
                    engine.Sound("Hit0").play()
                    if hitCounter == 0: engine.Sound().roundHit()
                    hitCounter = (hitCounter+1) % 5 
                # caso houve soco ou chute agachado fraco em alguém agachado:
                if ( ( (player1.isCpunching() or player1.isCkicking()) and player2.isCrouching() and not player2.isBblocking()  )  or ( (player2.isCpunching() or player2.isCkicking()) and player1.isCrouching() and not player1.isBblocking() ) ):
                    if player1.isCpunching() or player1.isCkicking():                        
                        player2.takeDownHit("Ehit")
                    if player2.isCpunching() or player2.isCkicking():    
                        player1.takeDownHit("Ehit")
                    print("socofraco**")
                    engine.Sound("Hit0").play()
                    if hitCounter == 0: engine.Sound().roundHit()
                    hitCounter = (hitCounter+1) % 5
                # caso houve bloqueio agachado:
                if ( (player1.isCpunching() or player1.isDpunching() or player1.isAkicking() or player1.isCkicking() ) and player2.isBblocking() ) or ( (player2.isCpunching() or player2.isDpunching() or player2.isAkicking() or player2.isCkicking() ) and player1.isBblocking() ):
                    if player1.isBblocking():                        
                        player1.takeDownHit("Bblocking")
                    if player2.isBblocking():                        
                        player2.takeDownHit("Bblocking")
                    engine.Sound("block").play()
                    player1.setX(x1-12)
                    player2.setX(x2+12) 
                    print("bblock")

            # caso houve special
            if ( player1.isSpecialMove() and (player2.isWalking() or player2.isDancing()) ) or ( player2.isSpecialMove() and (player1.isWalking() or player1.isDancing()) ):
                if player1.isSpecialMove() and collide(player1.getProjectile().getProjectileSprite(), player2.currentSprite()):   # and collide(projetil,player2)
                    player2.takeHit("special")
                if player2.isSpecialMove():   # and collide(projetil,player1)   
                    player1.takeHit("special")
                print("special")
                    

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
            if keyPressed("backspace"):
                pygame.quit()
            if keyPressed("esc"):
                self.goBack(player1,player2)
    
    def addFigther(self,scenario):
        player1 = _fighter.Fighter(0,scenario) # 0: subzero
        player2 = _fighter.Fighter(1,scenario) # 1: scorpion
        return player1,player2
    
    def goBack(self,player1,player2):
        player1.killPlayer()
        player2.killPlayer()
        del(player1)
        del(player2)
        sound = engine.Sound("back")  
        sound.play()
        pygame.mixer.music.stop()
        music = engine.Music("intro")
        music.play()     
        music.volume(0.5)
        menu.ScenarioMenu()
       
                        
def collide(sprite1,sprite2):
    return pygame.sprite.collide_mask(sprite1,sprite2)
