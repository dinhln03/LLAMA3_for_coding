'''
Borrowed from Asteroid.py  and Ship.py which was created by Lukas Peraza
    url: https://github.com/LBPeraza/Pygame-Asteroids

Subzero sprite borrowed from: https://www.spriters-resource.com/playstation/mkmsz/sheet/37161/
'''
import pygame
import os
from CollegiateObjectFile import CollegiateObject

# right in variable means facing right, left means facing left
class Character(CollegiateObject):
    
    @staticmethod
    def init(character):
        
        # Create a list of every image of a character
        images = []
        
        path = "images/%s/ordered images" %character
        
        # Upload each image in order, and resize accordingly
        
        maxDim = 70
        
        for imageName in os.listdir(path):
            
            maxDim = 70
            
            image = pygame.image.load(path + os.sep + imageName)
            
            if "effect" in imageName:
                
                # Resize special move effects images with static attribute maxDim
                if character == "goku" or character == "raizen" or character == "naruto" or character == "sasuke":
                    maxDim = 120
                else:
                    maxDim = 70
                w, h = image.get_size()
                factor = 1
                if w != maxDim:
                    factor = maxDim / w
                if h != maxDim:
                    factor = maxDim / h
                image = pygame.transform.scale( image, ( int(w * factor), int(h * factor) ) )
            
            elif "jump" in imageName:
                
                # Resize special move effects images with static attribute maxDim
                w, h = image.get_size()
                factor = 1
                if w != Character.maxWidth:
                    factor = Character.maxWidth / w
                image = pygame.transform.scale( image, ( int(w * factor), int(h * factor) ) )
                
            else:
                
                # Resize character images with static attribute maxWidth and maxHeight
                w, h = image.get_size()
                factor = 1
                if w != Character.maxWidth:
                    factor = Character.maxWidth / w
                if h != Character.maxHeight:
                    factor = Character.maxHeight / h
                image = pygame.transform.scale( image, ( int(w * factor), int(h * factor) ) )
            
            images.append(image)
        
        Character.charactersDict[character] = Character.charactersDict.get(character, images)
        
        
    
    # Create a dictionary of the images of a character mapped to the character
    charactersDict = {}
    maxWidth = 100
    maxHeight = 170
    maxDim = 70
    gravity = .75
    runVel = 10
    
    maxHealth = 300
    maxEnergy = 100
    
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    orange = (255, 128, 0)

    def __init__(self, character, screenWidth, screenHeight, isRight, player):
        
        self.character = character
        
        self.player = player
        
        self.isRight = isRight
        
        if self.character == "subzero": self.specialName = "freeze"
        
        elif self.character == "scorpion": self.specialName = "spear"
        
        elif self.character == "raizen": self.specialName = "spirit shotgun"
        
        elif self.character == "goku": self.specialName = "kamehameha"
        
        elif self.character == "naruto": self.specialName = "rasengan"
        
        elif self.character == "sasuke": self.specialName = "chidori"
        
        Character.maxHeight = screenHeight
        Character.maxWidth = screenWidth
        
        # Initiate health and energy bars
        margin = 5
        barMargin = margin + 45
        
        self.barHeight = 10
        
        self.healthY = 10
        self.healthWidth = 300
        self.health = Character.maxHealth
        
        self.healthColor = Character.green
        labeledge = 20
        if self.player == 1:
            self.healthX = barMargin
        elif self.player == 2:
            self.healthX = screenWidth - barMargin - self.healthWidth - labeledge 
        
        self.energyY = 30
        self.energy = Character.maxEnergy
        
        self.energyColor = Character.red
        
        if self.player == 1:
            self.energyX = barMargin
        elif self.player == 2:
            self.energyX = screenWidth - barMargin - self.healthWidth - labeledge
        
        self.images = Character.charactersDict[character]
        
        # All imported images are uploaded in the following order: icon, idle, jump, block, run, punch, kick, special, effect
        characterInstances = ["icon", "idle", "jump", "block", "damage1", "run", "punch", "kick", "special1", "effect1"]# added "damage", after block
        
        # Create a dictionary mapping the character instance to it's respective image
        self.spriteRightDict = {}
        i = 0
        
        for instance in characterInstances:
            
            self.spriteRightDict[instance] = self.spriteRightDict.get(instance, self.images[i])
            i += 1
        
        # Flip all pictures to face left for left disctionary
        self.spriteLeftDict = {}
        
        j = 0
        for sprite in characterInstances:
            
            # Don't want to flip run image yet
            if sprite == "run":
                self.spriteLeftDict[sprite] = self.images[j]
                
            image = pygame.transform.flip(self.images[j], True, False)
            
            self.spriteLeftDict[sprite] = self.spriteLeftDict.get(sprite, image)
            j += 1
        
        # Pass information to parent CollegiateObject class to initialize character
        self.spriteDict = {}
        
        # Get the starting image, and x location
        
        if self.isRight:
            self.spriteDict = self.spriteRightDict 
            idleImage = self.spriteRightDict["idle"]
            w, h = idleImage.get_size()
            x = margin + (w // 2)
        elif not self.isRight:
            self.spriteDict = self.spriteLeftDict
            idleImage =  self.spriteLeftDict["idle"]
            w, h = idleImage.get_size()
            x = screenWidth - margin -  (w // 2)
        
        r = max(w,h) // 2
        y = screenHeight - margin - (h // 2)
        
        super(Character, self).__init__(x, y, idleImage, r)
        
        # Get dictionary of sounds (actually set in run game, but initiated here)
        self.sounds = {}
        
        # Set other attributes
        self.isDead = False
        
        self.isFlipped = False
        
        self.isIdle = True
        self.idleCount = 0
        
        self.isAttack = False
        self.isDamage = False
        
        # Keep damage image for 1 second
        self.damageCount = 1
        
        self.isRunLeft = False
        self.isRunRight = False

        
        self.isJump = False
        self.jumpVel = 10
        self.peakJump = screenWidth // 4
        self.idleY = self.y
        
        self.isBlock = False
        
        self.isPunch = False
        
        # Keep punch image for 1 second
        self.punchCount = 1
        self.punchDamage = 20
        
        self.isKick = False
        self.kickCount = 20
        self.kickDamage = 25
        
        self.isSpecial = False
        self.specialCount = 30
        self.specialDamage = 50
        
        #print("Loaded Character")
    
    def loseHealth(self, damage):
        if self.isBlock: self.sounds["block"].play()
        if not self.isBlock: self.sounds["damage1"].play()
        if self.isDamage and self.health > 0:
            self.health -= damage
        if self.health <= 0:
            if self.healthColor == Character.green:
                self.health = Character.maxHealth
                self.healthColor = Character.orange
            elif self.healthColor == Character.orange:
                self.health = Character.maxHealth
                self.healthColor = Character.red
            else:
                self.health = 0
                self.isDead = True
        if not self.isBlock:
            self.baseImage = self.spriteDict["damage1"]
    
    def getEnergy(self):
        increment = 10
        maxEnergy = 100
        if self.energy <= (maxEnergy - increment) and self.isAttack:
            self.energy += increment
            if self.energy >= Character.maxEnergy:
                self.energy = Character.maxEnergy
            
    def update(self, dt, keysDown, screenWidth, screenHeight):
        
        # Change facing direction when characters switch sides
        if self.isRight:
            self.spriteDict = self.spriteRightDict
        elif not self.isRight:
            self.spriteDict = self.spriteLeftDict
        
        player1Moves = {"Left": keysDown(pygame.K_a), "Right": keysDown(pygame.K_d),
                        "Down": keysDown(pygame.K_s), "Up": keysDown(pygame.K_w), 
                        "Punch": keysDown(pygame.K_v), "Kick": keysDown(pygame.K_c), 
                        "Special1": keysDown(pygame.K_SPACE) }
                        
        player2Moves = {"Left": keysDown(pygame.K_LEFT), "Right": keysDown(pygame.K_RIGHT), 
                        "Down": keysDown(pygame.K_DOWN), "Up": keysDown(pygame.K_UP), 
                        "Punch": keysDown(pygame.K_l), "Kick": keysDown(pygame.K_k), 
                        "Special1": keysDown(pygame.K_j) }
        
        if self.player == 1:
            self.moves = player1Moves
            
        elif self.player == 2: 
            self.moves = player2Moves
            
        self.idleCount += 1
        
        margin = 5
        
        boarderLeft = 0 + margin + (self.width // 2)
        boarderRight = screenWidth - margin - (self.width // 2)
        boarderBottom = screenHeight - margin - (self.height // 2)

        if self.moves["Left"] and self.x > boarderLeft and not self.isJump and not self.isBlock and not self.isDamage:
            self.x -= Character.runVel
            self.baseImage = pygame.transform.flip(self.spriteDict["run"], True, False)
            self.isRunLeft = True
            self.isIdle = False
        
        if self.isRunLeft and not self.isJump and not self.moves["Left"]:
            self.isRunLeft = False
            self.isIdle = True
            self.baseImage = self.spriteDict["idle"]

        if self.moves["Right"] and self.x < boarderRight and not self.isJump and not self.isBlock and not self.isDamage:
            # not elif! if we're holding left and right, don't turn
            self.x += Character.runVel
            self.baseImage = self.spriteDict["run"]
            self.isRunRight = True
            self.isIdle = False
        
        if self.isRunRight and not self.isJump and not self.moves["Right"]:
            self.isRunRight = False
            self.isIdle = True
            self.baseImage = self.spriteDict["idle"]
            
        if self.moves["Down"] and not self.isJump and not self.isDamage:
            self.baseImage = self.spriteDict["block"]
            self.isBlock = True
            self.isIdle = False
        
        if self.isBlock and not self.moves["Down"]: 
            self.isBlock = False
            self.isIdle = True
            self.baseImage = self.spriteDict["idle"]
        
        if self.moves["Up"] and self.y >= boarderBottom and not self.isJump and not self.isBlock and not self.isDamage:# and self.isIdle:
            self.sounds["jump"].play()
            self.baseImage = self.spriteDict["jump"]
            self.isJump = True
            self.isIdle = False
            
            
        elif self.isJump:
            if self.jumpVel >= 0:
                self.y -= (self.jumpVel** 2) // 2
                if self.isRunLeft and (self.x - Character.runVel) >= boarderLeft:
                    self.x -= Character.runVel
                elif self.isRunRight and (self.x + Character.runVel) <= boarderRight:
                    self.x += Character.runVel
                self.jumpVel -= Character.gravity
                
            else:
                self.y += (self.jumpVel** 2) // 2
                if self.isRunLeft and (self.x - Character.runVel) >= boarderLeft:
                    self.x -= Character.runVel
                elif self.isRunRight and (self.x + Character.runVel) <= boarderRight:
                    self.x += Character.runVel
                self.jumpVel -= Character.gravity
            if self.y > self.idleY:
                self.baseImage = self.spriteDict["idle"]
                self.y = self.idleY
                self.isJump = False
                self.isRunLeft = False
                self.isRunRight = False
                self.isIdle = True
                self.jumpVel = 10
        
        if self.moves["Punch"] and self.isIdle and self.idleCount >= 20 and not self.isPunch and not self.isDamage:
            self.sounds["punch"].play()
            self.baseImage = self.spriteDict["punch"]
            self.isPunch = True
            self.isIdle = False
        
        elif self.isPunch:
            if self.punchCount >= 0:
                self.punchCount -= 1
            else:
                self.isPunch = False
                self.isIdle = True
                self.idleCount = 0
                self.punchCount = 20
                self.baseImage = self.spriteDict["idle"]
        
        if self.moves["Kick"] and self.isIdle and self.idleCount >= 20 and not self.isDamage:
            self.sounds["kick"].play()
            self.baseImage = self.spriteDict["kick"]
            self.isKick = True
            self.isIdle = False
        
        elif self.isKick:
            if self.kickCount >= 0:
                self.kickCount -= 1
            else:
                self.isKick = False
                self.isIdle = True
                self.idleCount = 0
                self.kickCount = 20
                self.baseImage = self.spriteDict["idle"]
        
        if self.moves["Special1"] and self.isIdle and self.idleCount >= 20 and (self.energy >= self.specialDamage) and not self.isJump and not self.isBlock and not self.isDamage:
            self.sounds["special1"].play()
            self.baseImage = self.spriteDict["special1"]
            self.isSpecial = True
            self.isIdle = False
            self.energy -= self.specialDamage
        
        elif self.isSpecial:
            if self.specialCount >= 0:
                self.specialCount -= 1
            else:
                self.isSpecial = False
                self.isIdle = True
                self.idleCount = 0
                self.specialCount = 30
                self.baseImage = self.spriteDict["idle"]
        
        super(Character, self).update(screenWidth, screenHeight)
