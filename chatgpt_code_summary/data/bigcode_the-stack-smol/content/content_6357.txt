from keras.utils import to_categorical
import tensorflow as tf
import pygame

class pytennis:
    def __init__(self, fps = 50):
        self.net = Network(150,450,100,600)
        self.updateRewardA = 0
        self.updateRewardB = 0
        self.updateIter = 0
        self.lossA = 0
        self.lossB = 0
        
        # Testing
        self.net = Network(150, 450, 100, 600)
        self.NetworkA = self.net.network(300, ysource=100, Ynew=600)  # Network A
        self.NetworkB = self.net.network(200, ysource=600, Ynew=100)  # Network B
        # NetworkA

        # display test plot of network A
        #sns.jointplot(NetworkA[0], NetworkA[1])

        # display test plot of network B
        #sns.jointplot(NetworkB[0], NetworkB[1])
        
        
        
        self.out = self.net.DefaultToPosition(250)
        
        
        self.lastxcoordinate = 350
        
        pygame.init()
        self.BLACK = ( 0,0,0)
        
        self.myFontA = pygame.font.SysFont("Times New Roman", 25)
        self.myFontB = pygame.font.SysFont("Times New Roman", 25)
        self.myFontIter = pygame.font.SysFont('Times New Roman', 25)
        
        
        self.FPS = fps
        self.fpsClock = pygame.time.Clock()
        
    def setWindow(self):

        # set up the window
        self.DISPLAYSURF = pygame.display.set_mode((600, 700), 0, 32)
        pygame.display.set_caption('REINFORCEMENT LEARNING (Discrete Mathematics) - TABLE TENNIS')
        # set up the colors
        self.BLACK = ( 0,0,0)
        self.WHITE = (255, 255, 255)
        self.RED= (255,0,0)
        self.GREEN = ( 0, 255,0)
        self.BLUE = ( 0,0, 255)
        
        return
        
        

        
    def display(self):
        self.setWindow()
        self.DISPLAYSURF.fill(self.WHITE)
        pygame.draw.rect(self.DISPLAYSURF, self.GREEN, (150, 100, 300, 500))
        pygame.draw.rect(self.DISPLAYSURF, self.RED, (150, 340, 300, 20))
        pygame.draw.rect(self.DISPLAYSURF, self.BLACK, (0, 20, 600, 20))
        pygame.draw.rect(self.DISPLAYSURF, self.BLACK, (0, 660, 600, 20))
        return
    
    
    
    def reset(self):
        return
    
    def evaluate_state_from_last_coordinate(self, c):
        """
        cmax: 450
        cmin: 150
        
        c definately will be between 150 and 450.
        state0 - (150 - 179)
        state1 - (180 - 209)
        state2 - (210 - 239)
        state3 - (240 - 269)
        state4 - (270 - 299)
        state5 - (300 - 329)
        state6 - (330 - 359)
        state7 - (360 - 389)
        state8 - (390 - 419)
        state9 - (420 - 450)
        """
        if c >= 150 and c <=179:
            return 0
        elif c >= 180 and c <= 209:
            return 1
        elif c >=210 and c <= 239:
            return 2
        elif c >=240 and c <= 269:
            return 3
        elif c>= 270 and c<=299:
            return 4
        elif c >= 300 and c <= 329:
            return 5
        elif c >= 330 and c <= 359:
            return 6
        elif c >= 360 and c <= 389:
            return 7
        elif c >= 390 and c <= 419:
            return 8
        elif c >= 420 and c <= 450:
            return 9
        
    def evaluate_action(self, action, expectedState):
        if action == expectedState:
            return True
        else:
            return False
        
    def randomVal(self, action):
        """
        cmax: 450
        cmin: 150
        
        c definately will be between 150 and 450.
        state0 - (150 - 179)
        state1 - (180 - 209)
        state2 - (210 - 239)
        state3 - (240 - 269)
        state4 - (270 - 299)
        state5 - (300 - 329)
        state6 - (330 - 359)
        state7 - (360 - 389)
        state8 - (390 - 419)
        state9 - (420 - 450)
        """
        if action == 0:
            val = np.random.choice([i for i in range(150, 180)])
        elif action == 1:
            val = np.random.choice([i for i in range(180, 210)])
        elif action == 2:
            val = np.random.choice([i for i in range(210, 240)])
        elif action == 3:
            val = np.random.choice([i for i in range(240, 270)])
        elif action == 4:
            val = np.random.choice([i for i in range(270, 300)])
        elif action == 5:
            val = np.random.choice([i for i in range(300, 330)])
        elif action == 6:
            val = np.random.choice([i for i in range(330, 360)])
        elif action == 7:
            val = np.random.choice([i for i in range(360, 390)])
        elif action == 8:
            val = np.random.choice([i for i in range(390, 420)])
        else:
            val = np.random.choice([i for i in range(420, 450)])
        return val
        
    def stepA(self, action, count = 0):
        #playerA should play
        if count == 0:
            #playerax = lastxcoordinate
            self.NetworkA = self.net.network(self.lastxcoordinate, ysource = 100, Ynew = 600) #Network A
            self.out = self.net.DefaultToPosition(self.lastxcoordinate)

            #update lastxcoordinate

            self.bally = self.NetworkA[1][count]
            #here
            #self.playerax = self.out[count]
            self.playerbx = self.randomVal(action)
            
            
#             soundObj = pygame.mixer.Sound('sound/sound.wav')
#             soundObj.play()
#             time.sleep(0.4)
#             soundObj.stop()
        elif count == 49:
            self.ballx = self.NetworkA[0][count]
            self.bally = self.NetworkA[1][count]
            
            # move playerbx with respect to action 
            self.playerbx = self.randomVal(action)


        else:
            self.ballx = self.NetworkA[0][count]
            self.bally = self.NetworkA[1][count]
            
            # move playerbx with respect to action 
#             self.playerbx = self.randomVal(action)
            
            
        obs = self.evaluate_state_from_last_coordinate(int(self.ballx)) # last state of the ball
        reward = self.evaluate_action(action, obs)
        done = True
        info = ''


        return obs, reward, done, info
    
    
    def stepB(self, action, count):
        #playerB can play
        if count == 0:
            #playerbx = lastxcoordinate
            self.NetworkB = self.net.network(self.lastxcoordinate, ysource = 600, Ynew = 100) #Network B
            self.out = self.net.DefaultToPosition(self.lastxcoordinate)

            #update lastxcoordinate
            self.bally = self.NetworkB[1][count]
            #self.playerax = self.out[count] 
            self.playerax = self.randomVal(action)

#             soundObj = pygame.mixer.Sound('sound/sound.wav')
#             soundObj.play()
#             time.sleep(0.4)
#             soundObj.stop()
        elif count ==49:
            self.ballx = self.NetworkA[0][count]
            self.bally = self.NetworkA[1][count]
            
            # move playerbx with respect to action 
            self.playerbx = self.randomVal(action)
            
        else:
            self.ballx = self.NetworkB[0][count]
            self.bally = self.NetworkB[1][count]
#             self.playerbx = self.randomVal(action)
            
        obs = self.evaluate_state_from_last_coordinate(int(self.ballx)) # last state of the ball
        reward = self.evaluate_action(action, obs)
        done = True
        info = ''
        
        return obs, reward, done, info
    
    def computeLossA(self, reward):
        if reward == 0:
            self.lossA += 1
        else:
            self.lossA += 0
        return

    def computeLossB(self, reward):
        if reward == 0:
            self.lossB += 1
        else:
            self.lossB += 0
        return
    
    def render(self):
        # diplay team players
        self.PLAYERA = pygame.image.load('images/cap.jpg')
        self.PLAYERA = pygame.transform.scale(self.PLAYERA, (50, 50))
        self.PLAYERB = pygame.image.load('images/cap.jpg')
        self.PLAYERB = pygame.transform.scale(self.PLAYERB, (50, 50))
        self.ball = pygame.image.load('images/ball.png')
        self.ball = pygame.transform.scale(self.ball, (15, 15))

        self.playerax = 150
        self.playerbx = 250
        
        self.ballx = 250
        self.bally = 300
        
        
        
        
        count = 0
        nextplayer = 'A'
        #player A starts by playing with state 0
        obs, reward, done, info = self.stepA(0)
        stateA = obs
        stateB = obs
        next_state = 0
        
        iterations = 20000
        iteration = 0
        restart = False
        
        while iteration < iterations:
            self.display()
            self.randNumLabelA = self.myFontA.render('A (Win): '+str(self.updateRewardA) + ', A(loss): '+str(self.lossA), 1, self.BLACK)
            self.randNumLabelB = self.myFontB.render('B (Win): '+str(self.updateRewardB) + ', B(loss): '+ str(self.lossB), 1, self.BLACK)
            self.randNumLabelIter = self.myFontIter.render('Iterations: '+str(self.updateIter), 1, self.BLACK)
            if nextplayer == 'A':

                if count == 0:
                    # Online DQN evaluates what to do
                    q_valueA = AgentA.model.predict([stateA])
                    actionA = AgentA.epsilon_greedy(q_valueA, iteration)
                    
                    # Online DQN plays
                    obs, reward, done, info = self.stepA(action = actionA, count = count)
                    next_stateA = obs
                    
                    # Let's memorize what just happened
                    AgentA.replay_memory.append((stateA, actionA, reward, next_stateA, 1.0 - done))
                    stateA = next_stateA
                    
                    
                else:                    
                    # Online DQN evaluates what to do
                    q_valueA = AgentA.model.predict([stateA])
                    actionA = AgentA.epsilon_greedy(q_valueA, iteration)
                    
                    # Online DQN plays
                    
                    obs, reward, done, info = self.stepA(action = actionA, count = count)
                    next_stateA = obs
                    
                    # Let's memorize what just happened
#                     AgentA.replay_memory.append((state, action, reward, next_state, 1.0 - done))
                    stateA = next_stateA
                    
                count += 1  
                if count == 50:
                    count = 0
                    

                    self.updateRewardA += reward
                    self.computeLossA(reward)
                    
                    #restart the game if player A fails to get the ball, and let B start the game
                    if reward == 0:
                        restart = True
                        time.sleep(0.5)
                        nextplayer = 'B'
                        self.playerbx = self.ballx
                    else:
                        restart = False
                        
                    # Sample memories and use the target DQN to produce the target Q-Value
                    X_state_val, X_action_val, rewards, X_next_state_val, continues = (AgentA.sample_memories(AgentA.batch_size))
                    next_q_values = AgentA.model.predict([X_next_state_val])
                    max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
                    y_val = rewards + continues * AgentA.discount_rate * max_next_q_values

                    # Train the online DQN
                    AgentA.model.fit(X_state_val,tf.keras.utils.to_categorical(X_next_state_val, num_classes=10), verbose = 0)
                    
                    nextplayer = 'B'
                    self.updateIter += 1
                    
            
                    #evaluate A
                else:
                    nextplayer = 'A'
                    
               

            else:

                if count == 0:
                    # Online DQN evaluates what to do
                    q_valueB = AgentB.model.predict([stateB])
                    actionB = AgentB.epsilon_greedy(q_valueB, iteration)
                    
                    # Online DQN plays
                    obs, reward, done, info = self.stepB(action = actionB, count = count)
                    next_stateB = obs
                    
                    # Let's memorize what just happened
                    AgentB.replay_memory.append((stateB, actionB, reward, next_stateB, 1.0 - done))
                    stateB = next_stateB
                else:
                    # Online DQN evaluates what to do
                    q_valueB = AgentB.model.predict([stateB])
                    actionB = AgentB.epsilon_greedy(q_valueB, iteration)
                    
                    # Online DQN plays
                    obs, reward, done, info = self.stepB(action = actionB, count = count)
                    next_stateB = obs
                    
                    # Let's memorize what just happened
#                     AgentB.replay_memory.append((state, action, reward, next_state, 1.0 - done))
                    stateB = next_stateB
                
                count += 1
                if count == 50:
                    count = 0
                    
                    
                    self.updateRewardB += reward
                    self.computeLossB(reward)
                    
                    
                    #restart the game if player A fails to get the ball, and let B start the game
                    if reward == 0:
                        restart = True
                        time.sleep(0.5)
                        nextplayer = 'A'
                        self.playerax = self.ballx
                    else:
                        restart = False
                    
                    # Sample memories and use the target DQN to produce the target Q-Value
                    X_state_val, X_action_val, rewards, X_next_state_val, continues = (AgentB.sample_memories(AgentB.batch_size))
                    next_q_values = AgentB.model.predict([X_next_state_val])
                    max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
                    y_val = rewards + continues * AgentB.discount_rate * max_next_q_values

                    # Train the online DQN
                    AgentB.model.fit(X_state_val,tf.keras.utils.to_categorical(X_next_state_val, num_classes=10), verbose = 0)
                    
                    nextplayer = 'A'
                    self.updateIter += 1
                    #evaluate B
                else:
                    nextplayer = 'B'

                count += 1
            #CHECK BALL MOVEMENT
            self.DISPLAYSURF.blit(self.PLAYERA, (self.playerax, 50))
            self.DISPLAYSURF.blit(self.PLAYERB, (self.playerbx, 600))
            self.DISPLAYSURF.blit(self.ball, (self.ballx, self.bally))
            self.DISPLAYSURF.blit(self.randNumLabelA, (300, 630))
            self.DISPLAYSURF.blit(self.randNumLabelB, (300, 40))
            self.DISPLAYSURF.blit(self.randNumLabelIter, (50, 40))

            #update last coordinate
            self.lastxcoordinate = self.ballx 

            pygame.display.update()
            self.fpsClock.tick(self.FPS)

            for event in pygame.event.get():

                if event.type == QUIT:
                    AgentA.model.save('AgentA.h5')
                    AgentB.model.save('AgentB.h5')
                    pygame.quit()
                    sys.exit()


           
        
    