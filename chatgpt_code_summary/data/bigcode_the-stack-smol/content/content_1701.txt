from chatterbot.trainers import ListTrainer
from chatterbot import ChatBot

bot = ChatBot('Test')

conversa = ['oi', 'olá', 'Tudo bem?', 'Estou bem']
conversa2 = ['Gosta de futebol?','Eu adoro,sou tricolor Paulista e você','Qual seu filme favorito?' , 'O meu é Rocky 1']

bot.set_trainer(ListTrainer)
bot.train(conversa)
bot.train(conversa2)

while True:
    quest = input ("Voce:")
    respota = bot.get_response(quest)
    #if float (response.confidence) >0.5
    print ('Bot:', respota)
    #else:
    # print ("Eu não sei")
