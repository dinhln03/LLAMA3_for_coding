import imp
from tkinter import *
from sys import exit
from teste.testeCores.corFunc import formatar

conta2x2 = 'x2y=5\n3x-5y=4'
root = Tk()
text = Text(root, width=20, height=10)
text.config(font='arial 20 bold')
text.insert(END, conta2x2)
text.pack()

def q_evento(event):
   exit()
root.bind('q', q_evento)

cs = conta2x2.split('\n')
print('cs', cs)
posicao = cs[0].find('y')
print('posicao:', posicao)

p1 = p2 = '1.'
p1 += str(posicao)
p2 += str(posicao+1)

print('p1:', p1, 'p2:', p2)

conta = conta2x2.split('\n')
formatado = list()
text.config(background='black', foreground='white')

for i, c in enumerate(conta):
   formatado.append(formatar(i, c))
fs = formatado[0][0]
print(fs)
print(fs['p1'])
for f1 in formatado:
   for f in f1:
      
      text.tag_add(f['nome'], f['p1'], f['p2'])
      text.tag_config(f['nome'], foreground=f['fg'])
# text.tag_add("y1", p1, p2)
# text.tag_config("y1", background="black", foreground="green")

text.tag_config('1', foreground="green")
root.mainloop()