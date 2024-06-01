from tkinter import *

class MainFrame(Frame):
    def __init__(self, parent):
        super().__init__()
        self['bd'] = 1
        self['relief'] = SOLID
        self['padx'] = 5
        self['pady'] = 5

        self.label_text1 = StringVar()
        self.label_text1.set('Digite seu nome')
        self.text_text1 = StringVar()

        # Widgets
        self.label1 = Label(self, textvariable=self.label_text1).grid()
        text1 = Entry(self, textvariable=self.text_text1).grid(pady=2)
        btn1 = Button(self, text='Clique', command=self.executar).grid()
    
    def executar(self):
        if not self.text_text1.get():
            self.label_text1.set('Você não digitou nada')
        else:
            self.label_text1.set(f'Olá, {self.text_text1.get().capitalize()}!')
            self.text_text1.set('')

root = Tk()
root.title('Passar valor')
icone = PhotoImage(file='images/icon.png')
root.iconphoto(False, icone)
root.geometry('200x110')

MainFrame(root).pack(pady=10)

root.mainloop()