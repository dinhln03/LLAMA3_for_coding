from tkinter import *

#Palomo, Nemuel Rico O.

class ButtonLab:

    def __init__(self, window):
        self.color = Button(window, text='Color', fg='red', bg='blue')
        self.button = Button(window, text='<---Click to change the color of the button :)', fg='black', command=self.changeColor)
        self.color.place(x=120, y=150)
        self.button.place(x=200, y=150)

    def changeColor(self):
        self.color.config(bg='yellow')




window = Tk()
mywin = ButtonLab(window)
window.title('Button (The Coders)')
window.geometry("500x220+10+10")
window.mainloop()







