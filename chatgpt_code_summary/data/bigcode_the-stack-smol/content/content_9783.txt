#-------------------------------------------------------------------------------
# Name:GUI Calculator
# Purpose:Simple calculator with GUI using tkinter
#
# Author: Daniel Campos
#
# Created: Monday Dec 1st, 2014
#-------------------------------------------------------------------------------
from tkinter import *
import math
class Calculator:
    '''GUI for the calculator'''
    def __init__(self, master):
        self.master = master   
        self.stringContents = ''
        self.displayStr = StringVar(self.stringContents)
        self.display = Label(master, textvariable=self.displayStr, width=25, anchor=E, relief=SUNKEN).grid(row=0, columnspan=4)

        self.seven = Button(master, width=3, text='7', command=lambda: self.addSymbol('7')).grid(row=1, column=0)
        self.eight = Button(master, width=3, text='8', command=lambda: self.addSymbol('8')).grid(row=1, column=1)
        self.nine = Button(master, width=3, text='9', command=lambda: self.addSymbol('9')).grid(row=1, column=2)
        self.div = Button(master, width=3, text='/', command=lambda: self.addSymbol('/')).grid(row=1, column=3)
        self.master.bind('7', self.addKeyboardSymbol)
        self.master.bind('8', self.addKeyboardSymbol)
        self.master.bind('9', self.addKeyboardSymbol)
        self.master.bind('/', self.addKeyboardSymbol)

        self.four = Button(master, width=3, text='4', command=lambda: self.addSymbol('4')).grid(row=3, column=0)
        self.five = Button(master, width=3, text='5', command=lambda: self.addSymbol('5')).grid(row=3, column=1)
        self.six  = Button(master, width=3, text='6', command=lambda: self.addSymbol('6')).grid(row=3, column=2)
        self.times = Button(master, width=3, text='*', command=lambda: self.addSymbol('*')).grid(row=3, column=3)
        self.master.bind('4', self.addKeyboardSymbol)
        self.master.bind('5', self.addKeyboardSymbol)
        self.master.bind('6', self.addKeyboardSymbol)
        self.master.bind('*', self.addKeyboardSymbol)

        self.one = Button(master, width=3, text='1', command=lambda: self.addSymbol('1')).grid(row=4, column=0)
        self.two = Button(master, width=3, text='2', command=lambda: self.addSymbol('2')).grid(row=4, column=1)
        self.three = Button(master, width=3, text='3', command=lambda: self.addSymbol('3')).grid(row=4, column=2)
        self.minus = Button(master, width=3, text='-', command=lambda: self.addSymbol('-')).grid(row=4, column=3)
        self.master.bind('1', self.addKeyboardSymbol)
        self.master.bind('2', self.addKeyboardSymbol)
        self.master.bind('3', self.addKeyboardSymbol)
        self.master.bind('-', self.addKeyboardSymbol)

        self.zero = Button(master, width=3, text='0', command=lambda: self.addSymbol('0')).grid(row=5, column=0)
        self.point = Button(master, width=3, text='.', command=lambda: self.addSymbol('.')).grid(row=5, column=1)
        self.equals = Button(master, width=3, text='=', command=lambda: self.evaluate()).grid(row=5, column=2)   
        self.plus = Button(master, width=3, text='+', command=lambda: self.addSymbol('+')).grid(row=5, column=3)

        self.master.bind('0', self.addKeyboardSymbol)
        self.master.bind('.', self.addKeyboardSymbol)
        self.master.bind('<Return>', self.evaluate)
        self.master.bind('+', self.addKeyboardSymbol)

        self.c = Button(master, width=3, text='C', command=lambda: self.clear()).grid(row=6, column=0)
        self.d = Button(master, width=3, text='D', command=lambda: self.backSpace()).grid(row=6, column=1)
        self.lparren = Button(master, width=3, text='(', command=lambda: self.addSymbol('(')).grid(row=6, column=2)
        self.rparren = Button(master, width=3, text=')', command=lambda: self.addSymbol(')')).grid(row=6, column=3)

        self.master.bind('C', self.clear)
        self.master.bind('c', self.clear)
        self.master.bind('<BackSpace>', self.backSpace)
        self.master.bind('(', self.addKeyboardSymbol)
        self.master.bind(')', self.addKeyboardSymbol)
    def addSymbol(self, char):
        '''Displays the inputted char onto the display''' 
        self.stringContents += char
        self.displayStr.set(self.stringContents)
    def addKeyboardSymbol(self,event):
        '''Displays the inputted char onto the display''' 
        self.stringContents += str(repr(event.char))[1:-1]
        self.displayStr.set(self.stringContents)
    def evaluate(self, evt=None):
        '''Evalutes the expression'''
        try:
            self.displayStr.set(eval(self.stringContents))
            self.stringContents = str(eval(self.stringContents))
        except Exception as e:
            self.displayStr.set('Error')
            self.stringContents = ''  
    def clear(self, evt=None):
        '''Clears the expression'''
        self.stringContents = ''
        self.displayStr.set(self.stringContents)
    def backSpace(self, evt=None):
        '''Backspace on expression'''
        self.stringContents = self.stringContents[:-1]
        self.displayStr.set(self.stringContents)
def Main():
    master = Tk()
    calculator = Calculator(master)
    calculator.master.title('Calculator')
    calculator.master.resizable(False, False)
    master.mainloop()
if __name__ == '__main__':
    Main()