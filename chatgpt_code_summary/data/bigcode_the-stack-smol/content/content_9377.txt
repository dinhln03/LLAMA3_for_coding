# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:37:31 2017

@author: Flame
"""

from TuringMachine import Rule, Q, Move, TuringMachine, Tape
from TuringMachine import EMTY_SYMBOL as empty


def check(input_str):
    rules= \
    [
        Rule(Q(1),'1',Q(1),'1', Move.Right),# приводим к первоначальному виду
        Rule(Q(1),'0',Q(1),'0', Move.Right),
        Rule(Q(1),',',Q(2),'#', Move.Right),
        Rule(Q(2),' ',Q(2),'#', Move.Right),
        Rule(Q(2),'#',Q(2),'#', Move.Left),
            Rule(Q(2),'*',Q(6),'*', Move.Left),
        Rule(Q(2),'1',Q(3),'#', Move.Left),     # операции со строками
        Rule(Q(3),'#',Q(3),'#', Move.Left),  
        Rule(Q(3),'1',Q(4),'0', Move.Right), #встретили единичку, значит вычитаем её
        Rule(Q(3),'0',Q(3),'1', Move.Left),  #встретили нолик, значит добавляем единичку и идём вычитать единичку у след порядка
        Rule(Q(4),'0',Q(4),'0', Move.Right), # идём вправо, чтобы найти разделитель
        Rule(Q(4),'1',Q(4),'1', Move.Right),
        Rule(Q(4),'#',Q(5),'#', Move.Right), # идём вправо, чтобы найти единичку
        Rule(Q(5),'#',Q(5),'#', Move.Right),
        Rule(Q(5),'1',Q(3),'#', Move.Left),
        Rule(Q(5),empty,Q(6),empty, Move.Left),
        Rule(Q(6),'#',Q(6),'*', Move.Left),    
        Rule(Q(6),'0',Q(6),'*', Move.Left),   
        Rule(Q(6),empty,Q(10),empty, Move.Stay), # значит строка верна, переходим в конечное состояние
       
    ]
    
    TM  = TuringMachine(rules, Q(1), Q(10))
    print(TM)
    print( "Right" if TM.check(Tape(input_str)) else "Wrong")
