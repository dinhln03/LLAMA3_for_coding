print('|------------------------------------------------|');
print('|       COMUTADOR DE GARRAFAS POR DINHEIRO       |');
print('|                                                |');
print('|--------GARRAFAS 1 LT - R$0.10 CENTAVOS---------|');
print('|--------GARRAFAS 2 LT - R$0.25 CENTAVOS---------|');
print('|------------------------------------------------|');
print('');

va1Lt = float(0.10)
va2Lt = float(0.25)

id = input('INSIRA A QUANTIDADE DE GARRAFAS DE 1LT PARA TROCA: ');
g_1Lt = float(id)
print('valor total 1 litro: R$ %1.2f'%(g_1Lt * va1Lt));
print()    
id2 = input('INSIRA A QUANTIDADE DE GARRAFAS DE 2LT PARA TROCA: ');
g_2Lt = float(id2)
print('valor total 2 litros: R$ %1.2f'%(g_2Lt * va2Lt));

total = g_1Lt * va1Lt + g_2Lt * va2Lt
print();
print('VALOR TOTAL A RECEBER: R$%1.2f' %(total));

import os
os.system("pause")

