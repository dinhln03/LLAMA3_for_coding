print('-'*20)
print('CADASTRE UMA PESSOA')
print('-'*20)
total = totalm = totalf = 0
while True:
    idade = int(input('Idade: '))
    if idade >= 18:
        total += 1
    sexo = ' '
    while sexo not in 'MF':
        sexo = str(input('Sexo: [M/F]')).strip().upper()[0]
        
# observações!
    if sexo == 'M':
        totalm += 1
            
    if sexo == 'F' and idade < 20:
        totalf +=1

    resposta = ' '
    while resposta not in 'SN':
        resposta = str(input('Quer continuar? [S/N]')).upper().strip()[0]
    if resposta == 'N':
        break
    
    
print('===== FIM DO PROGRAMA =====')
print(f'Total de pessoas com mais de 18 anos: {total}')
print(f'Ao todo temos {totalm} homens cadastrados')
print(f'E temos {totalf} mulher com menos de 20 anos')