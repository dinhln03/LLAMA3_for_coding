"""
Conjuntos são chamados de set's
- Set não possui duplicidade
- Set não possui valor ordenado
- Não são acessados via indice, ou seja, não são indexados

Bons para armazenar elementos são ordenação, sem se preocupar com chaves, valores e itens duplicados.

Set's são referenciados por {}
Diferença de set e dict
- Dict tem chave:valor
- Set tem apenas valor

---------------------------------------------------------------------------------------------------------------------
#  DEFENINDO SET
#  Forma 1
s = set ({1, 2, 3, 4, 5, 4, 5, 2, 1})  #  valores duplicados
print(type(s))
print(s)
#  OBS.: Ao criar um set, se uma valor estiver repetido, ele é ignorado, sem gerar erro.

#  Forma 2 - Mais comum
set = {1, 2, 3, 4, 5, 4, 5, 2, 1}  #  valores duplicados
print(type(set))
print(set)

#  Sem valores duplicados e sem ordenação entre eles
#  Pode-se colocar todos os tipos de dados

---------------------------------------------------------------------------------------------------------------------
#  PODE-SE ITERAR SOBRE UM SET

set = {1, 2, 3, 4, 5, 4, 5, 2, 1}
for valor in set:
    print(valor)

---------------------------------------------------------------------------------------------------------------------
#  USOS INTERESSANTES COM SET'S

#  Imagine que fizemos um formulario de cadastro de visitantes em um museu, onde as pessoas informam manualmente
#  sua cidade de origem

#  Nos adicionamos cada cidade em uma lista Python, ja que em lista pode-se adicionar novos elementos e ter repetição

cidade = ['Lavras', 'Bagé', 'Caçapava', 'Lavras', 'Bagé']
print(type(cidade))
print(cidade)
print(len(cidade))  #  para saber quantos visitantes teve
print(len(set(cidade)))  #  para saber quantas cidades distintas foram visitar

---------------------------------------------------------------------------------------------------------------------
#  ADICIONANDO ELEMENTOS EM UM SET

s = {1, 2, 3}
s.add(4)
print(s)

---------------------------------------------------------------------------------------------------------------------
#  REMOVANDO ELEMENTOS DE UM SET

#  Forma 1
conj = {1, 2, 3}
conj.remove(3)  #  se tentar remover um valor que não existe, gera um erro.
print(conj)

#  Forma 2
conj.discard(2)  #  se o elemento não existir, não vai gerar erro
print(conj)

---------------------------------------------------------------------------------------------------------------------
#  COPIANDO UM SET PARA OUTRO

conj = {1, 2, 3}

#  Forma 1 - Deep Copy (o novo conjunto fica independente)
novo = conj.copy()
print(novo)
novo.add(4)
print(conj, novo)

#  Forma 2 - Shallow Copy (o novo conjunto fica interligado ao primeiro)
novo2 = conj
print(novo2)
novo2.add(5)
print(conj, novo2)

---------------------------------------------------------------------------------------------------------------------
#  REMOVER TODOS OS DADOS DE UM SET
conj = {1, 2, 3}
conj.clear()
print(conj)

---------------------------------------------------------------------------------------------------------------------
#  METODOS MATEMÁTICOS DE CONJUNTOS

#  Dois conjuntos de estudantes, Python e Java.

python = {'Paulo', 'Luis', 'Marcos', 'Camila', 'Ana'}
java = {'Paulo', 'Fernando', 'Antonio', 'Joao', 'Ana'}

#  Precisamos juntar em um set, os alunos dos dois cursos, mas apenas nomes únicos

#  Forma 1 - usando union
unicos = python.union(java)
print(unicos)

#  Forma 2 - Usando o caracter pipe "|"
unicos2 = python|java
print(unicos2)

---------------------------------------------------------------------------------------------------------------------
#  GERANDO SET DE ESTUDANTES QUE ESTÃO NOS DOIS CURSOS

python = {'Paulo', 'Luis', 'Marcos', 'Camila', 'Ana'}
java = {'Paulo', 'Fernando', 'Antonio', 'Joao', 'Ana'}

#  Forma 1 - usando intersection
ambos = python.intersection(java)
print(ambos)

#  Forma 2 - usando &
ambos2 = python & java
print(ambos2)

---------------------------------------------------------------------------------------------------------------------
#  GERAR SET DE ESTUDANTES QUE ESTÃ EM UM CURSO, MAS QUE NÃO ESTÃO NO OUTRO

python = {'Paulo', 'Luis', 'Marcos', 'Camila', 'Ana'}
java = {'Paulo', 'Fernando', 'Antonio', 'Joao', 'Ana'}

so_python = python.difference(java)
print(so_python)

---------------------------------------------------------------------------------------------------------------------
#  SOMA*, MÁXIMO*, MÍNIMO*, TAMANHO.
#  * -> somente valores inteiros ou float

conj = {1, 2, 3, 4, 5}
print(sum(conj))
print(max(conj))
print(min(conj))
print(len(conj))

---------------------------------------------------------------------------------------------------------------------
"""
