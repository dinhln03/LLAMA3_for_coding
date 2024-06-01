## Ao arrumar os NCM nos registros nãoesteaindaC180 e C190 pode ocorrer duplicidade
## nos conjuntos de campos de acordo com o manual.
## No caso preciso juntar todos os registros com estas caracteristicas
import helper

def exec(conexao):
    cursor = conexao.cursor()

    print("RULE 02 - Inicializando",end=' ')

    select = " SELECT r0 FROM principal WHERE r1 = \"C010\" "
    select = cursor.execute(select)
    rselect = select.fetchall()
    rselect = [i[0] for i in rselect]
    #rselect.append(rselect[len(rselect)-1] + 1000)

    select = " SELECT max(r0) FROM principal WHERE "
    select = select + " r1 in (\"C191\",\"C195\") "
    select = select + " AND r0 > " + str(rselect[len(rselect)-1]) + " "
    temp = cursor.execute(select)
    temp = temp.fetchone()[0]
    rselect.append(temp == None and rselect[len(rselect)-1] + 1 or temp)

    n2 = rselect.pop(0)
    while len(rselect) > 0:
        print('-',end=' ')
        n1 = n2
        n2 = rselect.pop(0)

        # verifica se tem C190 repetido em cada C010
        select = " SELECT r2,r5,r6,r7,count(*) c "
        select = select + " FROM principal "
        select = select + " WHERE r1 = \"C190\" "
        select = select + " AND r0 BETWEEN " + str(n1) + " AND " + str(n2) + " "
        select = select + " GROUP BY r2,r5,r6,r7 "
        select = select + " HAVING COUNT(*) > 1 "
        select = select + " ORDER BY r5 "

        repetidos = cursor.execute(select)
        repetidos = repetidos.fetchall()
        ## se não tiver repetido, continua a olhar nos outros C010
        if len(repetidos) == 0:
            continue

        ## caso tenha C190 repetido é iterado nesses processos para concertar
        for i in repetidos:
            print('-',end=' ')
            ##/* pega o r0 de todos os C190 repetidos */
            select = " SELECT r0 FROM principal "
            select = select + " WHERE r1 = \"C190\" "
            select = select + " AND r0 BETWEEN " + str(n1) + " AND " + str(n2) + " "
            select = select + "	AND r2 = \"" + i[0] + "\" "
            select = select + " AND r5 = \"" + i[1] + "\" "
            select = select + " AND r6 = \"" + i[2] + "\" "
            
            r0s = cursor.execute(select)
            r0s = r0s.fetchall()
            r0s = [i[0] for i in r0s]

            primeiroID = r0s[0]
            qtrepetidos = len(r0s)

            ## coloca na lista todos os dados do C191 e C195 que fazem parte do C190
            lista = []
            for i2 in r0s:
                limit = helper.takeLimit(cursor,i2,"C190")
                select = " SELECT r0,r1,r2,r3,r4, "
                select = select + " (ROUND(CAST(replace(r5,',','.') AS FLOAT),2)) r5, "
                select = select + " r6,r7,r8,r9,r10,r11,r12 "
                select = select + " FROM principal WHERE "
                select = select + " r0 BETWEEN " + str(limit[0]) + " AND " + str(limit[1])
                select = select + " AND r1 in (\"C191\",\"C195\") "
                temp = cursor.execute(select)
                temp = temp.fetchall()
                lista.append(temp)

            if len(lista) > 1:
                lista1 = []
                for z in range(0,len(lista)):
                    lista1 = lista1 + lista[z]
                lista = []
                ids = []
                for i2 in lista1:
                    lista.append(i2[1:])
                    ids.append(i2[0])
                lista = list(set(lista))
                #ids.append(temp[1][0])

                ## deleta todos os registros para depois inserir os que não são repetidos
                delete = "DELETE FROM principal WHERE "
                delete = delete + " r0 BETWEEN " + str(ids[0]) + " AND " + str(ids[len(ids)-1]) + " "
                cursor.execute(delete)
                conexao.commit()

                ## insere os itens sem repetição e soma ao mesmo tempo o valor total do item
                valor_total = 0
                lista.sort()
                primeiroIDTemp = primeiroID
                for i3 in lista:
                    valor_total = valor_total + i3[4]
                    primeiroIDTemp = primeiroIDTemp + 1
                    stringt = "\",\"".join([str(iz) for iz in i3])
                    insert = ""
                    insert = insert + " INSERT INTO principal(r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12) "
                    insert = insert + " VALUES("
                    insert = insert + str(primeiroIDTemp) + ",\""
                    insert = insert + stringt.replace(".",",")
                    insert = insert + "\")"
                    cursor.execute(insert)
                    conexao.commit()

                ## atualiza valor total do C190
                update = ""
                update = update + " UPDATE principal SET "
                update = update + " r8 = \"" + str(round(valor_total / qtrepetidos,2)).replace(".",",") +"\""
                update = update + " where r0 = " + str(primeiroID)
                cursor.execute(update)
                conexao.commit()

    print("Finalizado")