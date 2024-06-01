# Un meșter trebuie să paveze întreaga pardoseală a unei bucătării cu formă
# dreptunghiulară de dimensiune L_1×L_2 centimetri, cu plăci de gresie
# pătrate, toate cu aceeași dimensiune. Știind că meșterul nu vrea să taie nici o
# placă de gresie și vrea să folosească un număr minim de plăci, să se
# determine dimensiunea plăcilor de gresie de care are nevoie, precum și
# numărul lor. De exemplu, dacă L_1=440 cm și L_2=280 cm, atunci meșterul
# are nevoie de 77 de plăci de gresie, fiecare având latura de 40 cm.

L_1 = int(input('L_1: '))
L_2 = int(input('L_2: '))

aria = L_1*L_2
# aflam cmmdc dintre latruri
while L_1 != L_2:
    if L_1 > L_2:
        L_1 -= L_2
    else:
        L_2 -= L_1

dim = L_1
nr = aria/dim**2
print(dim, nr)
