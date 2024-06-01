from minpiler.std import M

i = 0
M.label .loop
M.print(i)
i += 1
if i < 10:
    M.jump .loop
