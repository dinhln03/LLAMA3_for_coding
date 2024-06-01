
#1KiB
with open("Makeflow1KiB","w+") as f:
	f.write("CORES=1\nMEMORY=1000\nDISK=1\n\n")
	for x in range(100):
		f.write("out%i.txt:generate\n\t./generate out%i.txt %i\n\n"%(x,x,1024*1))

#10KiB
with open("Makeflow10KiB","w+") as f:
	f.write("CORES=1\nMEMORY=1000\nDISK=1\n\n")
        for x in range(100):
                f.write("out%i.txt:generate\n\t./generate out%i.txt %i\n\n"%(x,x,1024*10))

#100KiB
with open("Makeflow100KiB","w+") as f:
	f.write("CORES=1\nMEMORY=1000\nDISK=1\n\n")
        for x in range(100):
                f.write("out%i.txt:generate\n\t./generate out%i.txt %i\n\n"%(x,x,1024*100))

#1MiB
with open("Makeflow1MiB","w+") as f:
	f.write("CORES=1\nMEMORY=1000\nDISK=2\n\n")
        for x in range(100):
                f.write("out%i.txt:generate\n\t./generate out%i.txt %i\n\n"%(x,x,1024*1024*1))

#10MiB
with open("Makeflow10MiB","w+") as f:
	f.write("CORES=1\nMEMORY=1000\nDISK=20\n\n")
        for x in range(100):
                f.write("out%i.txt:generate\n\t./generate out%i.txt %i\n\n"%(x,x,1024*1024*10))

#100MiB
with open("Makeflow100MiB","w+") as f:
	f.write("CORES=1\nMEMORY=1000\nDISK=200\n\n")
        for x in range(100):
                f.write("out%i.txt:generate\n\t./generate out%i.txt %i\n\n"%(x,x,1024*1024*100))

#1GiB
with open("Makeflow1GiB","w+") as f:
	f.write("CORES=1\nMEMORY=1000\nDISK=2000\n\n")
        for x in range(100):
                f.write("out%i.txt:generate\n\t./generate out%i.txt %i\n\n"%(x,x,1024*1024*1024*1))

#10GiB
with open("Makeflow10GiB","w+") as f:
	f.write("CORES=1\nMEMORY=1000\nDISK=10738\n\n")
        for x in range(100):
                f.write("out%i.txt:generate\n\t./generate out%i.txt %i\n\n"%(x,x,1024*1024*1024*10))
