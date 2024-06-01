#!/usr/bin/python

#By Sun Jinyuan and Cui Yinglu, 2021

foldx_exe = "/user/sunjinyuan/soft/foldx"


def getparser():
    parser = argparse.ArgumentParser(description=
                                     'To run Foldx PositionScan with multiple threads, make sure' +
                                     ' that you have the foldx and your pdb in the same floder')
    parser.add_argument("-s", '--pdbfile', help="The pdb file, the repaired one")
    parser.add_argument("-nt", '--number_threads', help="How many threads to run the Foldx")
    parser.add_argument("-c", '--chain_id', help="Chain ID")
    args = parser.parse_args()
    return args


def SOfile2mutlist(pdbname, chain_id, foldx_exe):
    AA_list = ["Q", "W", "E", "R", "T", "Y", "I", "P", "A", "S", "D", "F", "G", "H", "K", "L", "V", "N", "M"]
    try:
        SO_file = open("SO_" + pdbname.replace("pdb", "fxout"), "r")
    except FileNotFoundError:
    	os.system(foldx_exe + " --command=SequenceOnly --pdb=" + pdbname)
        #os.system("/data/home/jsun/mhetase/FoldX/foldx5 --command=SequenceOnly --pdb=" + pdbname)
        SO_file = open("SO_" + pdbname.replace("pdb", "fxout"), "r")
    mut_lst = []
    for line in SO_file:
        lst = line.replace("\n", "").split("\t")
        if len(lst) > 3:
            if lst[1] == chain_id:
                wild_AA = lst[3][0]
                for AA in AA_list:
                    if AA != wild_AA:
                        mut_lst.append(lst[3] + AA + ";")

    return mut_lst


def multi_threads(mut_lst, threads, pdbname, foldx_exe):
    t = len(mut_lst) // (int(threads) - 1)
    n = 0
    for i in range(0, len(mut_lst), t):
        submutlst = mut_lst[i:i + t]
        n = n + 1
        # indi_lst_name = "individual_list_"+str(n)+"_.txt"
        sub_dir_name = "Subdirectory" + str(n)
        indi_lst_name = sub_dir_name + "/individual_list.txt"
        os.mkdir(sub_dir_name)
        os.system("cp " + pdbname + " " + sub_dir_name)
        with open(indi_lst_name, "w+") as ind_lst:
            for mut in submutlst:
                ind_lst.write(mut + "\n")
            ind_lst.close()

        readablefilename = sub_dir_name + "/List_Mutations_readable.txt"
        with open(readablefilename, "a+") as readablefile:
            # KA12G
            x = 1
            for mut in submutlst:
                readablefile.write(str(x)+" "+mut[0]+" "+mut[2:-2]+" "+mut[-2]+"\n")
                #readablefile.write(str(x) + " " + mut[0] + "  " + mut[2:-1] + "  " + mut[-1] + "\n")
                x += 1
            readablefile.close()

        cfg = "command=BuildModel\npdb=" + pdbname + "\nmutant-file=individual_list.txt\nnumberOfRuns=5"
        cfg_name = sub_dir_name + "/BM_" + str(n) + ".cfg"
        with open(cfg_name, "w+") as cfg_file:
            cfg_file.write(cfg)
            cfg_file.close()
        with open("todo_list.sh", "a+") as todo_file:
            todo_file.write("cd " + sub_dir_name + "\n")
            todo_file.write("nohup "+foldx_exe+" -f " + "BM_" + str(n) + ".cfg" + " &\n")
            todo_file.write("cd ..\n")
            todo_file.close()


if __name__ == "__main__":
    import os
    import argparse

    args = getparser()

    pdbname = args.pdbfile
    threads = args.number_threads
    chain_id = args.chain_id
    
    #print(foldx_exe)

    with open("todo_list.sh", "w+") as todo_file:
        todo_file.close()

    mut_lst = SOfile2mutlist(pdbname, chain_id, foldx_exe)
    multi_threads(mut_lst, threads, pdbname, foldx_exe)

