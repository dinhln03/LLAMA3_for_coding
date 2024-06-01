def deal(infilename,outfilename):
    infile = open(infilename)
    lines = infile.readlines()
    out = []
    for line in lines:
        line = line.split(',')
        val = line[-1][0]
        line = line[:-1]
        line.insert(0,val)
        out.append(line)
    print(out)
    str = ''
    for line in out:
        line_str =''
        for each in line:
            line_str = line_str + '{},'.format(each)
        str = str + line_str[:-1]+'\n'
    outfile=open(outfilename,'w')
    outfile.write(str)
deal('Haberman Data Set.txt','Haberman_data.txt')