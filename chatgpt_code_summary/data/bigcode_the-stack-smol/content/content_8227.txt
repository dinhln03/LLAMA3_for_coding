import sys, os
imgdir = sys.argv[1]
outfile = sys.argv[2]
flag = int(sys.argv[3])
fn = open(outfile, 'w')
fn.write("<html>\n")
fn.write("<body>\n")
namelist = os.listdir(imgdir)
namelist = sorted(namelist)
for idx, name in enumerate(namelist):
    #if idx > 10:
    #    break
    if flag:
        imname = "office1_%d.jpg"%idx
        impath = os.path.join(imgdir, imname)
    else:
        impath = os.path.join(imgdir, name)
    #impath = os.path.abspath(impath)
    #fn.write("<img src=\"%s\" width=\"640\" height=\"320\">\n"%impath)
    fn.write("<img src=\"%s\" width=\"960\" height=\"540\">\n"%impath)
    #print(impath)
fn.write("</body>\n")
fn.write("</html>")
fn.close()

