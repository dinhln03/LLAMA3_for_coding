import sys
import os

if len(sys.argv) > 1:
    folder = sys.argv[1]
    jsfile = open("./" + folder + ".js", "w+")
    images = [f[:len(f)-4] for f in os.listdir("./" + folder) if f.endswith(".svg")]
    varnames = []
    for i in images:
        varname = "svg_" + i.replace('-', '_');
        varnames.append(varname)
        jsfile.write("import " + varname + " from './" + folder + "/" + i + ".svg';\n")
    jsfile.write("\n")
    jsfile.write("const " + folder + " = [" + ", ".join(varnames) + "];\n")
    jsfile.write("\n")
    jsfile.write("export default " + folder + ";")
    jsfile.close()
