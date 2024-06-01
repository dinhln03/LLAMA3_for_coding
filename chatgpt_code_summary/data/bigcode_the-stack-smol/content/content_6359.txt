import re,sys

class Instruction:
	def __init__(self,defn):
		m = re.match("^([A-Fa-f0-9\-\,]+)\s+\"(.*?)\"\s+(.*)$",defn)
		assert m is not None,"Bad line "+defn
		range = m.group(1)
		range = range+"-"+range if len(range) == 2 else range
		range = range+",1" if len(range) == 5 else range
		self.first = int(range[:2],16)
		self.last = int(range[3:5],16)
		self.step = int(range[-1],16)
		self.name = m.group(2).strip()
		self.code = m.group(3).strip()
		#print(defn,range,self.first,self.last,self.step,self.getOpcodes())

	def getOpcodes(self):
		return range(self.first,self.last+self.step,self.step)

	def getMnemonics(self,opcode):
		base = self.name
		base = self.process(base,opcode)
		return base.lower()

	def getCode(self,opcode,type = "C"):
		base = self.process(self.code,opcode)
		if (opcode & 0xF0) == 0xC0:
			base = base + ";$CYCLES++"
		isFirst = True
		while base.find("$") >= 0:
			if isFirst:
				mWord = "$DF"
				isFirst = False
			else:
				m = re.search("(\$[A-Za-z]+)",base)
				mWord = m.group(1)			
			if type == "C":
				base = base.replace(mWord,mWord[1:].upper())
			elif type == "T":
				base = base.replace(mWord,"this."+mWord[1:].lower())
			else:
				raise Exception()
		while base.find(";;") >= 0:
			base = base.replace(";;",";")
		if base[0] == ';':
			base = base[1:]
		return base

	def process(self,s,opc):
		s = s.replace("@R","{0:X}".format(opc & 0x0F))
		s = s.replace("@P","{0:X}".format(opc & 0x07))
		s = s.replace("@E","{0:X}".format((opc & 0x03)+1))

		s = s.replace("@BRANCH","$R[$P] = ($R[$P] & 0xFF00) | $T8")
		s = s.replace("@LBRANCH","$R[$P] = $T16")
		s = s.replace("@FETCH16","$T16=$FETCH();$T16=($T16 << 8)|$FETCH()")
		s = s.replace("@LSKIP","$R[$P] = ($R[$P]+2) & 0xFFFF")

		if s[:4] == "@ADD":
			params = ["("+x+")" for x in s.strip()[5:-1].split(",")]
			s = "$T16 = "+("+".join(params))+";$D = $T16 & 0xFF;$DF = ($T16 >> 8) & 1"
			#print(s,params)
			#sys.exit(0)
		return s

src = open("1802.def").readlines()
src = [x if x.find("//") < 0 else x[:x.find("//")] for x in src]
src = [x.replace("\t"," ").strip() for x in src]
src = [x for x in src if x != ""]

instructions = [ None ] * 256
for l in src:
	instr = Instruction(l)
	for opc in instr.getOpcodes():
		assert instructions[opc] is None,"Duplicate opcode : "+l
		instructions[opc] = instr


mList = ",".join(['"'+instructions[x].getMnemonics(x)+'"' for x in range(0,256)])

open("_1802_mnemonics.h","w").write("{ "+mList+ " };\n\n")

h = open("_1802_case.h","w")
for i in range(0,256):
	h.write("case 0x{0:02x}: /*** {1} ***/\n".format(i,instructions[i].getMnemonics(i)))
	h.write("    "+instructions[i].getCode(i,"C")+";break;\n")
h.close()

h = open("_1802_opcodes.ts","w")
h.write("class CPU1802_Opcodes extends CPU1802_Base {\n\n")
h.write("public getOpcodeList():Function[] {\n    ")
h.write(",".join("opcode_{0:02x}()".format(n) for n in range(0,256)))
h.write("\n}\n\n")
for i in range(0,256):
	h.write("private opcode_{0:02x}(): void {{ /*** {1} ***/\n".format(i,instructions[i].getMnemonics(i)))
	h.write("    "+instructions[i].getCode(i,"T")+";\n}\n")
h.write("}\n")
h.close()

h = open("_1802_ports.h","w")
for p in range(1,8):
	h.write("#ifndef INPUT{0}\n#define INPUT{0}() (0)\n#endif\n".format(p))
	h.write("#ifndef OUTPUT{0}\n#define OUTPUT{0}(x) {{}}\n#endif\n".format(p))
for p in range(1,5):
	h.write("#ifndef EFLAG{0}\n#define EFLAG{0}() (0)\n#endif\n".format(p))
h.write("#ifndef UPDATEQ\n#define UPDATEQ(x) {{}}\n#endif\n".format(p))
h.close()