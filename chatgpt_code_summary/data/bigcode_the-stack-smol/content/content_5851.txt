#!/usr/bin/env python
import argparse, re, os
from StringIO import StringIO
import language

#*		Build instruction
#*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*#
def roundup(x, to=8):
	return x if x % to == 0 else x + to - x % to

def build_form(ins):
	form = ["%i"]

	for operand in ins[1:]:
		if operand.startswith("r"):
			form.append("%r")
		elif operand.startswith("q"):
			form.append("%q")
		elif operand.startswith("#") or operand.startswith("$"):
			if "." in operand:
				form.append("%f")
			else:
				form.append("%n")
		else:
			print "Error: Bad operand for instruction!"

	return form

def getnum(strnum):
	if strnum.startswith("0x"):
		return int(strnum[2:], 16)
	elif strnum.startswith("0b"):
		return int(strnum[2:], 2)
	else:
		return int(strnum)

def build_ins(line):
	line = line.strip()
	line = re.sub(" |,", " ", line)
	ins = line.split()

	hx = []

	# print ins

	if ins[0] in ["data", "byte", "d.byte", "d.int", "long"]:
		if ins[0] in ["data", "byte", "d.byte"]:
			hx.append( format(getnum(ins[1]), "08b") )
		elif ins[0] == "d.int":
			hx.append( format(getnum(ins[1]), "032b") )
			# print hx
		return [], [hx]
	else:
		# print ins
		# print build_form(ins)
		# print language.ins[ins[0]]
		form = build_form(ins)
		opcode = language.ins[ins[0]]["opcode"][language.ins[ins[0]]["form"].index(" ".join(form))]

		for f,op,i in zip(form, ins, range(len(ins))):
			if f == "%i":
				hx.append( format(opcode, "07b") )
			if f == "%r":
				hx.append( format(int(op[1:]), "07b") )
			if f == "%q":
				hx.append( format(int(op[1:])+(language.registers/2), "07b") )
			if f == "%f":
				hx.append( format( language.float_to_bits(float(op[1:])), "032b") )
			if f == "%n":
				if op[0] == "$":
					hx.append( op )
				elif i == 1:
					hx.append( format( (getnum(op[1:]) + (1 << 57)) % (1 << 57), "057b") )
				elif i == 2:
					hx.append( format( (getnum(op[1:]) + (1 << 50)) % (1 << 50), "050b") )
				elif i == 3:
					hx.append( format( (getnum(op[1:]) + (1 << 43)) % (1 << 43), "043b") )

		return [hx], []

def assemble(code):
	# read in the file
	if type(code) is file:
		lines = [l.rstrip().lower() for l in code.readlines()]
	else:
		lines = [l.rstrip().lower() for l in code.splitlines()]

	# remove comments
	lines = [l for l in lines if not l.lstrip().startswith("#")]
	# remove blank lines
	lines = [l for l in lines if not l.strip() == ""]

	# print lines

	labels = {}
	addr = 0
	ins = []
	data = []
	hexbytes = StringIO()

	# build the bit tuple for each instruction as well as label table
	for line in lines:
		# print line
		if line.startswith((" ", "\t")):
			i, d = build_ins(line)
			ins.extend(i)
			data.extend(d)

			if line.strip().startswith("d."):
				addr += 4
			else:
				addr = addr + 8
		elif line.endswith(":"):
			if "@" in line:
				key, address = line.split("@")
				labels[key] = int(address[:-1])
			else:
				labels[line[:-1]] = addr

	# print labels

	# second pass, find all labels and replace them with their program address component
	for inst in ins:
		# print inst
		for p,i in zip(inst, range(len(inst))):
			if p[0] == "$":
				if i == 1:
					inst[1] = format(labels[p[1:]], "057b")
				elif i == 2:
					inst[2] = format(labels[p[1:]], "050b")
				elif i == 3:
					inst[3] = format(labels[p[1:]], "043b")

	# convert the instructions to hex byte stream and write one instruction per line
	for inst in ins:
		inst = "".join(inst).ljust(64, "0")
		# print inst, len(inst)
		inst = format(int(inst, 2), "08x").rjust(16, "0")
		# print inst, len(inst)
		inst = " ".join(map(''.join, zip(*[iter(inst)]*2)))
		# print inst
		hexbytes.write(inst+"\n")

	# may need to fix this as we could have undefined behaviour if people put data before program
	# instructions!
	for d in data:
		d = "".join(d)
		d = d.rjust(roundup(len(d)), "0")
		# print d
		fstr = "0"+str(roundup(len(d)/4, 2))+"x"
		d = format(int(d, 2), fstr)
		d = " ".join(map(''.join, zip(*[iter(d)]*2)))

		hexbytes.write(d+"\n")

	return hexbytes.getvalue().strip()


#*		Main
#*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*#
if __name__ == "__main__":
	ap = argparse.ArgumentParser(description="SuperScalar assembler")
	ap.add_argument("file",
		type=str,
		nargs=1,
		help="Assembler file to assemble.")
	ap.add_argument("--out", "-o",
		type=str,
		nargs=1,
		metavar="FILE",
		dest="output",
		help="Specify an output file for the machine code")
	args = ap.parse_args()

	if args.output:
		hex_path = args.output[0]
	else:
		hex_path = os.path.splitext(args.file[0])[0]+".hex"
	
	if not os.path.exists(os.path.dirname(hex_path)):
		os.makedirs(os.path.dirname(hex_path))

	fp = open(args.file[0], "r")
	fpx = open(hex_path, "w")

	language.assign_opcodes()
	fpx.write(assemble(fp))

	# print args.file[0],"->",hex_path,"("+str(addr)+" bytes)"
