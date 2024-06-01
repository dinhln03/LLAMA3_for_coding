
import sys
import datetime

def capitalize(string):
	return string[0].upper() + string[1:]

action = sys.argv[1]
file_path = sys.argv[2]
project_name = sys.argv[3]
namespace = sys.argv[4]

now = datetime.datetime.now()
date = now.strftime("%m-%d-%Y %H:%M:%S")

args = sys.argv[6:]

username = "Logan Rickert"

def new_class():
	file_name = sys.argv[5]

	cpp_file_path = file_path + "src/" + file_name + ".cpp"
	h_file_path = file_path + "include/" + file_name + ".h"

	if len(args) % 2 != 0:
		print "You must have an even amount of arguments!"
		sys.exit()

	parse = []

	for arg in xrange(0,len(args),2):
		parse.append([args[arg], args[arg + 1]])

	cpp_file_contents = None
	h_file_contents = None

	with open(cpp_file_path, 'r') as f:
		cpp_file_contents = f.read()

	with open(h_file_path, 'r') as f:
		h_file_contents = f.read()

	cpp_file_contents = cpp_file_contents.replace(
		"{{class_name}}", file_name
	)

	cpp_file_contents = cpp_file_contents.replace(
		"{{namespace}}", namespace
	)

	cpp_file_contents = cpp_file_contents.replace(
		"{{date}}", date
	)

	cpp_file_contents = cpp_file_contents.replace(
		"{{username}}", username
	)

	if len(args) > 0:
		construct_init = file_name + "::" + file_name + "("

		for key, value in parse:
			construct_init += key + " s" + capitalize(value) + ", "

		construct_init = construct_init[:-2] + ") {"

		cpp_file_contents = cpp_file_contents.replace(
			"{{construct_init}}", construct_init
		)

		construct_init_equals = ""

		for key, value in parse:
			construct_init_equals += "\t" + value + " = s" + capitalize(value) + ";\n"

		construct_init_equals += "}"

		cpp_file_contents = cpp_file_contents.replace(
			"{{construct_init_equals}}", construct_init_equals
		)

		getters_setters = ""

		for key, value in parse:
			getters_setters += """%s %s::get%s() {
	return %s;
}

void %s::set%s(%s s%s) {
	%s = s%s;
}

""" % (
			key,
			file_name,
			capitalize(value),
			value,
			file_name,
			capitalize(value),
			key,
			capitalize(value),
			value,
			capitalize(value)
		)

		getters_setters = getters_setters[:-2]

		cpp_file_contents = cpp_file_contents.replace(
			"{{getters_setters}}", getters_setters
		)
	else:
		cpp_file_contents = cpp_file_contents.replace(
			"\n{{construct_init}}\n", ""
		)
		cpp_file_contents = cpp_file_contents.replace(
			"{{construct_init_equals}}\n", ""
		)
		cpp_file_contents = cpp_file_contents.replace(
			"\n{{getters_setters}}\n", ""
		)

	with open(cpp_file_path, 'w') as f:
		f.write(cpp_file_contents)

	h_file_contents = h_file_contents.replace(
		"{{class_name_caps}}", file_name.upper()
	)

	h_file_contents = h_file_contents.replace(
		"{{class_name}}", file_name
	)

	h_file_contents = h_file_contents.replace(
		"{{username}}", username
	)

	h_file_contents = h_file_contents.replace(
		"{{namespace}}", namespace
	)

	h_file_contents = h_file_contents.replace(
		"{{date}}", date
	)

	if len(args) > 0:
		class_construct_full = file_name + "("

		for key, value in parse:
			class_construct_full += key + ", "

		class_construct_full = class_construct_full[:-2] + ");"

		h_file_contents = h_file_contents.replace(
			"{{class_construct_full}}", class_construct_full
		)

		getters_setters = ""

		for key, value in parse:
			getters_setters += "\t\t" + key + " get" + capitalize(value) + "();\n"

		getters_setters += '\n'

		for key, value in parse:
			getters_setters += "\t\tvoid set" + capitalize(value) + "(" + key + " s" + capitalize(value) + ");\n"

		h_file_contents = h_file_contents.replace(
			"{{getters_setters}}", getters_setters
		)

		class_fields = ""

		for key, value in parse:
			class_fields += "\t\t" + key + " " + value + ";\n"

		h_file_contents = h_file_contents.replace(
			"{{class_fields}}", class_fields
		)
	else:
		h_file_contents = h_file_contents.replace(
			"\n\t\t{{class_construct_full}}", ""
		)

		h_file_contents = h_file_contents.replace(
			"{{getters_setters}}\n", ""
		)

		h_file_contents = h_file_contents.replace(
			"{{class_fields}}", ""
		)

	with open(h_file_path, 'w') as f:
		f.write(h_file_contents)

def new_main():
	cpp_file_path = file_path + "/src/Main.cpp"

	cpp_file_contents = None
	h_file_contents = None

	with open(cpp_file_path, 'r') as f:
		cpp_file_contents = f.read()

	cpp_file_contents = cpp_file_contents.replace(
		"{{class_name}}", "Main"
	)

	cpp_file_contents = cpp_file_contents.replace(
		"{{namespace}}", namespace
	)

	cpp_file_contents = cpp_file_contents.replace(
		"{{username}}", username
	)

	cpp_file_contents = cpp_file_contents.replace(
		"{{date}}", date
	)

	with open(cpp_file_path, 'w') as f:
		f.write(cpp_file_contents)

if action == "class":
	new_class()
elif action == "namespace" or action == "project":
	new_main()