#
# Copyright 2020 by 0x7c2, Simon Brecht.
# All rights reserved.
# This file is part of the Report/Analytic Tool - CPme,
# and is released under the "Apache License 2.0". Please see the LICENSE
# file that should have been included as part of this package.
#

from templates import check
import func

class check_performance_ispredundancy(check):
	page         = "Health.Firewall"
	category     = "Information"
	title        = "ISP Redundancy"
	isFirewall   = True
	isManagement = False
	minVersion   = 8020
	command      = "cpstat fw | grep -A5 'ISP link table' | grep '|'"
	isCommand    = True

	def run_check(self):
		for line in self.commandOut:
			fields = line.split('|')
			ispname = fields[1]
			ispstatus = fields[2]
			isprole = fields[3]
			if ispname != "Name":
				ipstatus = "WARN"
				if ispstatus == "OK":
					state = "PASS"
				self.add_result(self.title + " (Name: " + ispname + ")", state, "Role: " + isprole)
			else:
				self.add_result(self.title, "PASS", "disabled")



class check_performance_securexl_sum(check):
	page         = "Health.SecureXL"
	category     = "Information"
	title        = "SecureXL"
	isFirewall   = True
	isManagement = False
	minVersion   = 8020
	command      = "fwaccel stat | grep -v Template"
	isCommand    = True

	def run_check(self):
		for line in self.commandOut:
			state = "FAIL"
			data = line.strip('\n').split('|')
			if len(data) < 4 or data[1].replace(" ","") == "" or data[1].replace(" ","") == "Id":
				continue
			id = data[1].replace(" ", "")
			type = data[2].replace(" ", "")
			status = data[3].replace(" ", "")
			if status != "enabled":
				state = "WARN"
			else:
				state = "PASS"
				feature = True
			self.add_result(self.title + " (Instance: " + id + ", Name: " + type + ", Status: " + status + ")", state, "")


class check_performance_securexl_templates(check):
	page         = "Health.SecureXL"
	category     = "Templates"
	title        = "SecureXL"
	isFirewall   = True
	isManagement = False
	minVersion   = 8020
	command      = "fwaccel stat| grep Templates | sed s/\ \ */\/g| sed s/Templates//g"
	isCommand    = True

	def run_check(self):
		for line in self.commandOut:
			state = "FAIL"
			data = line.strip('\n').split(":")
			if len(data) < 2:
				continue
			if "disabled" in data[1]:
				state = "WARN"
			if "enabled" in data[1]:
				state = "PASS"
			self.add_result(self.title + " (" + data[0] + " Templates)", state, data[1]) 



class check_performance_securexl_statistics(check):
	page         = "Health.SecureXL"
	category     = "Statistics"
	title        = "SecureXL"
	isFirewall   = True
	isManagement = False
	minVersion   = 8020
	command      = "fwaccel stats -s  | sed 's/  */ /g' | sed 's/\t/ /g'"
	isCommand    = True

	def run_check(self):
		for line in self.commandOut:
			state = "PASS"
			data = line.strip('\n').split(":")
			if len(data) < 2:
				continue
			field = data[0].strip(' ')
			valraw = data[1].strip(' ').split(" ")
			valnum = valraw[0]
			valper = int(str(valraw[1]).replace('(','').replace(')','').replace('%',''))
			if "Accelerated conns" in field and valper < 30:
				state = "WARN"
			if "Accelerated pkts" in field and valper < 50:
				state = "WARN"
			if "F2Fed" in field and valper > 40:
				state = "FAIL"
			self.add_result(self.title + " (" + field + ")", state, valnum + "(" + str(valper) + "%)") 

class check_performance_vpn_accel(check):
	page         = "Health.SecureXL"
	category     = "Information"
	title        = "SecureXL VPN Acceleration"
	isFirewall   = True
	isManagement = False
	minVersion   = 8020
	command      = "vpn accel stat"
	isCommand    = True

	def run_check(self):
		found = False
		for line in self.commandErr:
			if "acceleration is enabled" in line:
				self.add_result(self.title, 'PASS', line.strip())
				found = True
		if not found:
			self.add_result(self.title, 'FAIL', str(self.commandOut) + str(self.commandErr))


