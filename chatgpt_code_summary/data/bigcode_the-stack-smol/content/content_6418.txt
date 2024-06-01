import sys
sys.path.append('..')
from utilities import jamfconfig
from utilities import apirequests
from computergroups import computergroups
import xml.etree.ElementTree as etree

jss_api_base_url = jamfconfig.getJSS_API_URL()
#print("JSS API Base URL: {}".format(jss_api_base_url))


def cleanupOutput(inputString):
	#print str(inputString)
	return inputString.replace(u"\u2018", "'").replace(u"\u2019", "'").replace(u"\u201c", "\"").replace(u"\u201d", "\"")


def getAllPolicies(username, password):
	''' List all policies in JSS to screen '''
	#print(username)

	print "We're Refactored!  Getting All JAMF Policies..."
	reqStr = jss_api_base_url + '/policies'

	r = apirequests.sendAPIRequest(reqStr, username, password, 'GET')

	if r == -1:
		return

	baseXml = r.read()
	responseXml = etree.fromstring(baseXml)

	for policy in responseXml.findall('policy'):
		policyName = policy.find('name').text
		policyID = policy.find('id').text

		print 'Policy ID: ' + policyID + ',  ' + 'Policy Name: ' + policyName + '\n'


def getPolicybyId(policyid, username, password):
	''' Method to search for Policy ID by ID number and return General Policy Information, Scoping Information, and Package Configuration information - send results to Stdout '''

	print 'Running refactored getpolicybyid ...'
	reqStr = jss_api_base_url + '/policies/id/' + policyid

	r = apirequests.sendAPIRequest(reqStr, username, password, 'GET')

	if r != -1:

		baseXml = r.read()
		responseXml = etree.fromstring(baseXml)

		general = responseXml.find('general')

		## General Policy Information
		name = general.find('name').text
		policy_id = general.find('id').text
		enabled = general.find('enabled').text
		trigger = general.find('trigger').text
		frequency = general.find('frequency').text

		print '\nGENERAL POLICY INFORMATION: '
		print 'Policy Name: ' + str(name)
		print 'Policy ID #: ' + str(policy_id)
		print 'Policy is Enabled: ' + str(enabled)
		print 'Policy Trigger: ' + str(trigger)
		print 'Policy Frequency: ' + str(frequency)

		## Policy Scope Information
		scope = responseXml.find('scope')
		allcomputers = scope.find('all_computers').text
		groups = scope.find('computer_groups')
		comp_groups = []
		computers = scope.find('computers')
		members = []

		## Add Header Row for output for info categories
		# headerRow = "Computer Name, JSS ID"
		# members += [ headerRow ]

		for computer in computers.findall('computer'):
			# compID = computer.find('id').text
			name = computer.find('name').text
			computerInfo = str(name)
			computerInfo = cleanupOutput(computerInfo)
			#print computerInfo.encode('ascii', 'ignore')
			members += [ computerInfo ]

		for g in groups.findall('computer_group'):
			group_name = g.find('name').text
			groupInfo = str(group_name)
			comp_groups += [ groupInfo ]


		print '\nPOLICY SCOPE INFORMATION:'
		print 'Scoped to All Computers: ' + str(allcomputers)
		print '\nCOMPUTER GROUPS IN SCOPE: '
		print '\n'.join (sorted(comp_groups))

		if members:
			print '\nADDITIONAL COMPUTERS IN SCOPE: '
			print '\n'.join (sorted(members))
			print '\nTotal Computers in Scope: ' + str(len(members))

		## Package Configuration Information
		pkgconfig = responseXml.find('package_configuration')
		packages = pkgconfig.find('packages')
		pkgheaderRow = "Package Name"
		pkglist = []

		for pkg in packages.findall('package'):
			pkg_name = pkg.find('name').text
			pkg_action = pkg.find('action').text
			pkgInfo = str(pkg_name) + ', ' + str(pkg_action)
			pkgInfo = cleanupOutput(pkgInfo)
			pkglist += [ pkgInfo ]


		print '\nPACKAGE CONFIGURATION: '
		print '\n'.join (sorted(pkglist))

	else:
		print 'Failed to find policy with ID ' + policyid


def listAllPolicies(username, password):
	''' List all policies in JSS - for function use  '''

	reqStr = jss_api_base_url + '/policies'

	r = apirequests.sendAPIRequest(reqStr, username, password, 'GET')

	if r == -1:
		return

	baseXml = r.read()
	responseXml = etree.fromstring(baseXml)
	PoliciesList = []
	for policy in responseXml.findall('policy'):
		policyName = policy.find('name').text
		policyID = policy.find('id').text
		PoliciesList.append({'name': policyName, 'id': policyID})

	return PoliciesList


def listAllPolicyIds(username, password):
	''' List all policy IDs in JSS - for function use - returns a list of Policy ID #s  '''

	reqStr = jss_api_base_url + '/policies'

	r = apirequests.sendAPIRequest(reqStr, username, password, 'GET')

	if r == -1:
		return

	baseXml = r.read()
	responseXml = etree.fromstring(baseXml)
	PolicyIDList = []
	for policy in responseXml.findall('policy'):
		policyID = policy.find('id').text
		PolicyIDList.append(policyID)

	return PolicyIDList


def listPolicyStatusbyId(policyid, username, password):
	''' Function to search for Policy ID by ID number and return status results for
	use in functions '''


	reqStr = jss_api_base_url + '/policies/id/' + policyid + '/subset/General'

	r = apirequests.sendAPIRequest(reqStr, username, password, 'GET')

	if r != -1:

		baseXml = r.read()
		responseXml = etree.fromstring(baseXml)
		general = responseXml.find('general')
		status = general.find('enabled').text

	return status


def listPolicyNamebyId(policyid, username, password):
	''' Function to search for Policy ID by ID number and return name for
	use in functions '''

	reqStr = jss_api_base_url + '/policies/id/' + policyid + '/subset/General'

	r = apirequests.sendAPIRequest(reqStr, username, password, 'GET')

	if r != -1:

		baseXml = r.read()
		responseXml = etree.fromstring(baseXml)
		general = responseXml.find('general')
		name = general.find('name').text

	return name


def listPolicyScopebyId(policyid, username, password):
	''' Function to search for Policy ID by ID number and return scope details as a
	dict for use in functions '''

	reqStr = jss_api_base_url + '/policies/id/' + policyid + '/subset/Scope'

	r = apirequests.sendAPIRequest(reqStr, username, password, 'GET')

	scopeData = []

	if r != -1:

		baseXml = r.read()
		responseXml = etree.fromstring(baseXml)
		scope = responseXml.find('scope')
		allcomputers = scope.find('all_computers').text
		groups = scope.find('computer_groups')
		comp_groups = []
		comp_groupIDs = []
		computers = scope.find('computers')
		members = []
		scope_details = {}

		for comp in computers.findall('computer'):
			if comp.find('name').text:
				name = comp.find('name').text
				members.append(name)

		for g in groups.findall('computer_group'):
			if g.find('name').text:
				group_name = g.find('name').text
				groupID = computergroups.getComputerGroupId(group_name, username, password)
				comp_groups.append(group_name)
				comp_groupIDs.append(groupID)

		scope_details = { "Policy ID: ": policyid, "All computers?: ": allcomputers, "Computer groups: ": comp_groups, "Computer group IDs: ": comp_groupIDs, "Specific computers: ": members }

	return scope_details


def listPolicyPackagesbyId(policyid, username, password):
	''' Function to search for Policy ID by ID number and return package details as a list
	for use in functions '''

	reqStr = jss_api_base_url + '/policies/id/' + policyid + '/subset/Packages'

	r = apirequests.sendAPIRequest(reqStr, username, password, 'GET')

	pkglist = []

	if r != -1:

		baseXml = r.read()
		responseXml = etree.fromstring(baseXml)
		pkgconfig = responseXml.find('package_configuration')
		packages = pkgconfig.find('packages')

		if packages.findall('package'):
			for pkg in packages.findall('package'):
				pkg_name = pkg.find('name').text
				pkglist.append(pkg_name)


	return pkglist


def listPolicybyId(policyid, username, password):
	''' Method to search for Policy ID by ID number and return General Policy Information, Scoping Information, and Package Configuration information - for use in functions '''

	reqStr = jss_api_base_url + '/policies/id/' + policyid

	r = apirequests.sendAPIRequest(reqStr, username, password, 'GET')

	if r != -1:

		baseXml = r.read()
		responseXml = etree.fromstring(baseXml)

		policyDict = {}

		## General Policy Information
		general = responseXml.find('general')
		polname = general.find('name').text
		policy_id = general.find('id').text
		enabled = general.find('enabled').text
		trigger = general.find('trigger').text
		frequency = general.find('frequency').text


		## Policy Scope Information
		scope = responseXml.find('scope')
		allcomputers = scope.find('all_computers').text
		groups = scope.find('computer_groups')
		comp_groups = []
		computers = scope.find('computers')
		members = []

		for computer in computers.findall('computer'):
			name = computer.find('name').text
			computerInfo = name.encode("utf-8")
			# computerInfo = cleanupOutput(computerInfo)
			# members.append(name)
			members += [ computerInfo ]

		for g in groups.findall('computer_group'):
			group_name = g.find('name').text
			groupInfo = str(group_name)
			comp_groups += [ groupInfo ]


		## Package Configuration Information
		pkgconfig = responseXml.find('package_configuration')
		packages = pkgconfig.find('packages')
		pkglist = []

		for pkg in packages.findall('package'):
			pkg_name = pkg.find('name').text
			pkg_action = pkg.find('action').text
			pkglist.append({"Package Name": pkg_name, "Package Action": pkg_action})

		## Add policy details to policyDict and return
		policyDict = { "Policy Name": polname,
					"Policy ID": policy_id,
					"Policy Enabled": enabled,
					"Policy Trigger": trigger,
					"Policy Frequency": frequency,
					"All Computers in Scope": allcomputers,
					"Scoped Computers": members,
					"Scoped Computer Groups": comp_groups,
					"Package Configuration": pkglist
					 }

		return policyDict

	else:
		print 'Failed to find policy with ID ' + policyid



