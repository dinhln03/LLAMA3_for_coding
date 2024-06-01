from django import forms
from django.conf import settings
from django.http import HttpResponse, HttpResponseRedirect, Http404
from Manager import models as m 
from . import templater

def process_request(request):
	group = request.urlparams[0]
	
	try:
		members = m.Membership.objects.filter(group = group)
		print(members)
		users = list()
		uWeights = list()
		print(users)
		for g in members:
			users.append(g.member)
		print(users)
		userWeights = m.Weight.objects.filter(user__in=(users))
		print(userWeights)
		for i in users:
			try:
				uw = m.Weight.objects.filter(user = i).latest('dateWeighed')
				print(uw)
				uWeights.append(uw.weightLost)
			except:
				print('no weight')
				uWeights.append('No weight entry yet')
		print(uWeights)
	except:
		members = False
	tvars = {
	'members' : users,
	'userWeights' : uWeights
	}
	return templater.render_to_response(request, 'groupMembers.html', tvars)
	