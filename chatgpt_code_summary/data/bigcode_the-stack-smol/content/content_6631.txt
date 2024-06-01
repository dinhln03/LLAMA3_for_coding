from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from registration.backends.simple.views import RegistrationView

from plan.models import Plan, UserProfile
from plan.forms import UserForm, RegistrationFormDHHD

def index(request):
	# Query the database for a list of all the plans currently stored.
	# Order the plans by the number of likes in descending order.
	# Retrieve the top 5 only - or all if less than 5.
	# Place the list in the context_dict dictionary which will be passed to the template engine.
	#popular_plan_list = Plan.objects.order_by('-views')[:3]
	popular_plan_list = reformat_plan(Plan.objects.filter(active=True).order_by('-views')[:3])
	# Most recent plans
	recent_plan_list = reformat_plan(Plan.objects.filter(active=True).order_by('-pub_date')[:3])
	
	context_dict = {'popular_plans': popular_plan_list, 'recent_plans': recent_plan_list}
	return render(request, 'index.html', context_dict)

	
def about(request):
	return render(request, 'about.html', {})

	
def reformat_plan(plan_list):
	for plan in plan_list:
		# Re-format the width and depth from floats to FF'-II" format.
		plan.width = str(int(plan.width)) + "'-" + str(round((plan.width - int(plan.width))*12)) + '"'
		plan.depth = str(int(plan.depth)) + "'-" + str(round((plan.depth - int(plan.depth))*12)) + '"'
		# Re-format bedrooms from float to int
		plan.bed = int(plan.bed)
		# Re-format bathrooms to int if the number of bathrooms is whole
		if not plan.bath%1:
			plan.bath = int(plan.bath)
	
	return plan_list

	
@login_required
def myplans(request):
	user_name = request.user.get_username()
	user = User.objects.get(username=user_name)
	profile = UserProfile.objects.get(user=user)
	plan_list = profile.fav_plans.all()
	return render(request, 'myplans.html', {'plan_list': plan_list})


class MyRegistrationView(RegistrationView):
	form_class = RegistrationFormDHHD
	def get_success_url(self, request, user):
		return '/'
	
	def register(self, request, **cleaned_data):
		new_user = RegistrationView.register(self, request, **cleaned_data)
		UserProfile(user=new_user).save()
		return new_user
