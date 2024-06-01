from django.shortcuts import render, redirect
from django.core.urlresolvers import reverse
from .forms import EventCreateForm, NoteCreateForm, PropertyCreateForm, FileUploadForm,AlertCreateForm
from models import Event, Property, Note, File, Alert
from wsgiref.util import FileWrapper
from django.http import HttpResponse, HttpResponseBadRequest
from django.contrib.auth.decorators import login_required
import datetime, mimetypes

# Create your views here.

@login_required(login_url='/accounts/login/')
def index(request):
	return render(request, "main/base.html")

@login_required(login_url='/accounts/login/')
def dashboard(request):
		return render(request, "main/base.html")

def properties(request):
	context = {
			"properties" : Property.objects.filter(user = request.user)
		}
	return render (request, "main/properties.html", context)

def sidebar(request):
	today = datetime.datetime.now()
	yesterday = today - datetime.timedelta(days=1)
	print today

	alerts = Alert.objects.filter(event__property__user=request.user, when__gt=yesterday).order_by("when")[:5]
	notes = Note.objects.filter(event__property__user=request.user).order_by("-created_at")[:5]
	for alert in alerts:
		print alert.when
	context = {
			"alerts" : alerts,
			"notes" : notes,
			"today" : today
		}
	return render(request, "main/sidebar.html", context)

def add_property(request):
	if request.method == "POST":
		prop_form = PropertyCreateForm(request.POST)
		if prop_form.is_valid():
			prop = prop_form.save(commit=False)
			prop.user = request.user
			prop.save()
			context = {
				"properties" : Property.objects.filter(user = request.user)
			}
			print "valid form"
			return HttpResponse(prop.id)
		else:
			context={
				'form':prop_form
			}
			print "invalid form"
			return HttpResponseBadRequest(render (request,'main/add_property.html',context))
	else:
		context={
			'form':PropertyCreateForm()
		}
		print "GET"
		return render(request,'main/add_property.html',context)


def event(request,event_id, prop_id):
	property = Property.objects.get(pk=prop_id)
	event = Event.objects.get(pk=event_id)
	notes = event.note_set.all()
	alerts = event.alert_set.all()

	context={
	'prop_id':prop_id,
	'event':event,
	"notes":notes,
	'alerts':alerts,
	'property':property
	}
	return render(request, 'main/event.html',context)

def events(request, prop_id):
	property = Property.objects.get(pk=prop_id)
	events = property.event_set.all()
	context={
	'property' : property,
	'events':events
	}
	return render(request, 'main/events.html',context)

def get_file(request,file_id):
	file = File.objects.get(pk=file_id)
	mimetype = mimetypes.guess_type(file.docfile.name)
	response = HttpResponse(content_type=mimetype[0])
	response['Content-Disposition'] = 'inline; filename=' + file.docfile.name.split('/')[-1]
	response['Accept-Ranges'] = 'bytes'
	response['Content-Length'] = file.docfile.size
	response.write(file.docfile.read())
	return response


def add_event(request, prop_id):
	if request.POST:
		event_form = EventCreateForm(request.POST)
		if event_form.is_valid():
			event = event_form.save(commit=False)
			event.property = Property.objects.get(pk=prop_id)
			event.save()
			return HttpResponse(event.id)
		else:
			context={
				'form':event_form,
				'prop_id':prop_id
			}
			return HttpResponseBadRequest(render (request,'main/add_event.html',context))
	else:
		context={
			'form':EventCreateForm(),
			'prop_id':prop_id
		}
		return render(request,'main/add_event.html',context)


def note(request,event_id,prop_id,note_id):
	note = Note.objects.get(pk=note_id)
	documents = note.file_set.all()
	docNames = []
	for document in documents:
		docNames.append((document.id,document.docfile.name.split('/')[-1]))
	print docNames
	form = FileUploadForm()
	property = Property.objects.get(pk=prop_id)
	event = Event.objects.get(pk=event_id)
	context={'form':form, 'documents': documents,'event_id':event_id,
	'prop_id':prop_id,"note_id":note.id, 'note':note, 'event':event, 'property':property, "docNames":docNames}
	return render(request, 'main/note.html', context)

def notes(request,event_id,prop_id):
		event = Event.objects.get(pk=event_id)
		notes = event.note_set.all()
		context={
		'event_id':event_id,
		'prop_id':prop_id,
		}
		return render(request, 'main/note.html', context)

def add_note(request,prop_id, event_id):
	if request.POST:
		note_form = NoteCreateForm(request.POST)
		if note_form.is_valid():
			note = note_form.save(commit=False)
			note.event = Event.objects.get(pk=event_id)
			note.save()
			return HttpResponse(note.id)
		else:
			context={
				'form':note_form,
				'prop_id':prop_id,
				'event_id':event_id
			}
			return HttpResponseBadRequest(render (request,'main/add_note.html',context))
	else:
		context={
			'form':NoteCreateForm(),
			'prop_id':prop_id,
			'event_id':event_id
		}
		return render(request,'main/add_note.html',context)

def update_note(request,prop_id, event_id):
	print ('update')
	if request.POST:
		name = request.POST['name']
		name = request.POST['comment']
		note = Event.objects.get(pk=event_id)
		note.name=name
		note.comment=comment
		note.save()
		return HttpResponse(note.id)



def add_file(request,prop_id, event_id, note_id):
	if request.method == 'POST':
		form = FileUploadForm(request.POST, request.FILES)
		note = Note.objects.get(pk=note_id)
		if form.is_valid():
			newdoc = File(docfile=request.FILES['docfile'] )
			newdoc.note = note
			newdoc.save()
			return HttpResponse("added file")
		else:
			form = FileUploadForm()
		documents = File.objects.all()
		context={'form':form, 'documents': documents,'event_id':event_id,
		'prop_id':prop_id,"note_id":note_id}
		return HttpResponseBadRequest(render (request,'main/note.html',context))

def alert(request,event_id,prop_id,alert_id):
	alert = Alert.objects.get(pk=alert_id)
	form = AlertCreateForm()
	property = Property.objects.get(pk=prop_id)
	event = Event.objects.get(pk=event_id)
	context={'form':form, 'event_id':event_id,
	'prop_id':prop_id,"alert_id":alert.id, 'alert':alert, 'property':property, 'event':event}
	return render(request, 'main/alert.html', context)

def add_alert(request,prop_id, event_id):
	if request.POST:
		alert_form = AlertCreateForm(request.POST)
		if alert_form.is_valid():
			alert = alert_form.save(commit=False)
			alert.event = Event.objects.get(pk=event_id)
			alert.save()
			return HttpResponse(alert.id)
		else:
			context={
				'form':alert_form,
				'prop_id':prop_id,
				'event_id':event_id
			}
			return HttpResponseBadRequest(render (request,'main/add_alert.html',context))
	else:
		context={
			'form':AlertCreateForm(),
			'prop_id':prop_id,
			'event_id':event_id
		}
		return render(request,'main/add_alert.html',context)
