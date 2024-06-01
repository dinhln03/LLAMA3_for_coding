from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from Logger.models import Run, Process
from Logger.libs import log_dealer
import json

# Create your views here.
def listen(request):

    log_dealer(request)
    context_dict = {}
    response = render(request, 'index.html', context=context_dict)
    return response

def show_run(request, runId):

    context_dict = {}

    try:
        run = Run.objects.get(runId=runId)
        procs = Process.objects.filter(runId=runId)
        context_dict['processes'] = procs
        context_dict['run'] = run
    except Run.DoesNotExist:
        context_dict['processes'] = None
        context_dict['run'] = None

    return render(request, 'run.html', context_dict)

def show_process(request, runId, h1, h2):

    context_dict = {}

    try:
        uid = '/'.join([runId, h1, h2])
        proc = Process.objects.get(uid=uid)
        context_dict['process'] = proc
        context_dict['script_lines'] = proc.script.split('\n')
    except Run.DoesNotExist:
        context_dict['process'] = None

    return render(request, 'process.html', context_dict)

def index(request):

    started_list = Run.objects.filter(runStatus='started').order_by('-startedDatetime')[:10]
    success_list = Run.objects.filter(runStatus='success').order_by('-startedDatetime')[:10]
    failed_list = Run.objects.filter(runStatus='failed').order_by('-startedDatetime')[:10]

    context_dict = {'started':started_list, 'success':success_list, 'failed':failed_list}

    return render(request, 'index.html', context_dict)

def search(request):

    query = request.GET['q']
    if '/' in query:
        return HttpResponseRedirect("/")
    else:
        return HttpResponseRedirect(f"/run/{query}")

def about(request):

    context_dict = {}

    return render(request, 'about.html', context_dict)
