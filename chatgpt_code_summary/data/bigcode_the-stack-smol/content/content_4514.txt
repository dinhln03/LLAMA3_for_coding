
from django.shortcuts import render

# Create your views here.
from .models import BallotText, Candidate, District


def index(request):
    districts = District.objects.all
    context = {'districts': districts}
    return render(request, 'vote/index.html', context)


def ballot(request, district_num):
    ballot_list = BallotText.objects.all
    context = {'ballot_list': ballot_list}
    return render(request, 'vote/'+str(district_num)+'/ballot.html', context)


def votetotals(request):
    candidates = Candidate.objects.all
    return render(request, 'vote/votetotals.html', {"candidates": candidates})

def tally(request):

    if request.method == "POST":
        list = request.POST
        candidates_id = list.items()
        all_candidates = Candidate.objects.all()
        for id in candidates_id:
            print(id[1])
            for candidate in all_candidates:
                print(candidate.candidate_text)
                if candidate.candidate_text == id[1]:
                    print(candidate.candidate_text + " " + id[1])
                    candidate.votes += 1
                    candidate.save()
        return render(request, 'vote/votetotals.html')
    else:
        return render(request, 'vote/votetotals.html')







