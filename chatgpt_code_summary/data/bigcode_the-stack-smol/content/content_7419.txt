from features.models import Attempt, Feature
from jury.models import JudgeRequestAssigment, JudgeRequest
from teams.models import Team

for team in Team.objects.all():
    for feature in Feature.objects.all():
        request = JudgeRequest.objects.filter(team=team, feature=feature).last()
        total, num = 0, 0
        if request is not None:
            for assignee in request.assignees.all():
                if assignee.score is not None:
                    total += assignee.score
                    num += 1
        if num > 0:
            attempt = Attempt.objects.get(team=team, feature=feature)
            if total/num != attempt.score:
                print("Inconsistent score for {}, feature {}: total_score: {}, "
                      "number of judges: {}, "
                      "attempt score: {}".format(team, feature, total, num, attempt.score))
        else:
            if Attempt.objects.filter(team=team, feature=feature).exists():
                print("Out of date attempt for {}, feature {}".format(team, feature))
