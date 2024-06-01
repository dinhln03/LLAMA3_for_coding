from __future__ import unicode_literals
import json
import collections
import string
from django.http import JsonResponse, HttpResponseRedirect, HttpResponseNotFound
from django.http import Http404
from django.shortcuts import render
import threading
import json
from django.views.decorators.csrf import csrf_exempt
from .forms import NameForm, RegionForm, PlaceForm, GetTweetsForm
from .app_logic import handle_region_form, handle_place_form, get_user, \
    init_tweet_accumulation_tweet_list, handle_search_form,\
    generate_tweet_sendaway, generate_user_sendaway, word_trends_merge_jsons, replace_string_character, contains_whitespace,\
    filter_strings, phrase_list_to_word_list, get_all_twitter_users_ids, slider_val_transform, convert_to_iso, \
    get_tweet_list, user_ext_to_json, single_word_obj, generate_days_list, parse_parameters, generate_users_tweets
import jsonpickle
from .Analytics import QueriesManager
from .classes import get_from_db, UserExtension, twitter_users_database_name, TweetExtension


def index(request):
    if request.method == 'POST':
        form_user = NameForm(request.POST)
        if form_user.is_valid():
            name = form_user.cleaned_data['your_name']
            name = name.replace(" ", "_")
            return HttpResponseRedirect('/tweets/' + name)
    # if a GET (or any other method) we'll create a blank form
    else:
        form_user = NameForm()

    return render(request, 'index.html', {'form': form_user})


def dashboard(request, name):
    user = get_user(name)

    if request.method == 'POST':
        form_region = RegionForm(request.POST or None)
        regions = handle_region_form(form_region, user)
        place_form = PlaceForm(request.POST or None)
        print(place_form)
        handle_place_form(place_form, user)
        search_form = GetTweetsForm(request.POST or None)
        region, place = handle_search_form(search_form)
        if region != "":
            user.remove_location(region, place)
        user.save_me_to_db()
        user = get_user(name)
        regions = collections.OrderedDict(user.get_regions())
        return render(request, 'dashboard.html',
                      {'name': name, 'regions': regions, 'region_form': RegionForm(), 'place_form': PlaceForm(), 'tweets_form': GetTweetsForm()})

    else:
        user = get_user(name)
        regions = collections.OrderedDict(user.get_regions())
        return render(request, 'dashboard.html',
                      {'name': name, 'regions': regions, 'region_form': RegionForm(), 'place_form': PlaceForm(),
                       'tweets_form': GetTweetsForm()})


def help_page(request, name):
    return render(request, 'help.html', {'name': name})


@csrf_exempt
def get_regions_places_list(request):
    if request.method == 'POST':
        user_name = request.POST.get('user_name', None)
        user = get_user(user_name)
        region_place_dict = user.get_region_place_dict()
        return JsonResponse(region_place_dict, safe=False)
    else:
        empty = {}
        return JsonResponse(empty)

@csrf_exempt
def get_search_words(request):
    if request.method == 'POST':
        user_name = request.POST.get('user_name', None)
        user = get_user(user_name)
        word_to_add = request.POST.get('to_add', None)
        if word_to_add != "":
            word_to_add = word_to_add.replace('"', "")
            user.add_search_word(word_to_add)
        words_to_remove = jsonpickle.decode(request.POST.get('to_remove', None))
        print(words_to_remove)
        if words_to_remove != "":
            words_to_remove = replace_string_character(words_to_remove)
            print(words_to_remove)
            user.remove_search_word(words_to_remove)
        print(user_name)
        print(word_to_add)
        print(user.all_search_words())
        return JsonResponse(user.all_search_words(), safe=False)
    else:
        empty = {}
        return JsonResponse(empty)


@csrf_exempt
def accumulate_tweets(request):
    if request.method == 'POST':
        name, locations, start_date, end_date, word_list, logic, exact = parse_parameters(request)
        user = get_user(name)
        for loc in locations:
            init_tweet_accumulation_tweet_list(user, loc['region'], loc['place'], word_list)
    empty = {}
    return JsonResponse(empty)


@csrf_exempt
def get_query_links(request):
    if request.method == 'POST':
        name = request.POST.get('user_name', None)
        locations = jsonpickle.decode(request.POST.get('locations_list', None))
        print(locations)
        user = get_user(name)
        results = []
        for loc in locations:
            results.append(user.get_region(loc['region']).get_place_by_name(loc['place']).get_query_string())
        print(results)
        return JsonResponse(results, safe=False)
    else:
        empty = {}
        return JsonResponse(empty)


def show_tweets_list(request, name):
    user = get_user(name)
    print("in show_tweets_list")
    if request.method == 'POST':
        search_form = GetTweetsForm(request.POST)
        region, place = handle_search_form(search_form)
        #quary = init_tweet_accumulation_tweet_list(user, region, place)
        quary = "I am lish lash"
        region = user.get_region(region)
        place = region.get_place_by_name(place)
        return render(request, 'tweets.html', { 'quary': quary, 'region': region, 'place': place, 'user': name})
    elif request.method == 'GET':
        quary = "I am lish lash"
        region = "Lash"
        place = "Lish"
        return render(request, 'tweets.html', { 'quary': quary, 'region': region, 'place': place, 'user': name})


@csrf_exempt
def popular_users_get(request):
    print("popular_users_get")
    users_list = []
    if request.method == 'POST':
        twitter_users, tweets = generate_users_tweets(request, tasdocs=True, uasdocs=True)
        if isinstance(twitter_users, Exception):
            return JsonResponse(str(twitter_users), safe=False, status=500)
        elif len(twitter_users) == 0:
            return JsonResponse("Error: No Tweets in location / date / search phrase (if included)", safe=False, status=500)
        slider_valus = slider_val_transform(jsonpickle.decode(request.POST.get('sliders_data', None)))
        print(len(twitter_users))
        print(len(tweets))
        # ["followers_slider", "retweet_slider", "favorites_slider", "tweets_slider"]
        # ["followers, statusses, favorites (likes), retweets]
        queriesManager = QueriesManager()
        params = ['Opinion_Leaders', [str(slider_valus[0][1])], [str(slider_valus[0][0])],
                  [str(slider_valus[3][1])], [str(slider_valus[3][0])], [str(slider_valus[2][1])],
                  [str(slider_valus[2][0])],
                  [str(slider_valus[1][1])], [str(slider_valus[1][0])]]
        print(params)
        print(twitter_users[0].keys())
        df = queriesManager.call_querie(params, tweets, twitter_users)
        print(df)
        if df.empty:
            return JsonResponse("Error: No opinion leaders found", safe=False)
        idlist = df['id'].tolist()
        print(len(idlist))
        users_list = get_from_db(idlist, twitter_users_database_name, UserExtension)
        print(len(users_list))
    user_ext_list = user_ext_to_json(users_list)
    return JsonResponse(user_ext_list, safe=False)


@csrf_exempt
def tweet_list_place(request):
    print("tweet_list_place")
    if request.method == 'POST':
        name, locations, start_date, end_date, word_list, logic, exact = parse_parameters(request)
        print("word list is: " + str(word_list))
        print("logic is: " + logic)
        if start_date is not "" and end_date is not "":
            days_list = generate_days_list(start_date, end_date)
        else:
            days_list = None
        print(days_list)
        total_tweets = []
        for loc in locations:
            place = get_user(name).get_region(loc['region']).get_place_by_name(loc['place'])
            mylist = place.get_tweets_directly(name, loc['region'], days_list, word_list, logic=logic, exact=exact)
            if isinstance(mylist, Exception):
                return JsonResponse(str(mylist), safe=False, status=500)
            #print(mylist)
            json_list = []
            for l in mylist:
                result = generate_tweet_sendaway(l.tweet)
                """ this is the paid data from ibm watson
                result.append(l.category)
                result.append(l.concept)
                result.append(l.entities)
                result.append(l.entities_sentiment)
                result.append(l.keywords)
                result.append(l.keywords_sentiment)
                """
                json_list.append(result)
            total_tweets = total_tweets + json_list
        #print(total_tweets)
        if len(total_tweets) == 0:
            return JsonResponse("Error: No Tweets in location / date / search phrase (if included)", safe=False, status=500)
        return JsonResponse(total_tweets, safe=False)
    else:
        empty = {}
        return JsonResponse(empty)


@csrf_exempt
def show_users_place(request):
    print("show_users_place")
    if request.method == 'POST':
        twitter_users, _ = generate_users_tweets(request)
        if isinstance(twitter_users, Exception):
            return JsonResponse(str(twitter_users), safe=False, status=500)
        elif len(twitter_users) == 0:
            return JsonResponse("Error: No Tweets in location and date", safe=False, status=500)
        return JsonResponse(user_ext_to_json(twitter_users), safe=False)
    else:
        empty = {}
        return JsonResponse(empty)


@csrf_exempt
def word_trends_get(request):
    print("word_trends_get")
    total_result = []
    if request.method == 'POST':
        queriesManager = QueriesManager()
        name, locations, start_date, end_date, word_list, logic, exact = parse_parameters(request)
        days_list = generate_days_list(start_date, end_date)
        total_tweets = get_tweet_list(locations, get_user(name), days_list, word_list=word_list, asdocs=True, exact=exact)
        if isinstance(total_tweets, Exception):
            return JsonResponse(str(total_tweets), safe=False, status=500)
        elif len(total_tweets) == 0:
            return JsonResponse("Error: No Tweets in location and date", safe=False, status=500)
        word_list, pharses_list = filter_strings(word_list)
        if not exact:
            for word in word_list:
                pharses_list.append(word)
            word_list = []
        params = ["Word_trend", word_list]
        print(params)
        print(len(total_tweets))
        df = queriesManager.call_querie(params, total_tweets, [])
        phrase_dfs = []
        for phrase in pharses_list:
            params = ['Phrase_trend', phrase]
            print(params)
            phrase_dfs.append({"df": queriesManager.call_querie(params, total_tweets, []), "phrase": phrase})
        print(phrase_dfs)
        for word in word_list:
            total_result.append(single_word_obj(word, 'word', df, days_list))
        for dic in phrase_dfs:
            total_result.append(single_word_obj(dic['phrase'], 'phrase', dic["df"], days_list))
        print(total_result)
    return JsonResponse(total_result, safe=False)


@csrf_exempt
def top_words_per_date_get(request):
    print("top_words_per_date_get")
    words_list = []
    counter_list = []
    days_list = []
    if request.method == 'POST':
        queriesManager = QueriesManager()
        name, locations, start_date, end_date, words_list, logic, exact = parse_parameters(request)
        days_list = generate_days_list(start_date, end_date)
        dates_counter = dict.fromkeys(days_list)
        print(days_list)
        total_tweets = get_tweet_list(locations, get_user(name), days_list, None, asdocs=True)
        if isinstance(total_tweets, Exception):
            return JsonResponse(str(total_tweets), safe=False, status=500)
        elif len(total_tweets) == 0:
            return JsonResponse("Error: No Tweets in location and date", safe=False, status=500)
        for k, _ in dates_counter.items():
            dates_counter[k] = {'word': "", 'count': 0}
        params = ["Popular_word_per_date"]
        print(params)
        print(len(total_tweets))
        df = queriesManager.call_querie(params, total_tweets, [])
        print(df)
        words_list = ["No Tweets"] * len(days_list)
        counter_list = [0] * len(days_list)
        for i, day in enumerate(days_list):
            print(day)
            col = df.loc[df['date'] == day]
            for index, row in col.iterrows():
                words_list[i] = row['popular_word']
                counter_list[i] = row['counter']
        print(days_list)
        print(words_list)
        print(counter_list)
    return JsonResponse({'dates': days_list, 'words': words_list, 'counter': counter_list}, safe=False)


@csrf_exempt
def popularity_of_words_get(request):
    print("popularity_of_words_get")
    df = ""
    if request.method == 'POST':
        queriesManager = QueriesManager()
        name, locations, start_date, end_date, word_list, logic, exact = parse_parameters(request)
        word_list, pharase_list = filter_strings(word_list)
        word_list = word_list + phrase_list_to_word_list(pharase_list)
        days_list = generate_days_list(start_date, end_date)
        total_tweets = get_tweet_list(locations, get_user(name), days_list, word_list=word_list, asdocs=True, exact=exact)
        if isinstance(total_tweets, Exception):
            return JsonResponse(str(total_tweets), safe=False, status=500)
        elif len(total_tweets) == 0:
            return JsonResponse("Error: No Tweets in location and date", safe=False, status=500)
        params = ["Popularity_of_word_bank_per_place", word_list]
        print(params)
        print(len(total_tweets))
        df = queriesManager.call_querie(params, total_tweets, [])
        if df.empty:
            return JsonResponse("Error: No appearances of search phrases in tweets", safe=False)
        rows = df.shape[0]
        df = df.to_json()
        df = df[:-1]
        df = df + ', "rows": ' + str(rows)
        df = df + ', "word_list": ' + str(word_list) + '}'
        print(df)
        df = df.replace("'", '"')
        print(df)
    return JsonResponse(df, safe=False)


@csrf_exempt
def most_popular_word_get(request):
    print("most_popular_word_get")
    place_list = []
    word_list = []
    counter_list = []
    if request.method == 'POST':
        queriesManager = QueriesManager()
        name, locations, start_date, end_date, word_list, logic, exact = parse_parameters(request)
        days_list = generate_days_list(start_date, end_date)
        total_tweets = get_tweet_list(locations, get_user(name), days_list, None, asdocs=True)
        if isinstance(total_tweets, Exception):
            return JsonResponse(str(total_tweets), safe=False, status=500)
        elif len(total_tweets) == 0:
            return JsonResponse("Error: No Tweets in location and date", safe=False, status=500)
        params = ["Popular_word_per_place"]
        df = queriesManager.call_querie(params, total_tweets, [])
        print(df)
        for i, row in df.iterrows():
            place_list.append(row['place_name'])
            word_list.append(row['popular_word'])
            counter_list.append(row['counter'])
    return JsonResponse({'places': place_list, 'words': word_list, 'counters': counter_list}, safe=False)


@csrf_exempt
def first_time_get(request):
    print("first_time_get")
    df_list = []
    if request.method == 'POST':
        queriesManager = QueriesManager()
        name, locations, start_date, end_date, word_list, logic, exact = parse_parameters(request)
        max_results = request.POST.get('max_results', None)
        total_users, total_tweets = generate_users_tweets(request, use_words=True, tasdocs=True, uasdocs=True)
        if isinstance(total_users, Exception):
            return JsonResponse(str(total_users), safe=False, status=500)
        elif len(total_tweets) == 0:
            return JsonResponse("Error: No Tweets in location and date", safe=False, status=500)
        print(len(total_tweets))
        print(len(total_users))
        for word in word_list:
            if exact:
                params = ["First_Time", [" " + word + " "], [max_results]]
            else:
                params = ["First_Time", [word], [max_results]]
            print(params)
            df = queriesManager.call_querie(params, total_tweets, total_users)
            print(df)
            if df.empty:
                continue
            row_data = []
            # [id, text, user_id, screen_name, full_date, time_rnk]
            for index, row in df.iterrows():
               row_data.append([row["id"], row["text"], row["screen_name"], convert_to_iso(row["full_date"]),
                                row["time_rnk"]])
            df_list.append({"word": word, "len": df.shape[0], "row_data": row_data})
        if len(df_list) == 0:
            return JsonResponse("Error: No Tweets in location and date contain search phrase", safe=False, status=500)
    return JsonResponse(df_list, safe=False)


@csrf_exempt
def most_retweeted_get(request):
    print("most_retweeted_get")
    row_data = []
    if request.method == 'POST':
        queriesManager = QueriesManager()
        name, locations, start_date, end_date, word_list, logic, exact = parse_parameters(request)
        max_results = request.POST.get('max_results', None)
        total_users, total_tweets = generate_users_tweets(request, use_words=True, tasdocs=True, uasdocs=True)
        if isinstance(total_users, Exception):
            return JsonResponse(str(total_users), safe=False, status=500)
        elif len(total_tweets) == 0:
            return JsonResponse("Error: No Tweets in location and date", safe=False, status=500)
        print(len(total_tweets))
        print(len(total_users))
        for word in word_list:
            if exact:
                params = ["Most_Retweeted", [" " + word + " "], [max_results]]
            else:
                params = ["Most_Retweeted", [word], [max_results]]
            print(params)
            df = queriesManager.call_querie(params, total_tweets, total_users)
            print(df)
            if df.empty:
                continue
            # [phrase, id, text, user_id, screen_name, full_date, retweet_count, retweet_rnk
            for index, row in df.iterrows():
               row_data.append([row["id"], row["text"], row["screen_name"], convert_to_iso(row["full_date"]),
                                row["retweet_count"], row["retweet_rnk"], row["phrase"]])
    print(row_data)
    if len(row_data) == 0:
        return JsonResponse("Error: No Tweets in location and date contain search phrase", safe=False, status=500)
    return JsonResponse(row_data, safe=False)


def health(request):
    state = {"status": "UP"}
    return JsonResponse(state)


def handler404(request):
    return render(request, '404.html', status=404)


def handler500(request):
    return render(request, '500.html', status=500)
