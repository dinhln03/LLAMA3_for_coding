################################################################################
#
# Author:           Diego Montufar
# Date:             Apr/2015
# Name:             __init__.py
# Description:      Here we define the available web services which can be accessed by
#				    following the app.route path as URL. The indexer module will perform the hard work
#                   as it communicates directly with the elasticsearch and couchdb instances through RESTful calls.
#                   Although Flask apps allow us to perform more complex tasks, we only are using basic calls
#                   to the corresponding inexer module method and return to the wenb interface a json response.
# 
# Dependencies:     Flask     -> Provides you with tools, libraries and technologies that allow you to build a web application
#                   indexer   -> For communicating with elasticsearch and couchdb
#
################################################################################

from services import indexer #indexer module
from datetime import timedelta #datetime tools
from flask import make_response, request, current_app, jsonify, Flask , render_template #flask crossdomain tools
from functools import update_wrapper #python wrappers

app = Flask(__name__)

#Python version workaround
try:
	unicode = unicode
except NameError:
	# 'unicode' is undefined, must be Python 3
	str = str
	unicode = str
	bytes = bytes
	basestring = (str,bytes)
else:
	# 'unicode' exists, must be Python 2
	str = str
	unicode = unicode
	bytes = str
	basestring = basestring

#Define a wrapper for supporting crossdomain calls
def crossdomain(origin=None, methods=None, headers=None, max_age=21600, attach_to_all=True, automatic_options=True):
	
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers
            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            h['Access-Control-Allow-Credentials'] = 'true'
            h['Access-Control-Allow-Headers'] = \
                "Origin, X-Requested-With, Content-Type, Accept, Authorization"
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator

#Service:       /
#Description:   Main web page index.html is called here. i.e http://{localhost}/ or http://{localhost}/index
#Parameters: 	none
#output: 	    index.html
@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

#Service: 		genericSearch
#Description:	Perform custom searches based on elasticsearch queries in json format
#Parameters: 	<jsonQuery> (String) is a json based tring, which must follow the elasticsearch query structure
#output: 		a json object containing the matched results
@app.route('/genericSearch/<jsonQuery>',methods=['GET', 'OPTIONS'])
@crossdomain(origin='*')
def getGenericSearch(jsonQuery):
	return jsonify(indexer.genericSearch(jsonQuery))

#Service: 		genericGeoSearch
#Description:   Perform custom geo searches based on elasticsearch queries in json format
#Parameters: 	<term>        (String)  -> Text you want to search for. i.e. AFL or Tony Abbott or *
#               <suburbCode>  (String)  -> Suburb code. i.e. 206041122
#               <startP>      (Int)     -> Pagination support. i.e. 0 if you want all the results, or 15 if you want to skip the first 15 results
#               <sizeP>       (Int)     -> Pagination support. i.e 50 if you want up to 50 resutls by taking account of startP parameter
#               <startDate>   (Int)     -> Timestamp start date. i.e 1428069500339
#               <endDate>     (Int)     -> Timestamp end date i.e 1430578700339
#output: 		a json object containing the matched results
@app.route('/genericGeoSearch/<term>/<suburbCode>/<startP>/<sizeP>/<startDate>/<endDate>',methods=['GET', 'OPTIONS'])
@crossdomain(origin='*')
def getGenericGeoSearch(term,suburbCode,startP,sizeP,startDate,endDate):
	return jsonify(indexer.getTweetsBySuburb(term,suburbCode,startP,sizeP,startDate,endDate))

#Service:       sentimentTotals
#Description:   Search for sentiment totals by terms, suburb and date range
#Parameters:    <term>        (String)  -> Text you want to search for. i.e. AFL or Tony Abbott or *
#               <suburbCode>  (String)  -> Suburb code. i.e. 206041122
#               <startDate>   (Int)     -> Timestamp start date. i.e 1428069500339
#               <endDate>     (Int)     -> Timestamp end date i.e 1430578700339
#output:        a json object containing the matched results
@app.route('/sentimentTotals/<term>/<suburbCode>/<startDate>/<endDate>',methods=['GET', 'OPTIONS'])
@crossdomain(origin='*')
def getSentimentTotals(term,suburbCode,startDate,endDate):
	return jsonify(indexer.statisticsByTerm(term,suburbCode,startDate,endDate))

#Service:		suburbsByCountry
#Description: 	Get a list of the suburbs of main cities of AU as defined on the ABS 2011 census database
#Parameters: 	<countryCode> (Int)   -> Country code. In this case we only care about Autralia, that is countryCode = 1
#output: 		 a json object containing the matched results
@app.route('/suburbsByCountry/<countryCode>',methods=['GET', 'OPTIONS'])
@crossdomain(origin='*')
def getSuburbsList(countryCode):
	return jsonify(indexer.getSuburbsList(countryCode))


#Service:		culturesByState
#Description: 	Get GeoJson File containing information relatd to countries of birth by suburb of AU as defined on the ABS 2011 census database
#Parameters: 	<stateCode> (String) -> the code of the state: i.e: VIC, TAS, etc
#return: 		GeoJson object
@app.route('/culturesByState/<stateCode>',methods=['GET', 'OPTIONS']) 
@crossdomain(origin='*')
def getCulturesByState(stateCode):
	return jsonify(indexer.getCulturesByState(stateCode))

#Service:       languagesByCountry
#Description:   Get a list of languages related to its corresponding country where they are spoken
#Parameters:    <countryCode> (String) -> Country code. In this case we only care about Autralia, that is countryCode = 1
#return:         a json object containing the matched results
@app.route('/languagesByCountry/<countryCode>',methods=['GET', 'OPTIONS'])
@crossdomain(origin='*')
def getLanguagesByCountry(countryCode):
	return jsonify(indexer.getLanguages(countryCode))

#Service:       tweetsByCountryOfBirth
#Description:   Get the count of tweets grouped by language as defined on the languages database
#Parameters:    <term>        (String)  -> Text you want to search for. i.e. AFL or Tony Abbott or *
#               <stateCode>   (String)  -> the code of the state: i.e: VIC, TAS, etc
#               <suburbCode>  (String)  -> Suburb code. i.e. 206041122
#               <startDate>   (Int)     -> Timestamp start date. i.e 1428069500339
#               <endDate>     (Int)     -> Timestamp end date i.e 1430578700339
#output:        a json object containing the matched results
@app.route('/tweetsByCountryOfBirth/<term>/<stateCode>/<suburbCode>/<startDate>/<endDate>',methods=['GET', 'OPTIONS'])
@crossdomain(origin='*')
def getTweetsByCountryOfBirth(term,stateCode,suburbCode,startDate,endDate):
	return jsonify(indexer.getTweetsByCountryOfBirth(term,stateCode,suburbCode,startDate,endDate))


#Service:       topListBySuburb
#Description:   List the top N list count of a particular field within a date range by suburb
#Parameters:    <term>        (String)  -> Text you want to search for. i.e. AFL or Tony Abbott or *
#               <suburbCode>  (String)  -> Suburb code. i.e. 206041122
#               <field>       (String)  -> Twitter field to aggreagate i.e. user.screen_name
#               <size>        (Int)     -> N i.e 5 for a Top five list 
#               <startDate>   (Int)     -> Timestamp start date. i.e 1428069500339
#               <endDate>     (Int)     -> Timestamp end date i.e 1430578700339
#output:        a json object containing the matched results
@app.route('/topListBySuburb/<term>/<suburbCode>/<field>/<size>/<startDate>/<endDate>',methods=['GET', 'OPTIONS'])
@crossdomain(origin='*')
def getTopList(term,suburbCode,field,size,startDate,endDate):
	return jsonify(indexer.getTopListBySuburb(term,suburbCode,field,size,startDate,endDate))

#Service:       sentimentTotalsByCity
#Description:   Search for the totals of sentiment analysis in all the cities of AU within a date range
#Parameters:    <term>        (String)  -> Text you want to search for. i.e. AFL or Tony Abbott or *
#               <startDate>   (Int)     -> Timestamp start date. i.e 1428069500339
#               <endDate>     (Int)     -> Timestamp end date i.e 1430578700339
#output:        a json object containing the matched results
@app.route('/sentimentTotalsByCity/<term>/<startDate>/<endDate>',methods=['GET', 'OPTIONS'])
@crossdomain(origin='*')
def getSentimentTotalsByCity(term,startDate,endDate):
	return jsonify(indexer.getAllSentimentTotalsByCity(term,startDate,endDate))


#Service:       topListByCity
#Description:   Get the top N list by city within a date range
#Parameters:    <field>       (String)  -> Twitter field to aggreagate i.e. user.screen_name
#               <size>        (Int)     -> N i.e 5 for a Top five list 
#               <startDate>   (Int)     -> Timestamp start date. i.e 1428069500339
#               <endDate>     (Int)     -> Timestamp end date i.e 1430578700339
#output:        a json object containing the matched results
@app.route('/topListByCity/<field>/<size>/<startDate>/<endDate>',methods=['GET','OPTIONS'])
@crossdomain(origin='*')
def getAllTopListByCity(field,size,startDate, endDate):
	return jsonify(indexer.getAllTopListsByCity(field,size,startDate, endDate))


#Service:       cultureTotalsByCity
#Description:   Get the count of languages found on the tweets by terms within a date range
#Parameters:    <term>        (String)  -> Text you want to search for. i.e. AFL or Tony Abbott or *
#               <stateCode>   (String)  -> the code of the state: i.e: VIC, TAS, etc
#               <startDate>   (Int)     -> Timestamp start date. i.e 1428069500339
#               <endDate>     (Int)     -> Timestamp end date i.e 1430578700339
#output:        a json object containing the matched results
@app.route('/cultureTotalsByCity/<term>/<stateCode>/<startDate>/<endDate>',methods=['GET','OPTIONS'])
@crossdomain(origin='*')
def getAllLanguagesTotalsByCity(term, stateCode, startDate, endDate):
	return jsonify(indexer.getAllLanguagesTotalsByCity(term, stateCode, startDate, endDate))


#Service:       sentimentTotalsByCity
#Description:   Get the total sentiment by City, by term within a date range.
#Disclaimer:    This search takes a long time. Must be reviewed.
#Parameters:    <term>        (String)  -> Text you want to search for. i.e. AFL or Tony Abbott or *
#               <stateCode>   (String)  -> the code of the state: i.e: VIC, TAS, etc
#               <startDate>   (Int)     -> Timestamp start date. i.e 1428069500339
#               <endDate>     (Int)     -> Timestamp end date i.e 1430578700339
#output:        a json object containing the matched results
@app.route('/sentimentTotalsByCity/<term>/<stateCode>/<startDate>/<endDate>',methods=['GET','OPTIONS'])
@crossdomain(origin='*')
def getAllSentimentByCity(term,stateCode, startDate, endDate):
	return jsonify(indexer.getAllSentimentByCity(term,stateCode, startDate, endDate))


#Service:       sentimentAnalysis
#Description:   Perform sentiment Analysis using the tweet_classifier open source library
#Parameters:    <text> (String) -> i.e. I'm happy to be here :)
#output:        json results containing the sentiment analysis performed on the provided text
@app.route('/sentimentAnalysis/<text>',methods=['GET','OPTIONS'])
@crossdomain(origin='*')
def getSentimentAnalysis(text):
	return jsonify(indexer.getSentimentAnalysis(text))


#Service:       cultureSentimentBySuburb
#Description:   Get the count of sentiment of languages found on the tweets by terms within a date range
#Parameters:    <term>        (String)  -> Text you want to search for. i.e. AFL or Tony Abbott or *
#               <suburbCode>  (String)  -> Suburb code. i.e. 206041122
#               <startDate>   (Int)     -> Timestamp start date. i.e 1428069500339
#               <endDate>     (Int)     -> Timestamp end date i.e 1430578700339
#output:        a json object containing the matched results
@app.route('/cultureSentimentBySuburb/<term>/<suburbCode>/<startDate>/<endDate>',methods=['GET','OPTIONS'])
@crossdomain(origin='*')
def getLanguagesSentimentBySuburb(term, suburbCode, startDate, endDate):
    return jsonify(indexer.getLanguagesSentimentBySuburb(term, suburbCode, startDate, endDate))

#main
if __name__ == "__main__":
	app.run(debug=True)