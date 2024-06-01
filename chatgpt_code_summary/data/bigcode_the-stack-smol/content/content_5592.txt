# coding=utf-8
import json
import urllib
import urllib2

from bs4 import BeautifulSoup


def run(bot, chat_id, user, keyConfig='', message='', totalResults=1):
    requestText = str(message)
    if requestText == '':

        totalSteamGames = int(Get_steam_total())
        totalGOGGames = int(Get_GOG_total())
        if totalSteamGames is not None and totalGOGGames is not None:
            bot.sendMessage(chat_id=chat_id, text='I\'m sorry ' + (user if not user == '' else 'Dave') + \
                                                  ', there are ' + str(int(totalSteamGames) + int(totalGOGGames)) +
                                                  ' total games on Steam and GOG combined. Pick one.')
            return True

    retryCount = 3
    appId = ''
    while retryCount > 0 and appId == '':
        retryCount -= 1
        rawSteamSearchResultsMarkup = urllib.urlopen('http://store.steampowered.com/search/?category1=998&term=' + requestText).read()
        appId = steam_results_parser(rawSteamSearchResultsMarkup)

    if appId:
        steamGameLink = 'http://store.steampowered.com/app/' + appId
        bypassAgeGate = urllib2.build_opener()
        #this bypasses the "mature content - continue/cancel" screen
        bypassAgeGate.addheaders.append(('Cookie', 'mature_content=1; path=/; max-age=31536000;expires=Fri, 26 Mar 2027 20:00:00 GMT'))
        bypassAgeGate.open(steamGameLink)
        #this bypasses the "enter your date of birth" screen
        bypassAgeGate.addheaders.append(('Cookie', 'birthtime=0; path=/; max-age=31536000;expires=Fri, 26 Mar 2027 20:00:00 GMT'))
        code = bypassAgeGate.open(steamGameLink).read()
        if 'id=\"agegate_wizard\"' in code:
            gameTitle = steam_age_gate_parser(code)
            bot.sendMessage(chat_id=chat_id, text='I\'m sorry ' + (user if not user == '' else 'Dave') + \
                                              ', I\'m afraid that \"' + gameTitle + '\" is protected by an age gate.\n' +
                            steamGameLink)
            return False

        gameResults = steam_game_parser(code, steamGameLink)
        try:
            bot.sendMessage(chat_id=chat_id, text=gameResults,
                            disable_web_page_preview=True, parse_mode='Markdown')
        except:
            bot.sendMessage(chat_id=chat_id, text=gameResults)
        return True
    else:
        gogSearchData = json.load(urllib.urlopen('http://embed.gog.com/games/ajax/filtered?mediaType=game&search=' + requestText))
        appId, price, discount = gog_results_parser(gogSearchData)
        if appId:
            gogGameLink = 'http://api.gog.com/products/' + str(appId) + '?expand=downloads,expanded_dlcs,description,screenshots,videos,related_products,changelog'
            data = json.load(urllib.urlopen(gogGameLink))
            gameResults = gog_game_parser(data, price, discount)
            bot.sendMessage(chat_id=chat_id, text=gameResults,
                            disable_web_page_preview=True, parse_mode='Markdown')
        else:
            bot.sendMessage(chat_id=chat_id, text='I\'m sorry ' + (user if not user == '' else 'Dave') + \
                                                  ', I\'m afraid I can\'t find the game ' + \
                                                  requestText.encode('utf-8'))


def steam_results_parser(rawMarkup):
    soup = BeautifulSoup(rawMarkup, 'html.parser')
    resultList = []
    for resultRow in soup.findAll('a', attrs={'class':'search_result_row'}):
        if 'data-ds-appid' in resultRow.attrs:
            resultList.append(resultRow['data-ds-appid'])
        if 'data-ds-bundleid' in resultRow.attrs:
            resultList.append(resultRow['data-ds-bundleid'])
    if len(resultList) > 0:
        return resultList[0]
    return ''

def Get_steam_total():
    rawMarkup = urllib.urlopen('http://store.steampowered.com/search/?category1=998&term=#').read()
    soup = BeautifulSoup(rawMarkup, 'html.parser')
    findPaginationString = soup.find('div', attrs={'class': 'search_pagination_left'})
    if findPaginationString:
        rawPaginationString = findPaginationString.string
        return rawPaginationString.replace('showing 1 - 25 of', '').strip()
    return 'uncountable'

def Get_GOG_total():
    GogSearchResultsData = json.load(urllib.urlopen('http://embed.gog.com/games/ajax/filtered?mediaType=game&sort=bestselling'))
    if 'totalGamesFound' in GogSearchResultsData:
        return GogSearchResultsData['totalGamesFound']
    return 'uncountable'

def steam_age_gate_parser(rawMarkup):
    soup = BeautifulSoup(rawMarkup, 'html.parser')
    rawTitleString = soup.find('title').string
    return rawTitleString.strip()

def steam_game_parser(code, link):
    soup = BeautifulSoup(code, 'html.parser')
    AllGameDetailsFormatted = ''

    titleDiv = soup.find('div', attrs={'class':'apphub_AppName'})
    if titleDiv:
        gameTitle = titleDiv.string
        AllGameDetailsFormatted += '*' + gameTitle

        priceDiv = soup.find('div', attrs={'class':'game_purchase_price price'})
        if priceDiv:
            gamePrice = priceDiv.string
            AllGameDetailsFormatted += ' - ' + gamePrice.strip()
        else:
            priceDiv = soup.find('div', attrs={'class':'discount_final_price'})
            if priceDiv:
                gamePrice = priceDiv.string
                AllGameDetailsFormatted += ' - ' + gamePrice.strip()
                discountPercentageDiv = soup.find('div', attrs={'class':'discount_pct'})
                if discountPercentageDiv:
                    percentageDiscountedBy = discountPercentageDiv.string
                    AllGameDetailsFormatted += ' (at ' + percentageDiscountedBy.strip() + ' off)'
        AllGameDetailsFormatted += '*\n'
    else:
        print('Cannot parse title as div with class apphub_AppName from Steam page for ' + link)

    descriptionDiv = soup.find('div', attrs={'class':'game_description_snippet'})
    if descriptionDiv:
        descriptionSnippet = descriptionDiv.string.replace('\r', '').replace('\n', '').replace('\t', '').replace('_', ' ')
        AllGameDetailsFormatted += descriptionSnippet + '\n'

    if AllGameDetailsFormatted:
        AllGameDetailsFormatted += link + '\n'

    dateSpan = soup.find('div', attrs={'class':'date'})
    if dateSpan:
        releaseDate = dateSpan.string
        AllGameDetailsFormatted += 'Release Date: ' + releaseDate + '\n'

    featureList = ''
    featureLinks = soup.findAll('a', attrs={'class':'name'})
    if len(featureLinks) > 0:
        for featureLink in featureLinks:
            if featureLink.string != None:
                featureList += '     ' + featureLink.string.replace('Seated', 'Seated VR') + '\n'
        AllGameDetailsFormatted += 'Features:\n' + featureList

    reviewRows = ''
    reviewDivs = soup.findAll('div', attrs={'class':'user_reviews_summary_row'})
    if len(reviewDivs) > 0:
        for reviewRow in reviewDivs:
            reviewSubtitleRawDiv = reviewRow.find('div', attrs={'class':'subtitle column'})
            reviewSubtitleDiv = ''
            if reviewSubtitleRawDiv is not None:
                reviewSubtitleDiv = reviewSubtitleRawDiv.string
            reviewSummaryRawDiv = reviewRow.find('div', attrs={'class':'summary column'})
            reviewSummaryDiv = ''
            if reviewSummaryRawDiv is not None:
                reviewSummaryDiv = reviewSummaryRawDiv.string
            if not reviewSummaryDiv:
                findReviewSummaryDiv = reviewRow.find('span', attrs={'class':'game_review_summary'})
                if findReviewSummaryDiv:
                    reviewSummaryDiv = findReviewSummaryDiv.string
            reviewSummaryDiv = reviewSummaryDiv.replace('\r', '').replace('\n', '').replace('\t', '')
            if reviewSummaryDiv != 'No user reviews':
                reviewRows += '     ' + reviewSubtitleDiv + \
                              reviewSummaryDiv.replace('-', '')\
                                  .replace(' user reviews', '')\
                                  .replace(' of the ', ' of ') + '\n'
        if reviewRows:
            AllGameDetailsFormatted += 'Reviews:\n' + reviewRows.replace('Recent Reviews:', '')
        if AllGameDetailsFormatted.endswith('\n'):
            AllGameDetailsFormatted = AllGameDetailsFormatted[:AllGameDetailsFormatted.rfind('\n')]

    tagList = ''
    tagLinks = soup.findAll('a', attrs={'class':'app_tag'})
    if len(tagLinks) > 0:
        for tagLink in tagLinks:
            tagList += tagLink.string.replace('\r', '').replace('\n', '').replace('\t', '') + ', '
        AllGameDetailsFormatted += '\n' + 'Tags:\n`' + tagList
    if AllGameDetailsFormatted.endswith(', '):
        AllGameDetailsFormatted = AllGameDetailsFormatted[:AllGameDetailsFormatted.rfind(', ')]
        AllGameDetailsFormatted += '`'

    return AllGameDetailsFormatted

class NoRedirectHandler(urllib2.HTTPRedirectHandler):
    def http_error_302(self, req, fp, code, msg, headers):
        infourl = urllib.addinfourl(fp, headers, req.get_full_url())
        infourl.status = code
        infourl.code = code
        return infourl
    http_error_300 = http_error_302
    http_error_301 = http_error_302
    http_error_303 = http_error_302
    http_error_307 = http_error_302

def gog_results_parser(searchData):
    if len(searchData['products']) > 0:
        firstResult = searchData['products'][0]
        if 'id' in firstResult:
            return firstResult['id'], firstResult['price']['finalAmount'], firstResult['price']['discountPercentage']
    return '', '', ''

def gog_game_parser(data, price, discount):
    AllGameDetailsFormatted = ''
    if 'title' in data:
        gameTitle = data['title']
        AllGameDetailsFormatted += '*' + gameTitle
    else:
        raise Exception('Cannot parse title from gog api object for this game.')
    if price > 0:
        AllGameDetailsFormatted += ' - ' + str(price) + '$'
    else:
        AllGameDetailsFormatted += ' - free to play'
    if discount > 0:
        AllGameDetailsFormatted += ' (at ' + discount + '% off)'
    AllGameDetailsFormatted += '*\n'

    if 'description' in data and 'full' in data['description']:
        descriptionSnippet = data['description']['full']
        AllGameDetailsFormatted += descriptionSnippet\
                                       .replace('*','')\
                                       .replace('<br>', '')\
                                       .replace('<b>', '*')\
                                       .replace('</b>', '*') + '\n'

    #if 'links' in data and 'product_card' in data['links']:
    #    AllGameDetailsFormatted += data['links']['product_card'] + '\n'

    #if 'release_date' in data:
    #    AllGameDetailsFormatted += 'Release Date: ' + data['release_date'] + '\n'

    return AllGameDetailsFormatted
