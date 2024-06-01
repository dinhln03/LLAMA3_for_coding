#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author: vs@webdirect.md
Description: Very simple reminder

'''
from core.people.person import Profile, Session
from core.utils.utils import text2int
import re
from crontab import CronTab
from getpass import getuser
from core.config.settings import logger, ROBOT_DIR


class Reaction:
    """remind me every ...  reaction"""
    response = ''
    request = ''

    def __str__(self):
        return 'Remind me every ... reaction'

    @classmethod
    def __init__(self, *args, **kwargs):
        """ original request string """
        #get request object
        self.req_obj = kwargs.pop('req_obj')

        #request word sequence
        self.request = self.req_obj.get('request', '')

        #request received from (julius, jabber any other resources)
        self.req_from = self.req_obj.get('from', '')

        self.response = ''

    @classmethod
    def run(self):
        """default method"""

        sess = Session()
        sender = self.req_obj.get('sender', '')

        if sender:
            #exctract sender email
            email = sender.split('/')[0]

        #find user profile by primary email
        profile = sess.query(Profile).filter(Profile.email == email).one()

        cron = CronTab(getuser())

        DAYS = {'sunday': 'SUN'
                , 'monday': 'MON'
                , 'tuesday': 'TUE'
                , 'wednesday': 'WED'
                , 'thursday': 'THU'
                , 'friday': 'FRI'
                , 'saturday': 'SAT'}

        req = self.request.replace('remind me every', '', 1)
        #r = re.compile(re.escape('remind me every'), re.IGNORECASE)
        #req = r.sub('', request)
        m = re.search('\s+?(by|with|to|of)\s+message\s+?(.+)', req)

        if m and m.group(2):
            msg = m.group(2)
        else:
            m = re.search('\s+?(by|with|to|of)\s+?(.+)', req)
            if m and m.group(2):
                msg = m.group(2)
            else:
                msg = 'This a reminder. Unfortunatelly I could not parse your message, \
                        but I guess you can remember what you wanted to do.'

        job = cron.new(command='/usr/bin/python %s/core/cron/cronjob.py --uuid=%s \
                        --cmd="send jabber message" --arguments="%s"' % (ROBOT_DIR, profile.uuid, msg.replace('"', '')))

        skip_other = False

        if req.strip().startswith('month'):
            job.minute.on(0)
            job.hour.on(0)
            job.dom.on(1)
            skip_other = True

        if req.strip().startswith('week'):
            job.minute.on(0)
            job.hour.on(0)
            job.dow.on(0)
            skip_other = True

        if req.strip().startswith('year'):
            job.dom.on(0)
            job.month.on(0)
            skip_other = True

        dow = False
        for dw, cron_day in DAYS.items():
            if req.strip().lower().startswith(dw):
                dow = True
                break

        if dow:
            job.dow.on(cron_day.upper())
            #req = req.replace(dw, '', 1)   - ignore case problem
            r = re.split(r'^' + dw, req.strip(), flags=re.IGNORECASE)
            if r and len(r) == 2:
                req = r.pop()

            if req.strip().startswith('at '):
                ################################################
                #   every monday/tuesday/wednesday at 00:00
                ################################################

                time = re.search("[^0-9](\d{1,2})\so'clock", req)
                if time and time.group(1):
                    job.minute.on(0)
                    job.hour.on(time.group(1))
                    skip_other = True

                if not skip_other:

                ################################################
                #   every monday/tuesday/wednesday at 00:00
                ################################################
                    time = re.search('[^0-9](\d{1,2}):(\d{2})[^0-9]', req)
                    if time and time.group(1) and time.group(2):
                        job.minute.on(time.group(2))
                        job.hour.on(time.group(1))
                        skip_other = True

                ################################################
                #   every monday/tuesday/wednesday hourly
                ################################################
                if not skip_other and req.strip().startswith('hourly'):
                    #hourly
                    job.minute.on(0)
                    skip_other = True

            ################################################
            #   every monday/tuesday/wednesday from 00:00 to 00:00
            ################################################
            elif not skip_other:
                #@todo
                #time = re.search('\s?from\s(\d{1,2}):(\d{2})\sto\s(\d{1,2}):(\d{2})[^0-9]+?', req.strip())
                time = re.search('\s?from\s(\d{1,2}):(\d{2})\sto\s(\d{1,2}):(\d{2})[^0-9]+', req)

                #@todo
                if time and time.group(1):
                    job.hour.during(time.group(1), time.group(3))
                    #todo every minute, every 5 minutes
                    job.minute.during(time.group(2), time.group(4)).every(5)
                    skip_other = True

            ################################################
            #   every monday/tuesday/wednesday
            ################################################
            elif not skip_other:
                job.minute.on(0)
                #by default 10:00
                job.hour.on(10)
                skip_other = True

        if not skip_other and req.strip().startswith('day'):

            #cut day word
            req = req.replace('day', '', 1)

            if req.strip().startswith('at '):

                ################################################
                #   every day at 00:00
                ################################################

                time = re.search("[^0-9](\d{1,2})\so'clock", req)
                if time and time.group(1):
                    job.minute.on(0)
                    job.hour.on(time.group(1))
                    skip_other = True

                if not skip_other:

                ################################################
                #   every day at 00:00
                ################################################

                    time = re.search('[^0-9](\d{1,2}):(\d{2})[^0-9]', req)
                    if time and time.group(1) and time.group(2):
                        job.minute.on(time.group(2))
                        job.hour.on(time.group(1))
                        skip_other = True

                ################################################
                #   every day hourly
                ################################################
                if not skip_other and req.strip().startswith('hourly'):
                    #hourly
                    job.minute.on(0)
                    skip_other = True

            ################################################
            #   every day every 5 hours
            ################################################
            if not skip_other and req.strip().startswith('every'):
                req = req.replace('every', '', 1)

                hour = re.search('\s?(\d+)\s+(hour|hours|hs|h)', req)
                if hour and hour.group(1):
                    job.hour.every(hour.group(1))
                    skip_other = True
                else:
                    #if hour presents in human word : one, two etc.
                    hour = re.search('^\s?([a-zA-Z]+?)\s(hours|hour)', req)
                    if hour and hour.group(1):
                        h = text2int(hour.group(1))
                        job.hour.every(h)
                        job.minute.on(0)
                        skip_other = True

            ################################################
            #   every day from 00:00 to 00:00
            ################################################
            elif not skip_other and req.strip().startswith('from'):
                #@todo
                time = re.search('^from\s(\d{1,2}):(\d{2})\sto\s(\d{1,2}):(\d{2})[^0-9]+', req.strip())

                #@todo
                if time and time.group(1):
                    job.hour.during(time.group(1), time.group(3))
                    #todo every minute, every 5 minutes
                    job.minute.during(time.group(2), time.group(4)).every(5)
                    skip_other = True

            ################################################
            #   every day
            ################################################
            elif not skip_other:
                job.minute.on(0)
                #by default 10:00
                job.hour.on(10)
                skip_other = True
                print(job)
            else:
                pass

        if not skip_other and req.strip().startswith('with message'):
            job.minute.on(0)
            #by default 10:00
            job.hour.on(10)
            skip_other = True

        if not skip_other and req.strip().startswith('hour'):
            #every hour
            job.minute.on(0)
            skip_other = True

        if not skip_other and req.strip().startswith('minute'):
            #every minute
            job.minute.every(1)
            skip_other = True

        if not skip_other:

            ################################################
            #   hours
            ################################################
            hour = re.search('^(\d+)\s+(hour|hours|hs|h)', req.strip())

            if hour and hour.group(1):
                job.hour.every(hour.group(1))
                skip_other = True
            else:
                #if hour presents in human word : one, two etc.
                hour = re.search('^([a-zA-Z]+?)\s(hours|hour)', req.strip())
                if hour and hour.group(1):
                    h = text2int(hour.group(1))
                    job.hour.every(h)
                    job.minute.on(0)
                    skip_other = True

        if not skip_other:

            #######################################################################################################
            #   days
            #######################################################################################################

            day = re.search('^(\d+)\s+(days|day|d)', req.strip())

            if day and day.group(1):

                #remove the matched part of the string which describes number of days: ex. 10 days
                req = req.replace(day.group(0), '', 1)

                ################################################
                #   days at 00:00
                ################################################

                if req.strip().startswith('at '):

                    req = req.replace('at', '', 1)
                    ################################################
                    #   days at 8 o'clock
                    ################################################
                    time = re.search("^(\d{1,2})\so'clock", req.strip())

                    if time and time.group(1):
                        job.dow.every(day.group(1))
                        job.minute.on(0)
                        job.hour.on(time.group(1))
                        skip_other = True

                    ################################################
                    #   days hourly
                    ################################################
                    if not skip_other and req.strip().startswith('hourly'):
                        #hourly
                        job.minute.on(0)
                        job.dow.every(day.group(1))
                        skip_other = True

                    ################################################
                    #   days at 00:00
                    ################################################
                    if not skip_other:
                        time = re.search('^(\d{1,2}):(\d{2})[^0-9]', req.strip())

                        if time and time.group(1) and time.group(2):
                            job.dom.every(day.group(1))
                            job.minute.on(time.group(2))
                            job.hour.on(time.group(1))
                            skip_other = True

                ################################################
                #   10 days from 00:00 to 00:00
                ################################################
                if not skip_other and req.strip().startswith('from'):
                    #@todo
                    req = req.replace('from', '', 1)
                    time = re.search('^(\d{1,2}):(\d{2})\sto\s(\d{1,2}):(\d{2})[^0-9]+?', req.strip())

                    if time and time.group(1):
                        job.hour.during(time.group(1), time.group(3))
                        job.dom.every(day.group(1))
                        #todo every 5 minutes
                        job.minute.during(time.group(2), time.group(4)).every(5)
                        skip_other = True

            #################################################
            #   every two days
            #################################################
            elif not skip_other:

                day = re.search('^\s?([a-zA-Z]+?)\s(days|day)', req)

                if day and day.group(1):

                    d = text2int(day.group(1))
                    req = req.replace(day.group(0), '', 1)

                    ################################################
                    #   ten days from 00:00 to 00:00
                    ################################################
                    if not skip_other and req.strip().startswith('from'):

                        time = re.search('^from\s(\d{1,2}):(\d{2})\sto\s(\d{1,2}):(\d{2})[^0-9]+?', req.strip())

                        if time and time.group(1):
                            job.hour.during(time.group(1), time.group(3))
                            job.dom.every(d)

                            #todo every 5 minutes
                            # remove from .. to and check for "every" 5 minutes
                            req = req.replace(day.group(0), '', 1)
                            req = req.replace(time.group(0), '', 1)

                            if req.strip().startswith('every'):

                                mins = re.search('^every\s(\d{1,2})[^0-9]+?(min|minute|minutes)', req.strip())

                                if mins and mins.group(0):
                                    job.minute.during(time.group(2), time.group(4)).every(mins.group(1))
                                    skip_other = True

                                #check once again but now we expect minutes as word not number
                                else:
                                    mins = re.search('^every\s([^0-9\s]+)\s?(min|minute|minutes)', req.strip())
                                    #if exists
                                    if mins and mins.group(1):
                                        m = text2int(mins.group(1))
                                        job.minute.during(time.group(2), time.group(4)).every(m)
                                        skip_other = True
                                    else:
                                        raise
                            # if not starts with "every"
                            else:
                                job.minute.during(time.group(2), time.group(4)).every(5)
                                skip_other = True

                    else:
                        job.dom.every(d)
                        job.minute.on(0)
                        #by default 10:00
                        job.hour.on(10)
                        #print(job)
                        skip_other = True

            else:
                print(req)
                raise
                #job.minute.on(0)
                #job.hour.on(10) #by default 10:00
                #skip_other=True
                #job.dow.every(day.group(1))
                #skip_other = True

        if not skip_other:
            #######################################################################################################
            #   minutes
            #######################################################################################################
            min = re.search('\s?(\d+)\s+(minutes|min|minute|m)', req)

            if min and min.group(1):
                job.minute.every(min.group(1))
            else:
                #if day presents in human word : one, two etc.
                min = re.search('^\s?([a-zA-Z]+?)\s(minutes|min|mins)', req)
                if min and min.group(1):
                    m = text2int(min.group(1))
                    job.minute.every(m)

        cron.write()

        logger.info('adding cronjob %s' % cron.render())
        response = 'ok, cronjob added %s' % job.render()

        if self.req_from == 'jabber':
            todo = {'text': response, 'jmsg': response, 'type': 'response'}
            self.response = todo

        if self.req_from == 'julius':
            from core.broadcast import say, bang
            bang()
            todo = {'say': response, 'text': response, 'type': 'response'}
            self.response = say(self.request.replace('say', '').upper())

        return self.response
#n = Reaction(*{'reserved':''}, **{'req_obj':{'from':'jabber', 'request':'remind me every 2 minutes with "hey don\'t forget about pizza"', 'sender': 'vasilii.pascal@gmail.com'}})
#n.run()
