import requests
import json


class BuddyAPI():
    '''
    An API of buddymojo.com 

    :returns: An API 
    '''

    def __init__(self):
        self.payload = {'type': 'friend',
                        'action': 'finish'}
        self.payloadf = {'userQuizId': 1,
                         'type': 'friend',
                         'stats': '1'}

        self.url = 'https://cn.buddymojo.com/api/v1/quiz/18'
        self.match = 'https://cn.buddymojo.com/match/'

    def send_single_ans(self, ID, name: str):
        '''
        Send a single message to specific id with a specific name.

        :params ID: User quiz id.
        :type ID: int
        :params name: Name you want on the message.
        :type name: str
        '''
        self.data = {'userFullName': name,
                     'userQuizId': 1}
        self.data.update(userQuizId=ID)
        self.payloadf.update(userQuizId=ID)

        try:
            req = requests.request('GET', self.url, params=self.payloadf)
            questions = json.loads(req.text).get('data').get('questions')
            # d = text.get('data')
            # questions = d.get('questions')

            for j, q in enumerate(questions):
                qval = q.get('choosenOption')
                self.data.update(
                    {'questions['+str(j)+'][choosenOption]': qval})

            reqi = requests.post(self.url, params=self.payload, data=self.data)
            print('sending post to userQuizId: '+str(ID))
        except:
            print('User not found')

    def send_range_ans(self, start, end, name: str):
        '''
        Send messages to a range of users id.

        :params start: The start user id.
        :type start: int
        :params end: The end user id.
        :type end: int
        :params name: The name you want.
        :type name: str
        '''
        for i in range(start, end):
            data = {'userFullName': name,
                    'userQuizId': 1}
            data.update(userQuizId=i)
            self.payloadf.update(userQuizId=i)

            try:
                req = requests.request('GET', self.url, params=self.payloadf)
                questions = json.loads(req.text).get('data').get('questions')
                # d = text.get('data')
                # questions = d.get('questions')

                for j, q in enumerate(questions):
                    qval = q.get('choosenOption')
                    data.update({'questions['+str(j)+'][choosenOption]': qval})

                reqi = requests.post(self.url, params=self.payload, data=data)
                print('sending post to userQuizId: '+str(i))
            except:
                continue

    # Still working out
    def get_userQuizId(self, encUserQuizId):
        '''
        Returns a user id string of the encUserQuizId.
        '''
        try:
            req = requests.request('GET', str(match+encUserQuizId))
            data = json.loads(req.text)
            print(data)
        except:
            return 'User not found'

    def get_link(self, ID):
        '''
        Returns a url string of the id.

        :params ID: The id to get the url from.
        :type ID: int
        :returns: A url string.
        :rtype: String
        '''
        self.payloadf.update(userQuizId=ID)

        try:
            req = requests.request('GET', self.url, params=self.payloadf)
            data = json.loads(req.text).get('data').get('encUserQuizId')

            return self.match + data
        except:
            return 'User not found'
