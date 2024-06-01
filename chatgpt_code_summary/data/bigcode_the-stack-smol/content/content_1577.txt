from sources import Sources
from stories import Stories

class Bot():

    def __init__(self, config):
        self.url = config.get_url()
        self.sources = None
        self.stories = None

    def load(self):
        self.sources = Sources(self.url)        
        self.stories = Stories(self.sources)        
        return self.stories.load()
    
    def start(self, url):  
        message = 'Бот для сайта {0}'.format(url)
        return message

    def help(self):
        message = "/get - читать истории из: \n\t{0}\n"\
        "/random - случайные истории\n"\
        "/stop - прервать диалог с ботом".format(
            '\n\t'.join(['{0}'.format(y) for (x,y) in self.stories.get_description().items()]))
        return message

    def random(self, num=None, site_names=None):
        if site_names is None:
            site_names = list(self.stories.get_names().keys())  
        sites = list(self.stories.get_names().values())
        messages = []
        stories = self.stories.get(num=num, site_names=site_names,
                                   sites=sites, random=True)
        for s in stories:
            messages.append(s.get().get('story'))
        return messages

    def get(self, num=None, site_names=None):
        if site_names is None:
            site_names = list(self.stories.get_names().keys())    
        sites = list(self.stories.get_names().values())    
        messages = []
        stories = self.stories.get(num=num, site_names=site_names,
                                   sites=sites)
        for s in stories:
            messages.append(s.get().get('story'))
        return messages

    def get_sources_sites(self):
        sites = set()
        for sites_list in self.sources.get():
            for site in sites_list:
                sites.add(site.get('site'))
        return list(sites)
            
    def get_sources_names(self, site):
        names = set()
        for sites_list in self.sources.get():
            for s in sites_list:
                if s.get('site') == site:
                    names.add((s.get('name'), s.get('desc')))
        return list(names)