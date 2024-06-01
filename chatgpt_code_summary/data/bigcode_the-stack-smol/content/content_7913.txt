from pyquilted.quilted.section import Section


class Work(Section):
    """The work section in a quilted resume

       The work object is a complex section. It contains blocks of jobs
       and optionally a list of slugs. As a section it mixes in the
       sectionable functionality.
    """
    def __init__(self, blocks=None, slugs=None, icon=None):
        self.label = 'Work'
        self.icon = icon or 'fa-briefcase'
        self.blocks = blocks or []
        self.compact = False

    def add_job(self, job):
        self.blocks.append(vars(job))

    def add_slugs(self, slugs):
        self.slugs = slugs


class Job:
    """The job block in the work section"""
    def __init__(self, dates=None, location=None, company=None, title=None,
                 slugs=None, previously=None, **kwargs):
        self.dates = dates
        self.location = location
        self.company = company
        self.title = title
        self.slugs = slugs
        self.history = History(previously=previously).to_dict()


class Slugs():
    """The additional list of slugs in the work section"""
    def __init__(self, slugs=None):
        self.blocks = slugs


class History():
    def __init__(self, previously=None):
        self.previously = previously

    def to_dict(self):
        if self.previously:
            return vars(self)
        return None
