from datetime import datetime


class Worklog:
    def __init__(self, name: str, date: datetime, issue_id: str, duration: float):
        self.issue_id = issue_id
        self.name = name
        self.date = date
        self.duration = duration
