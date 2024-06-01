class TopicDTO:
    name = str
    description = str
    popularity = int
    def __init__(self, name="", popularity=0, description = ""):
        self.name=name
        self.popularity=popularity
        self.description = description
    