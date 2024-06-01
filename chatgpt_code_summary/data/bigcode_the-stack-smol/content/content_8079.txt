class SpaceAge(object):
    ORBITAL_PERIOD = 31557600  # seconds

    def __init__(self, seconds):
        self.seconds = seconds

    def on_mercury(self) -> float:
        return round(self.seconds / float(self.ORBITAL_PERIOD) / 0.2408467, 2)

    def on_venus(self) -> float:
        return round(self.seconds / float(self.ORBITAL_PERIOD) / 0.61519726, 2)

    def on_earth(self) -> float:
        return round(self.seconds / float(self.ORBITAL_PERIOD), 2)

    def on_mars(self) -> float:
        return round(self.seconds / float(self.ORBITAL_PERIOD) / 1.8808158, 2)

    def on_jupiter(self) -> float:
        return round(self.seconds / float(self.ORBITAL_PERIOD) / 11.862615, 2)

    def on_saturn(self) -> float:
        return round(self.seconds / float(self.ORBITAL_PERIOD) / 29.447498, 2)

    def on_uranus(self) -> float:
        return round(self.seconds / float(self.ORBITAL_PERIOD) / 84.016846, 2)

    def on_neptune(self) -> float:
        return round(self.seconds / float(self.ORBITAL_PERIOD) / 164.79132, 2)

