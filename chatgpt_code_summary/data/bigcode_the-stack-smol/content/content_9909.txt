
def get_workout(day):
    if day == 'Monday':
        return 'Chest+biceps'
    elif day == 'Tuesday':
        return 'Back+triceps'
    elif day == 'Wednesday':
        return 'Core'
    elif day == 'Thursday':
        return 'Legs'
    elif day == 'Friday':
        return 'Shoulders'
    elif day in ('Saturday', 'Sunday'):
        return 'Rest'
    raise ValueError('Not a day')

# use a dict to sort it out
workwouts = {
    'Monday': 'Chest+biceps',
    'Tuesday': 'Back+triceps',
    'Wednesday': 'Core',
    'Thursday': 'Legs',
    'Friday': 'Shoulders',
    'Saturday': 'Rest',
    'Sunday': 'Rest'
}

# one other way
days = 'Monday Tuesday Wednesday Thursday Friday Saturday Sunday'.split()
routines = 'Chest+biceps Back+triceps Core Legs Shoulders Rest Rest'.split()
workouts2 = dict(zip(days, routines))
print(workouts2)

def get_workout(day):
    routine = workwouts.get(day)
    if routine is None:
        raise ValueError('Not a day')
    return routine

print(get_workout('Monday'))