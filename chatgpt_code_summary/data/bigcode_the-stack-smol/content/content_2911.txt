#fake database to get the pygame running
import random

questions = ["Question 1?", "Question 2?", "Question 3?", "Question 4?"]
answers = ["Answer 1", "Answer 2", "Answer 3", "Answer 4"]

def get_question():
    return(random.choice(questions))

def get_answer():
    return(random.choice(answers))