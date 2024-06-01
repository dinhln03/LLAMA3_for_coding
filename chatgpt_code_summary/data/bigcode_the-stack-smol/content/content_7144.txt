#!/usr/bin/env python
"""
Memory Loss
github.com/irlrobot/memory_loss
"""
from __future__ import print_function
from random import choice, shuffle
from alexa_responses import speech_with_card
from brain_training import QUESTIONS

def handle_answer_request(player_answer, session):
    """check if the answer is right, adjust score, and continue"""
    print("=====handle_answer_request fired...")
    attributes = {}
    should_end_session = False
    print("=====answer heard was:  " + player_answer)

    current_question = session['attributes']['question']
    correct_answer = current_question['answer']
    shuffle(QUESTIONS)
    next_question = choice(QUESTIONS)

    if correct_answer == player_answer:
        answered_correctly = True
    else:
        log_wrong_answer(current_question['question'], player_answer, correct_answer)
        answered_correctly = False

    next_tts = "Next question in 3... 2... 1... " + next_question['question']
    attributes = {
        "question": next_question,
        "game_status": "in_progress"
    }

    if answered_correctly:
        speech_output = "Correct!" + next_tts
        card_title = "Correct!"
    else:
        speech_output = "Wrong!" + next_tts
        card_title = "Wrong!"

    card_text = "The question was:\n" + current_question['question']
    return speech_with_card(speech_output, attributes, should_end_session,
                            card_title, card_text, answered_correctly)

def log_wrong_answer(question, answer, correct_answer):
    """log all questions answered incorrectly so i can analyze later"""
    print("[WRONG ANSWER]:" + question + ":" + answer + ":" + correct_answer)
