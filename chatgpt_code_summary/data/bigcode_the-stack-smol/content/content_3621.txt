import uuid

from turkit2.common import TextClassification
from turkit2.qualifications import Unique, Locale, AcceptRate
from utils import get_client

client = get_client()

quals = [Locale(), AcceptRate()]
task = TextClassification(client, 'Test3', '0.01', 'test test', 600, 6000, ['positive', 'negative'], question='Which class does this text match, positive or negative?', qualifications=quals)

documents = [f'test{i}' for i in range(5)]

def proc(text):
    for answer, assignment in task.ask(verbosity=100, text=text):
        print(answer)
        print(assignment)

def main():
    tasks = []
    for text in documents:
        proc(text)

main()
