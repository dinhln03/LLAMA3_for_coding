import threading, queue
import time
import random
import logging

logging.basicConfig(level=logging.DEBUG, format='(%(threadName)-9s) %(message)s',)
NUMBER_OF_THREADS = 4
TIMEOUT_SECONDS = 5

class SampleThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, id=None, kwargs=None):
        super().__init__(group=group, target=target, name=name)
        self.id = id
        self.kwargs = kwargs
        self.queue = kwargs['queue']
        return

    def run(self):
        # do some work here
        logging.debug(f'Tunning thread id={self.id}')
        r = random.uniform(0, 5)
        time.sleep(r)
        self.queue.put(f'Thread id={self.id} finished running in {r} seconds')

if __name__ == '__main__':

    print('Starting threads')
    # create a list to hold running SampleThread object instances
    threads = list()
    # build a single queue to send to all thread objects
    q = queue.Queue() 

    for i in range(NUMBER_OF_THREADS):
        t = SampleThread(id = i, kwargs={'queue':q})
        t.start()
        threads.append(t)

    # wait until all threads are finished
    logging.debug('Waiting for all threads to finish running')
    [t.join() for t in threads]

    logging.debug('All processes are finished running')
    logging.debug('Results')
    while not q.empty():
        logging.debug(q.get())


        
