import multiprocessing as mp
import time


def f(name, timeout, queue):
    time.sleep(timeout)
    print('hello', name)
    queue.put(name + ' done!')


queue = mp.SimpleQueue()  # queue for communicating with the processes we will spawn
bob = mp.Process(target=f, args=('bob', 0.3, queue))
bob.start()  # start the process
alice = mp.Process(target=f, args=('alice', 0.1, queue))
alice.start()  # start the process

# wait for processes to complete
bob.join()
alice.join()

# print results from intercommunication object
for result in iter(queue.get, None):
    print(result)