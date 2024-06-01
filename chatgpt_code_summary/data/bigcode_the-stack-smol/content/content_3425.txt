#sum = 10


def func1():
    #sum = 20
    print('Local1:', sum)

    def func2():
        #sum = 30
        print('Local 2:', sum)

    func2()

func1()
print("Global:", sum([1, 2, 3]))
