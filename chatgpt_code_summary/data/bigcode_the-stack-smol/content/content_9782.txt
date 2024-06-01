import numpy as np 
import sys

class RBF():
    def __init__(self, Input, Output, Ptypes, Nclasses):

        self.input = Input
        self.hidden = Ptypes * Nclasses
        self.output = Output
        self.ptypes = Ptypes
        self.nclasses = Nclasses

        self.protos = 0
        self.weights = 0
        self.spread = 0
    
    def createPrototypes(self, data):

        groups = np.random.randint(0, data.shape[0], size = (self.hidden))
        
        prototypes = np.zeros((self.hidden, data.shape[1]))
        
        i = 0
        
        for element in groups:        
            prototypes[i] = data[element, :]
            i += 1
    
        self.protos = prototypes

    def sigma(self):
        
        temp = 0
        
        for i in range(self.hidden):
            for j in range(self.hidden):
                distance = np.square(np.linalg.norm(self.protos[i] - self.protos[j]))
                
                if distance > temp:
                    temp = distance
                    
        self.spread = temp/np.sqrt(self.hidden)

    def train(self, data, classes):

        self.createPrototypes(data)
        self.sigma()
        hidden_out = np.zeros(shape=(0,self.hidden))
        
        for data in data:
            output=[]
            
            for proto in self.protos:
                distance = np.square(np.linalg.norm(data - proto))
                neuron_output = np.exp(-(distance)/(np.square(self.spread)))
                output.append(neuron_output)
            hidden_out = np.vstack([hidden_out,np.array(output)])
    
        self.weights = np.dot(np.linalg.pinv(hidden_out), classes)

    def test(self, data, classes):
        
        right = 0
        
        for i in range(len(data)):
            
            d = data[i]
            output = []
            
            for proto in self.protos:
                distance = np.square(np.linalg.norm(d-proto))
                neuron_output = np.exp(-(distance)/np.square(self.spread))
                output.append(neuron_output)
             
            network_output = np.dot(np.array(output),self.weights)
            
            print ("Expected: ", classes[i].argmax(axis=0) +1)
            print ("Result: ", network_output.argmax(axis=0) + 1)
            print ()

            if network_output.argmax(axis=0) + 1 == classes[i].argmax(axis=0) +1:
                right += 1
                
        print ("Accuracy(%): ", (right * 100) / len(data))

def read_iris(percentage):
    
    dataset = np.loadtxt('iris.data', delimiter=',', skiprows=0)

    np.random.shuffle(dataset)
    
    q = int(dataset.shape[0] * percentage) + 2
    
    X_training = dataset[0:q, 0:4]
    Y_training = dataset[0:q, 4]
    
    X_test = dataset[q:150, 0:4]
    Y_test = dataset[q:150, 4]
    
    return X_training, Y_training, X_test, Y_test

def process_iris_data(data):
        
    p_data = np.zeros((data.shape[0], data.shape[1]))

    max_col1 = np.amax(data[:,0])
    max_col2 = np.amax(data[:,1])
    max_col3 = np.amax(data[:,2])
    max_col4 = np.amax(data[:,3])

    for n in range(len(data)):
            
        p_data[n, 0] = data[n,0] / max_col1
        p_data[n, 1] = data[n,1] / max_col2
        p_data[n, 2] = data[n,2] / max_col3
        p_data[n, 3] = data[n,3] / max_col4

    return p_data

def process_iris_labels(labels, operation):
        
    if operation == 0:
        
        p_labels = np.zeros((labels.shape[0], 3))

        for n in range(len(labels)):
            p_labels[n, int(labels[n])] = 1 

        return p_labels
    else:
        p_labels = np.argmax(labels, axis=1)
        return p_labels


if __name__ == '__main__':
    
    # input params
    # percentage 
    
    parameters = (sys.argv)
    print(parameters)

    x1, y1, x2, y2 = read_iris(float(parameters[1]))
    xp = process_iris_data(x1)
    yp = process_iris_labels(y1,0)

    nn = RBF(xp.shape[1], y1.shape[0], xp.shape[1], 3)

    nn.train(xp, yp) 

    xp = process_iris_data(x2)
    yp = process_iris_labels(y2,0)
    nn.test(xp, yp)