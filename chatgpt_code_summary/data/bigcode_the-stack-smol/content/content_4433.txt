'''
Exercise 5: Vectors
A vector of dimension 𝑛𝑛 can be represented by a list in Python. For example, a vector of
dimension 3 could represent a point in space, and a vector of dimension 4 could represent a
point in space and time (the fourth dimension being the time). In mathematical notation, a
vector of dimension 3 is represented as follow:
�
𝑎𝑎
𝑏𝑏
𝑐𝑐
�
The vector could be stored in a Python list [a, b, c]. There are two simple operations that
can be done on vector, and the result of the two operation is also a vector. The two operations
are:
Scalar product: 𝜆𝜆 ∙ �
𝑎𝑎
𝑏𝑏
𝑐𝑐
� = �
𝜆𝜆 ∙ 𝑎𝑎
𝜆𝜆 ∙ 𝑏𝑏
𝜆𝜆 ∙ 𝑐𝑐
�
Addition: �
𝑎𝑎
𝑏𝑏
𝑐𝑐
� + �
𝑑𝑑
𝑒𝑒
𝑓𝑓
� = �
𝑎𝑎 + 𝑑𝑑
𝑏𝑏 + 𝑒𝑒
𝑐𝑐 + 𝑓𝑓
�

Implement two functions:
1. scalar_product(scalar, vector) where scalar is a float and vector is a list
of float. The function returns the scalar product of the two parameters.
2. vector_addition(vector1, vector2) where vector1 and vector2 are
lists of float. The function returns the vector addition of the two parameters. If
vector1 and vector2 don’t have the same dimension, you should print an error
message and return None.
'''
def scalar_product(scalar, vector):
    for x in range(len(vector)):
        vector[x]*=scalar
    return vector

def vector_addition(vector1, vector2):
    if(len(vector2)!=len(vector1)):
        return 'Error'
    for x in range(len(vector1)):
        vector1[x]=int(vector1[x])+int(vector2[x])
    return vector1


print(scalar_product(int(input('Enter Scalar Value: ')),input('Enter a Matrix seperated by coma: ').split(',') ))

print(vector_addition( input('Enter first Matrix to add: ').split(',') , input('Enter second Matrix to add: ').split(',') ))