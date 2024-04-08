import MatrixMult           
import numpy as np

'''
Project is an absolute mess right now
Will reorganize and rename files/functions/variables
'''

random_key_26 = np.random.randint(0, 10, size = (3,3))      #Generate random keys
random_key_29 = np.random.randint(0, 10, size = (3,3))

determinant_26 = int(round(np.linalg.det(random_key_26)))   #Numpy function finds determinant
determinant_29 = int(round(np.linalg.det(random_key_29)))

test_Matrix = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])

print(test_Matrix)


while determinant_26 % 2 == 0 or determinant_26 % 13 == 0:          #Loops check to make sure gcd of determinant and modulus is 1
    random_key_26 = np.random.randint(0, 10, size = (3,3))          #Generates new key until satisfied
    determinant_26 = int(round(np.linalg.det(random_key_26)))

while determinant_29 % 29 == 0:
    random_key_29 = np.random.randint(0, 10, size = (3,3))
    determinant_29 = int(round(np.linalg.det(random_key_29)))

    
def modular_inverse(det, mod):      #Modular inverse found by power function
    return pow(det, -1, mod)        #Finds det^x % 26 == 1, -1 indicates inverse




modinverse_26 = modular_inverse(determinant_26, 26)
modinverse_29 = modular_inverse(determinant_29, 29)




#Functions for finding inverse matrix
def matrix_cofactor(matrix, row, column):
    eliminate = np.delete(np.delete(matrix, row, axis=0), column, axis=1)             #Eliminates rows/columns for finding , axis=0 indicates horiz., axis=1 indicates vert.     
    minor = ((-1) ** (row + column)) * int(np.round(np.linalg.det(eliminate)))        #Finds minors of the matrix (determinants of square submatrices)
    return minor


def matrix_adjugate(matrix):
    cofactor_matrix = np.zeros(matrix.shape, dtype=int)
    for i in range(matrix.shape[0]):                                #.shape[0] Returns number of rows of array
        for j in range(matrix.shape[1]):                            #.shape[1] Returns number of columns of array
            cofactor_matrix[i, j] = matrix_cofactor(matrix, i, j)
    adjugate_matrix = cofactor_matrix.T                             #Swaps matrix rows with columns
    return adjugate_matrix




inverted_random_key_26 = matrix_adjugate(random_key_26)
inverted_random_key_29 = matrix_adjugate(random_key_29)


#Multiplies adjugate matrices by respective modular inverse for finished inverted matrix
for i in range(len(inverted_random_key_26)):                    
    for j in range(len(inverted_random_key_26[i])):
        inverted_random_key_26[i][j] = inverted_random_key_26[i][j] * modinverse_26

for i in range(len(inverted_random_key_29)):
    for j in range(len(inverted_random_key_29[i])):
        inverted_random_key_29[i][j] = inverted_random_key_29[i][j] * modinverse_29



finalMatrix_26 = MatrixMult.mod_func(inverted_random_key_26, 26)
print(finalMatrix_26)
finalMatrix_29 = MatrixMult.mod_func(inverted_random_key_29, 29)
print(finalMatrix_29)

test_mult = MatrixMult.matrix_mult(random_key_26, test_Matrix)
test_mult = MatrixMult.mod_func(test_mult, 26)
print(test_mult)
test_mult2 = MatrixMult.matrix_mult(random_key_29, test_mult)
test_mult2 = MatrixMult.mod_func(test_mult2, 29)
print(test_mult2)
test_mult3 = MatrixMult.matrix_mult(inverted_random_key_29, test_mult2)
test_mult3 = MatrixMult.mod_func(test_mult3, 29)
print(test_mult3)
test_mult4 = MatrixMult.matrix_mult(inverted_random_key_26, test_mult3)
test_mult4 = MatrixMult.mod_func(test_mult4, 26)
print(test_mult4)