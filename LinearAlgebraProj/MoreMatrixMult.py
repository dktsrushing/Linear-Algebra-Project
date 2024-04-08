import MatrixMult
import numpy as np

random_key_26 = np.random.randint(0, 10, size = (3,3))
random_key_29 = np.random.randint(0, 10, size = (3,3))

determinant_26 = int(round(np.linalg.det(random_key_26)))
determinant_29 = int(round(np.linalg.det(random_key_29)))

test_Matrix = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])

print(test_Matrix)


while determinant_26 % 2 == 0 or determinant_26 % 13 == 0:
    random_key_26 = np.random.randint(0, 10, size = (3,3))
    determinant_26 = int(round(np.linalg.det(random_key_26)))

while determinant_29 % 29 == 0:
    random_key_29 = np.random.randint(0, 10, size = (3,3))
    determinant_29 = int(round(np.linalg.det(random_key_29)))

    
def modular_inverse(det, mod):
    return pow(det, -1, mod)




modinverse_26 = modular_inverse(determinant_26, 26)
modinverse_29 = modular_inverse(determinant_29, 29)





def calculate_cofactor(matrix, row, column):
    minor = np.delete(np.delete(matrix, row, axis=0), column, axis=1)
    cofactor = ((-1) ** (row + column)) * int(np.round(np.linalg.det(minor)))
    return cofactor


def calculate_adjugate(matrix):
    cofactor_matrix = np.zeros(matrix.shape, dtype=int)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            cofactor_matrix[i, j] = calculate_cofactor(matrix, i, j)
    adjugate_matrix = cofactor_matrix.T  
    return adjugate_matrix




inverted_random_key_26 = calculate_adjugate(random_key_26)
inverted_random_key_29 = calculate_adjugate(random_key_29)


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