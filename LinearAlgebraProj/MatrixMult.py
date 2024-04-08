import numpy as np

key26 = [[3, 11],
        [4, 15]]

invkey26 = [[15, 15],
            [22, 3]]

key29 = [[10, 15],
         [5, 9]]

invkey29 = [[18, 28],
            [19, 20]]

sampleMatrix = [[5, 1],
                [25, 16]]   #F B Z Q


def mod_func(matrix, mod):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = matrix[i][j] % mod

    return matrix

def matrix_mult(key, matrix):
    result = np.zeros((len(key), len(matrix[0])), dtype = int)

    for i in range(len(key)):
        for j in range(len(matrix[0])):
            for k in range(len(matrix)):

                result[i][j] += key[i][k] * matrix[k][j]

    return result

def encipher(message):
    enciphered_message = matrix_mult(key26, message)
    mod_func(enciphered_message, 26)
    enciphered_message = matrix_mult(key29, enciphered_message)
    mod_func(enciphered_message, 29)

    return enciphered_message

def decipher(message):
    deciphered_message = matrix_mult(invkey29, message)
    mod_func(deciphered_message, 29)
    deciphered_message = matrix_mult(invkey26, deciphered_message)
    mod_func(deciphered_message, 26)

    return deciphered_message

final = encipher(sampleMatrix)
print(final)
finalDecipher = decipher(final)
print(finalDecipher)






