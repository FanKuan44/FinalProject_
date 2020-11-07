import numpy as np


def combine_matrix1D_and_opsINT(matrix, ops):
    x = matrix.tolist()
    indices = [6, 11, 15, 18, 20]
    acc = 0
    for i in range(len(ops)):
        x.insert(indices[i] + acc, ops[i])
        acc += 1
    return x


def split_to_matrix1D_and_opsINT(x):
    matrix_1D = []
    ops_INT = []
    for i in range(len(x)):
        if i == 6 or i == 12 or i == 17 or i == 21 or i == 24:
            ops_INT.append(x[i])
        else:
            matrix_1D.append(x[i])
    return np.array(matrix_1D), ops_INT


def create_model():
    """
    create matrix2D and ops_STRING for making a real model
    :return:
    matrix2D
    ops_STRING
    """
    matrix_1D = []
    n_edges = 0
    allowed_edges = [0, 1]
    for _ in range(21):
        edge = np.random.choice(allowed_edges)
        if n_edges < 9:
            if edge == 1:
                n_edges += 1
            matrix_1D.append(edge)
        else:
            matrix_1D.append(0)
    matrix_2D = encoding_matrix(matrix_1D)

    ops_INT = np.random.choice([0, 1, 2], size=5).tolist()
    ops_STRING = encoding_ops(ops_INT)

    return matrix_2D, ops_STRING


def encoding_ops(ops_INT):
    """
    convert ops_INT to ops_STRING
    :param ops_INT
    :return: ops_STRING
    """
    ops_STRING = ['input']
    for op in ops_INT:
        if op == 0:
            ops_STRING.append('conv1x1-bn-relu')
        elif op == 1:
            ops_STRING.append('conv3x3-bn-relu')
        elif op == 2:
            ops_STRING.append('maxpool3x3')
    ops_STRING.append('output')
    return ops_STRING


def decoding_ops(ops_STRING):
    """
    convert ops_STRING to ops_INT
    :param ops_STRING
    :return: ops_INT
    """
    ops_INT = []
    for op in ops_STRING[1:-1]:
        if op == 'conv1x1-bn-relu':
            ops_INT.append(0)
        elif op == 'conv3x3-bn-relu':
            ops_INT.append(1)
        else:
            ops_INT.append(2)
    return ops_INT


def encoding_matrix(matrix1D):
    """
    convert matrix 1D to 2D
    :param matrix1D
    :return: matrix2D
    """
    matrix2D = np.zeros((7, 7), dtype=np.int)
    matrix2D[0][1:] = matrix1D[:6]
    matrix2D[1][2:] = matrix1D[6:11]
    matrix2D[2][3:] = matrix1D[11:15]
    matrix2D[3][4:] = matrix1D[15:18]
    matrix2D[4][5:] = matrix1D[18:20]
    matrix2D[5][6:] = matrix1D[-1]
    return matrix2D


def decoding_matrix(matrix2D):
    """
    convert matrix in 2D to 1D
    :param matrix2D
    :return: matrix1D
    """
    matrix1D = np.zeros(21, dtype=np.int)
    matrix1D[:6] = matrix2D[0][1:]
    matrix1D[6:11] = matrix2D[1][2:]
    matrix1D[11:15] = matrix2D[2][3:]
    matrix1D[15:18] = matrix2D[3][4:]
    matrix1D[18:20] = matrix2D[4][5:]
    matrix1D[-1] = matrix2D[5][6:]
    return matrix1D


if __name__ == '__main__':
    MATRIX_2D, OPS_STRING = create_model()
    MATRIX_1D = decoding_matrix(MATRIX_2D)
    OPS_INT = decoding_ops(OPS_STRING)
    print(MATRIX_1D, OPS_INT)
    X = combine_matrix1D_and_opsINT(matrix=MATRIX_1D, ops=OPS_INT)
    MATRIX_1D, OPS_INT = split_to_matrix1D_and_opsINT(X)
    print(MATRIX_1D, OPS_INT)

