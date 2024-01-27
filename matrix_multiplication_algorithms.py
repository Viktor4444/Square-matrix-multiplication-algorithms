from itertools import product

import numpy as np


def native_square_matrix_mult(A, B):
    N = len(A)
    C = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    return C


# DEPRECATED!
# def tranpose_square_matrix_mult(A, B):
# 	N = len(A)
# 	B_trans = [list(row) for row in zip(*B)]
# 	C = [[0 for _ in range(N)] for _ in range(N)]
# 	for i in range(N):
# 		for j in range(N):
# 			for k in range(N):
# 				C[i][j] += A[i][k] * B_trans[j][k]
# 	return C


def one_stroke_square_matrix_mult(A, B):
    return [[sum(a * b for a, b in zip(Arow, Bcol)) for Bcol in zip(*B)] for Arow in A]


def split_to_2x2_blocks(matrix):
    return list(map(
        lambda row: np.hsplit(row, 2),
        np.vsplit(matrix, 2)
    ))


def strassen_mul_2x2(lb, rb):
    d = strassen_mul(lb[0][0] + lb[1][1], rb[0][0] + rb[1][1])
    d_1 = strassen_mul(lb[0][1] - lb[1][1], rb[1][0] + rb[1][1])
    d_2 = strassen_mul(lb[1][0] - lb[0][0], rb[0][0] + rb[0][1])

    left = strassen_mul(lb[1][1], rb[1][0] - rb[0][0])
    right = strassen_mul(lb[0][0], rb[0][1] - rb[1][1])
    top = strassen_mul(lb[0][0] + lb[0][1], rb[1][1])
    bottom = strassen_mul(lb[1][0] + lb[1][1], rb[0][0])

    return [[d + d_1 + left - top, right + top],
            [left + bottom, d + d_2 + right - bottom]]


def trivial_mul(left, right):
    height, mid_size = left.shape
    mid_size, width = right.shape

    result = np.zeros((height, width))
    for row, col, mid in product(*map(range, [height, width, mid_size])):
        result[row][col] += left[row][mid] * right[mid][col]

    return result


def strassen_mul(left, right):
	TRIVIAL_MULTIPLICATION_BOUND = 8
	if left.shape[0] <= TRIVIAL_MULTIPLICATION_BOUND:
		return trivial_mul(left, right)
	
	return np.block(
        strassen_mul_2x2(*map(split_to_2x2_blocks, [left, right])))
