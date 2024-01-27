import csv
import time

import numpy as np
from matrix_multiplication_algorithms import (native_square_matrix_mult,
                                              one_stroke_square_matrix_mult,
                                              strassen_mul)


def test_mult():
    methods = {
        "native": native_square_matrix_mult,
        "one stroke": one_stroke_square_matrix_mult,
        "numpy emplemented": np.matmul,
        "strassen numpy": strassen_mul
    }

    with open("compare_result.csv", 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        zero_row = ["square matrix size"]
        zero_row.extend([method_name for method_name in methods.keys()])
        csv_writer.writerow(zero_row)

        for i in range(1, 10):
            N = 2**i
            result_row = [N]
            a_np = np.random.randint(100, size=(N, N))
            b_np = np.random.randint(100, size=(N, N))

            for method_name, mult_method in methods.items():
                if method_name in ["native", "one stroke"]:
                    A = a_np.tolist()
                    B = b_np.tolist()
                else:
                    A = a_np
                    B = b_np

                start_time = time.time()
                C = mult_method(A, B)
                end_time = time.time()
                del C

                result_row.append(f"{(end_time - start_time):.10f}")
            csv_writer.writerow(result_row)
            print(f"Finish N = {N}")


if __name__ == "__main__":
    test_mult()
