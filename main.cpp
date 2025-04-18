#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <climits>

using namespace std;

double** createMatrix(int size) {
    double** matrix = new double*[size];
    for (int i = 0; i < size; i++) {
        matrix[i] = new double[size];
    }
    return matrix;
}

void freeMatrix(double** matrix, int size) {
    for (int i = 0; i < size; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

void dgemmBlass_v1(double** A, double** B, double** C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < size; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void dgemmBlass_v2(double** A, double** B, double** C, int size) {
    for (int i = 0; i < size; i++) {
        for (int k = 0; k < size; k++) {
            for (int j = 0; j < size; j++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void dgemmBlass_v3(double** A, double** B, double** C, int size, int block_size) {
    for (int i = 0; i < size; i += block_size) {
        for (int j = 0; j < size; j += block_size) {
            for (int k = 0; k < size; k += block_size) {
                int i_end = min(i + block_size, size);
                int j_end = min(j + block_size, size);
                int k_end = min(k + block_size, size);

                for (int ii = i; ii < i_end; ++ii) {
                    for (int kk = k; kk < k_end; ++kk) {
                        double a = A[ii][kk];
                        for (int jj = j; jj < j_end; ++jj) {
                            C[ii][jj] += a * B[kk][jj];
                        }
                    }
                }
            }
        }
    }
}

void fillRandomMatrix(double** matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = (rand() % 100000) / 100.0;
        }
    }
}

void testOptimalBlockSize() {
    const int test_size = 2048;
    const int trials = 3;

    double** A = createMatrix(test_size);
    double** B = createMatrix(test_size);
    double** C = createMatrix(test_size);

    fillRandomMatrix(A, test_size);
    fillRandomMatrix(B, test_size);

    cout << "Поиск оптимального размера блока для матриц " << test_size << "x" << test_size << ":\n";

    int best_block = 0;
    double best_time = INT_MAX;

    for (int block_size = 16; block_size <= 256; block_size *= 2) {
        double total_time = 0;

        for (int t = 0; t < trials; t++) {
            for (int i = 0; i < test_size; i++) {
                memset(C[i], 0, test_size * sizeof(double));
            }

            clock_t start = clock();
            dgemmBlass_v3(A, B, C, test_size, block_size);
            clock_t end = clock();

            total_time += double(end - start) / CLOCKS_PER_SEC;
        }

        double avg_time = total_time / trials;
        cout << "Размер блока: " << block_size << "\tСреднее время: " << avg_time << " сек" << endl;

        if (avg_time < best_time) {
            best_time = avg_time;
            best_block = block_size;
        }
    }

    cout << "\nОптимальный размер блока: " << best_block << " (лучшее время: " << best_time << " сек)" << endl;

    freeMatrix(A, test_size);
    freeMatrix(B, test_size);
    freeMatrix(C, test_size);
}

int main(int argc, char* argv[]) {
    int size;
    if (argc > 1)
    {
        size = atoi(argv[1]);

        srand(time(0));

        double** A = createMatrix(size);
        double** B = createMatrix(size);
        double** C = createMatrix(size);

        fillRandomMatrix(A, size);
        fillRandomMatrix(B, size);

        clock_t start = clock();
        dgemmBlass_v1(A, B, C, size);
        clock_t end = clock();

        double final_time = double(end - start) / CLOCKS_PER_SEC;
        cout << "Итоговое время умножения для матриц " << size << "x" << size << ": " << final_time << " секунд\n";

        freeMatrix(A, size);
        freeMatrix(B, size);
        freeMatrix(C, size);
    } else
    {
        cout << "Введите номер способа оптимизации: 1 - без оптимизации, 2 - построчный перебор, 3 - блочный перебор" << endl;
        int chose;
        cin >> chose;

        if (chose == 1)
        {
            cout << "Время перемножения без оптимизации" << endl;
            for (int i = 1000; i < 3000; i += 1000)
            {
                double** A = createMatrix(i);
                double** B = createMatrix(i);
                double** C = createMatrix(i);

                fillRandomMatrix(A, i);
                fillRandomMatrix(B, i);

                clock_t start = clock();
                dgemmBlass_v1(A, B, C, i);
                clock_t end = clock();

                double final_time = double(end - start) / CLOCKS_PER_SEC;
                cout << "Итоговое время умножения для матриц " << i << "x" << i << ": " << final_time << " секунд\n";

                freeMatrix(A, i);
                freeMatrix(B, i);
                freeMatrix(C, i);
            }
        }
        else if (chose == 2)
        {
            cout << "Построчный перебор:" << endl;
            for (int i = 1000; i < 3000; i += 1000)
            {
                double** A = createMatrix(i);
                double** B = createMatrix(i);
                double** C = createMatrix(i);

                fillRandomMatrix(A, i);
                fillRandomMatrix(B, i);

                clock_t start = clock();
                dgemmBlass_v2(A, B, C, i);
                clock_t end = clock();

                double final_time = double(end - start) / CLOCKS_PER_SEC;
                cout << "Итоговое время умножения для матриц " << i << "x" << i << ": " << final_time << " секунд\n";

                freeMatrix(A, i);
                freeMatrix(B, i);
                freeMatrix(C, i);
            }
        }
        else
        {
            cout << "Введите размер блока: ";
            int block_size;
            cin >> block_size;
            cout << "Блочный перебор:" << endl;
            for (int i = 1000; i < 3000; i += 1000)
            {
                double** A = createMatrix(i);
                double** B = createMatrix(i);
                double** C = createMatrix(i);

                fillRandomMatrix(A, i);
                fillRandomMatrix(B, i);

                clock_t start = clock();
                dgemmBlass_v3(A, B, C, i, block_size);
                clock_t end = clock();

                double final_time = double(end - start) / CLOCKS_PER_SEC;
                cout << "Итоговое время умножения для матриц " << i << "x" << i << ": " << final_time << " секунд\n";

                freeMatrix(A, i);
                freeMatrix(B, i);
                freeMatrix(C, i);
            }
        }
    }

    return 0;
}