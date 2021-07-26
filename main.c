#include "adeline.h"
#include <stddef.h>

int main(int argc, char **argv) {
    const int N = 10;

    adeline_t ada = create_adaline(3, 0.1);

    const double samples[10][3] = {{8, 1, 2},   {2, -8, -3}, {-1, 3, 7}, {3, -1, -6}, {2, 1, 6},
                                   {3, -2, -8}, {-3, -3, 7}, {-3, 6, 9}, {-4, 2, 8},  {-5, -7, 9}};

    double **X = (double **)malloc(N * sizeof(double *));
    const size_t Y[10] = {1, -1, 1, -1, -1, -1, 1, 1, 1, -1};

    for (size_t index = 0; index < N; index++) {
        X[index] = (double *)samples[index];
    }

    fit_adaline(&ada, X, Y, N);

    double test[] = {3, -3};
    size_t pred = predict_adaline(&ada, test, NULL);

    printf("Predict for x=(3,-3): % d", pred);

    free(X);
    delete_adaline(&ada);
}
