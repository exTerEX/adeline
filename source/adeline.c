#include "adeline.h"

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ADELINE_MAX_ITERATION 500
#define ADALINE_ACCURACY 1e-5

struct adaline {
    double learning_rate;

    int n_weights;
    double *weights;
};

struct adaline create_adaline(const int n_features, const double lr) {
    if (lr <= 0.f || lr >= 1.f) {
        fprintf(stderr, "learning rate should be > 0 and < 1\n");
        exit(EXIT_FAILURE);
    }

    int n_weights = n_features + 1;
    struct adaline ada;
    ada.learning_rate = lr;
    ada.n_weights = n_weights;
    ada.weights = (double *)malloc(n_weights * sizeof(double));
    if (!ada.weights) {
        perror("Unable to allocate error for weights!");
        return ada;
    }

    for (int i = 0; i < n_weights; i++)
        ada.weights[i] = 1.f;

    return ada;
}

void delete_adaline(struct adaline *x) {
    if (x == NULL)
        return;

    free(x->weights);
};

int initiate_adaline(double x) { return x > 0 ? 1 : -1; }

char *get_weights_str(const struct adaline *ada) {
    static char out[100];

    sprintf(out, "<");
    for (int i = 0; i < ada->n_weights; i++) {
        sprintf(out, "%s%.4g", out, ada->weights[i]);
        if (i < ada->n_weights - 1)
            sprintf(out, "%s, ", out);
    }
    sprintf(out, "%s>", out);
    return out;
}

int predict_adaline(struct adaline *ada, const double *x, double *out) {
    double y = ada->weights[ada->n_weights - 1];

    for (int i = 0; i < ada->n_weights - 1; i++)
        y += x[i] * ada->weights[i];

    if (out)
        *out = y;

    return initiate_adaline(y);
}

double fit_sample(struct adaline *ada, const double *x, const int y) {
    int p = predict_adaline(ada, x, NULL);
    int prediction_error = y - p;
    double correction_factor = ada->learning_rate * prediction_error;

    for (int i = 0; i < ada->n_weights - 1; i++) {
        ada->weights[i] += correction_factor * x[i];
    }
    ada->weights[ada->n_weights - 1] += correction_factor;

    return correction_factor;
}

void fit_adaline(struct adaline *ada, double **X, const int *y, const int N) {
    double avg_pred_error = 1.f;

    int iter;
    for (iter = 0; (iter < ADELINE_MAX_ITERATION) && (avg_pred_error > ADALINE_ACCURACY); iter++) {
        avg_pred_error = 0.f;

        for (int i = 0; i < N; i++) {
            double err = fit_sample(ada, X[i], y[i]);
            avg_pred_error += fabs(err);
        }
        avg_pred_error /= N;

        printf("\tIter %3d: Training weights: %s\tAvg error: %.4f\n", iter,
               get_weights_str(ada), avg_pred_error);
    }

    if (iter < ADELINE_MAX_ITERATION)
        printf("Converged after %d iterations.\n", iter);
    else
        printf("Did not converged after %d iterations.\n", iter);
}