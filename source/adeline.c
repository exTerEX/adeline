/*
 * Copyright 2021 Andreas Sagen
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "adeline.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ADELINE_MAX_ITERATION 500
#define ADALINE_ACCURACY 1e-5

adeline_t create_adaline(const size_t n_features, const double lr) {
    if (lr <= 0.f || lr >= 1.f) {
        fprintf(stderr, "learning rate should be > 0 and < 1\n");
        exit(EXIT_FAILURE);
    }

    size_t n_weights = n_features + 1;
    adeline_t ada;
    ada.learning_rate = lr;
    ada.n_weights = n_weights;
    ada.weights = (double *)malloc(n_weights * sizeof(double));
    if (!ada.weights) {
        perror("Unable to allocate error for weights!");
        return ada;
    }

    for (size_t i = 0; i < n_weights; i++) {
        ada.weights[i] = 1.f;
    }

    return ada;
}

void delete_adaline(adeline_t *x) {
    if (x == NULL) {
        return;
    }

    free(x->weights);
};

char *get_weights_str(const adeline_t *ada) {
    static char out[100];

    sprintf(out, "<");
    for (size_t i = 0; i < ada->n_weights; i++) {
        sprintf(out, "%s%.4g", out, ada->weights[i]);
        if (i < ada->n_weights - 1)
            sprintf(out, "%s, ", out);
    }
    sprintf(out, "%s>", out);
    return out;
}

size_t predict_adaline(adeline_t *ada, const double *x, double *out) {
    double y = ada->weights[ada->n_weights - 1];

    for (size_t i = 0; i < ada->n_weights - 1; i++) {
        y += x[i] * ada->weights[i];
    }
    if (out) {
        *out = y;
    }
    return y > 0 ? 1 : -1;
}

double fit_sample(adeline_t *ada, const double *x, const size_t y) {
    size_t p = predict_adaline(ada, x, NULL);
    size_t prediction_error = y - p;
    double correction_factor = ada->learning_rate * prediction_error;

    for (size_t i = 0; i < ada->n_weights - 1; i++) {
        ada->weights[i] += correction_factor * x[i];
    }
    ada->weights[ada->n_weights - 1] += correction_factor;

    return correction_factor;
}

void fit_adaline(adeline_t *ada, double **X, const size_t *y, const size_t N) {
    double avg_pred_error = 1.f;

    size_t iter;
    for (iter = 0; (iter < ADELINE_MAX_ITERATION) && (avg_pred_error > ADALINE_ACCURACY); iter++) {
        avg_pred_error = 0.f;

        for (size_t i = 0; i < N; i++) {
            double err = fit_sample(ada, X[i], y[i]);
            avg_pred_error += fabs(err);
        }
        avg_pred_error /= N;

        printf("\tIter %3d: Training weights: %s\tAvg error: %.4f\n", iter, get_weights_str(ada),
               avg_pred_error);
    }

    if (iter < ADELINE_MAX_ITERATION) {
        printf("Converged after %d iterations.\n", iter);
    } else {
        printf("Did not converged after %d iterations.\n", iter);
    }
}
