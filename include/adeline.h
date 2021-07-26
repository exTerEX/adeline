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

#include <stddef.h>

typedef struct adeline {
    double learning_rate;

    size_t n_weights;
    double *weights;
} adeline_t;

adeline_t create_adaline(const size_t n_features, const double lr);

void delete_adaline(adeline_t *x);

char *get_weights_str(const adeline_t *ada);

size_t predict_adaline(adeline_t *ada, const double *x, double *out);

void fit_adaline(adeline_t *ada, double **X, const size_t *y, const size_t N);
