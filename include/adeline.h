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
