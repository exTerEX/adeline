
struct adeline;

struct adaline create_adaline(const int n_features, const double lr);

void delete_adaline(struct adaline *x);

int initiate_adaline(double x);

char *get_weights_str(const struct adaline *ada);

int predict_adaline(struct adaline *ada, const double *x, double *out);

double fit_sample(struct adaline *ada, const double *x, const int y);

void fit_adaline(struct adaline *ada, double **X, const int *y, const int N);

