#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

#include <xmmintrin.h>

void set_npas(const unsigned int);
void add_pa(const unsigned int, const double *, const double);
void set_pasize(const int, const int, const int, const double, const double, const double, const double, const double, const double);

void free();

void getField3(const unsigned int, const double *, double *);
void fastAdjust(const unsigned int, const double);
void fastAdjustAll(const double *);

