#include "accelerator2D.h"

static double *restrict voltage;

static unsigned int npas;
static int nx, nr;
static double dx, dr, hx, hr;

typedef struct pa pa;

struct pa {
	double voltage;
	double *restrict potential;
};

static pa *restrict pas;

void set_npas(unsigned int n)
{
	npas = n;
	pas = malloc(npas*sizeof(pa));
}

void set_pasize(const int nx_l, const int nr_l, const double dx_l, const double dr_l)
{
	nx = nx_l;
	nr = nr_l;
	dx = dx_l;
	dr = dr_l;
	hx = dx/2.0;
	hr = dr/2.0;
}

void add_pa(const unsigned int n, const double * potential, const double voltage)
{
	pas[n].voltage = voltage;
	pas[n].potential = __builtin_assume_aligned(potential, 16);
}

void fastAdjustAll(const double * voltage)
{
	for (unsigned int n = 0; n < npas; n++)
	{
		pas[n].voltage = voltage[n];
	}
}

void fastAdjust(const unsigned int n, const double v)
{
	printf("adjusting voltage\n");
	pas[n].voltage = v;
}

static inline double getPotential(const double x, const double r)
{
	double Q11 = 0, Q12 = 0, Q21 = 0, Q22 = 0;
	double r1, r2, x1, x2;
	
	double ixf = x/dx - 1;
	double irf = r/dr - 1;
	
	int ix = ceil(ixf); 
	int ir = ceil(irf);
	
	// Integer part of potential array index.
	ix = ix >= nx - 1 ? nx - 2 : ix;
	ir = ir >= nr - 1 ? nr - 2 : ir;
	
	ix = ix < 0 ? 0 : ix;
	ir = ir < 0 ? 0 : ir;
	
	
	for (unsigned int n = 0; n < npas; n++)
	{
		Q11 += pas[n].voltage*pas[n].potential[ix*nr + ir];
		Q12 += pas[n].voltage*pas[n].potential[(ix + 1)*nr + ir];
		Q21 += pas[n].voltage*pas[n].potential[ix*nr + ir + 1];
		Q22 += pas[n].voltage*pas[n].potential[(ix + 1)*nr + ir + 1];
	}
	
	// Calculate distance of point from gridlines.
	r1 = (irf - floor(irf));
	r2 = 1 - r1;
	x1 = (ixf - floor(ixf));
	x2 = 1 - x1;
	
	// Linear interpolation function.
	return ((Q11*r2*x2) + (Q21*r1*x2) + (Q12*r2*x1) + (Q22*x1*r1));
}

static inline void getSingleField(const double x, const double r, double * result)
{
	static double p1, p2, p3, p4;
	p1 = getPotential(x-hx, r);
	p2 = getPotential(x+hx, r);
	p3 = getPotential(x, r-hr);
	p4 = getPotential(x, r+hr);
	
	result[0] = (p2-p1)/dx;
	result[1] = (p4-p3)/dr;
}

static inline void getSingleField3(const double * pos, double * result)
{
	const double x = pos[0];
	const double r = sqrt(pos[1]*pos[1] + pos[2]*pos[2]);
	
	static double p1, p2, p3, p4, dfr;
	
	p1 = getPotential(x-hx, r);
	p2 = getPotential(x+hx, r);
	p3 = getPotential(x, r-hr);
	p4 = getPotential(x, r+hr);
	
	
	dfr = (p4-p3)/dr;
	
	result[0] = (p2-p1)/dx;
	result[1] = dfr*sin(atan2(pos[1], pos[2]));
	result[2] = dfr*cos(atan2(pos[1], pos[2]));
	
}

static void getSingleFieldGradient(const double x, const double r, double * result)
{
	double E0[2];
	double dx1[2], dx2[2], dr1[2], dr2[2];
	
	getSingleField(x, r, &E0[0]);
	double normE = sqrt(E0[0]*E0[0] + E0[1]*E0[1]);
	
	if (normE == 0)
	{
		result[0] = 0;
		result[1] = 0;
		return;
	}
	
	getSingleField(x+hx, r, &dx2[0]);
	getSingleField(x-hx, r, &dx1[0]);
	getSingleField(x, r+hr, &dr2[0]);
	getSingleField(x, r-hr, &dr1[0]);
	
	result[0] = (E0[0]*(dx2[0] - dx1[0]) + E0[1]*(dx2[1] - dx1[1]))/normE/dx;
	result[1] = (E0[0]*(dr2[0] - dr1[0]) + E0[1]*(dr2[1] - dr1[1]))/normE/dr;
}

void getFieldGradient(const unsigned int nParticles, const double * x, const double * r, double * result)
{
	// proper way to access potentials is ix*ny + iy
	for (unsigned int n = 0; n < nParticles; n++)
	{
		// get field gradient for individual particles
		getSingleFieldGradient(x[n], r[n], &result[2*n]);
	}
}

void getField3(const unsigned int nParticles, const double * pos, double * result)
{
	for (unsigned int n = 0; n < nParticles; n++)
	{
		getSingleField3(&pos[3*n], &result[3*n]);
	}
}

void cleanup()
{
	free(pas);
}
