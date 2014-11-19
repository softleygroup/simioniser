#include "accelerator3D.h"

static double *restrict voltage;

static unsigned int npas;
static int nx, ny, nz;
static double dx, dy, dz, hx, hy, hz, x0, y0, z0;

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

void set_pasize(const int nx_l, const int ny_l, const int nz_l, const double dx_l, const double dy_l, const double dz_l, const double x0_l, const double y0_l, const double z0_l)
{
	nx = nx_l;
	ny = ny_l;
	nz = nz_l;
	dx = dx_l;
	dy = dy_l;
	dz = dz_l;
	hx = dx/2.0;
	hy = dy/2.0;
	hz = dz/2.0;
	x0 = x0_l;
	y0 = y0_l;
	z0 = z0_l;
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
	pas[n].voltage = v;
}

static inline double getPotential(const double x, const double y, const double z)
{
	double Q111 = 0, Q112 = 0, Q121 = 0, Q122 = 0;
	double Q211 = 0, Q212 = 0, Q221 = 0, Q222 = 0;
	
	double ixf = (x-x0)/dx - 1;
	double iyf = (y-y0)/dy - 1;
	double izf = abs(z-z0)/dz - 1;
	
	int ix = ceil(ixf);
	int iy = ceil(iyf);
	int iz = ceil(izf);
	
	ix = ix >= nx - 1 ? nx - 2 : ix;
	iy = iy >= ny - 1 ? ny - 2 : iy;
	iz = iz >= nz - 1 ? nz - 2 : iz;
	
	ix = ix < 0 ? 0 : ix;
	iy = iy < 0 ? 0 : iy;
	iz = iz < 0 ? 0 : iz;
	
	double xd = (ixf - floor(ixf));
	double yd = (iyf - floor(iyf));
	double zd = (izf - floor(izf));	
	
	for (unsigned int n = 0; n < npas; n++)
	{
		Q111 += pas[n].voltage*pas[n].potential[ix*ny*nz + iy*nz + iz];
		Q112 += pas[n].voltage*pas[n].potential[ix*ny*nz + iy*nz + iz + 1];
		Q121 += pas[n].voltage*pas[n].potential[ix*ny*nz + (iy + 1)*nz + iz];
		Q122 += pas[n].voltage*pas[n].potential[ix*ny*nz + (iy + 1)*nz + iz + 1];
		Q211 += pas[n].voltage*pas[n].potential[(ix + 1)*ny*nz + iy*nz + iz];
		Q212 += pas[n].voltage*pas[n].potential[(ix + 1)*ny*nz + iy*nz + iz] + 1;
		Q221 += pas[n].voltage*pas[n].potential[(ix + 1)*ny*nz + (iy + 1)*nz + iz];
		Q222 += pas[n].voltage*pas[n].potential[(ix + 1)*ny*nz + (iy + 1)*nz + iz + 1];
	}
	
	
	
	double i1 = (xd*Q211 + (1-xd)*Q111);
	double i2 = (xd*Q221 + (1-xd)*Q121);
	double j1 = (xd*Q212 + (1-xd)*Q112);
	double j2 = (xd*Q222 + (1-xd)*Q122);
		
	double k1 = (yd*i2 + (1-yd)*i1);
	double k2 = (yd*j2 + (1-yd)*j1);
	
	return (zd*k2 + (1-zd)*k1);
}


static inline void getSingleField3(const double * pos, double * result)
{
	const double x = pos[0];
	const double y = pos[1];
	const double z = pos[2];
	
	static double px2, px1, py2, py1, pz2, pz1;
	
	px2 = getPotential(x+hx, y, z);
	px1 = getPotential(x-hx, y, z);
	py2 = getPotential(x, y+hy, z);
	py1 = getPotential(x, y-hy, z);
	pz2 = getPotential(x, y, z+hz);
	pz1 = getPotential(x, y, z-hz);
	
	result[0] = (px2-px1)/dx;
	result[1] = (py2-py1)/dy;
	result[2] = (pz2-pz1)/dz;
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
