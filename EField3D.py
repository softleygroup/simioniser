from __future__ import print_function # for python3-compatibility

from .readSimion import simion, accelerator
import numpy as np
from matplotlib import pyplot as plt

import ctypes
from ctypes import c_double, c_ulong, c_uint
c_double_p = ctypes.POINTER(c_double)

class EField3D(simion):
	def __init__(self, filename, voltages, scale, offset, use_accelerator=False, prune_electrodes=False):
		super(EField3D, self).__init__(filename, voltages, prune_electrodes)
		
		self.x0 = offset[0]
		self.y0 = offset[1]
		self.z0 = offset[2]
		
		self.dx = 1./scale
		self.dy = 1./scale
		self.dz = 1./scale
		
		self.xmax = self.nx*self.dx + self.x0;
		self.ymax = self.ny*self.dy + self.y0;
		self.zmax = self.nz*self.dz + self.z0;
		self.xmin = self.x0;
		self.ymin = self.y0;
		self.zmin = -self.zmax;
		
		if use_accelerator:
			a = accelerator(3)
			a.set_npas(len(voltages))
			a.set_pasize(self.nx, self.ny, self.nz, self.dx, self.dy, self.dz, self.x0, self.y0, self.z0)
			for n, p in enumerate(self.pas):
				a.add_pa(n, p.potential.ctypes.data_as(c_double_p), voltages[n])
			
			self.fastAdjustAll = lambda V: a.fastAdjustAll(V.ctypes.data_as(c_double_p))
			self.fastAdjust = lambda n, V: a.fastAdjust(n, V)
			self.getField3 = a.getField3
			
			#del EField3D.getPotential # to prevent anyone from accidentally trying to call these


	def getPotential(self, x, y, z):
		# POTENTIAL Get the magnitude of the electric potential.
		#	 Calculate the magnitude of the electrostatic potential at
		#	 coordinates x, y, z by interpolating the scaled fields from
		#	 each electrode. If r or x is outside the boundary, the
		#	 value at the boundary is returned.
		
		# Fractional potential array index.
		ixf = (x-self.x0)/self.dx - 1
		iyf = (y-self.y0)/self.dy - 1
		izf = abs(z-self.z0)/self.dz - 1
		
		# Integer part of potential array index.
		ix = np.where(np.ceil(ixf) < self.nx - 1, np.ceil(ixf), self.nx-2).astype(np.int)
		iy = np.where(np.ceil(iyf) < self.ny - 1, np.ceil(iyf), self.ny-2).astype(np.int)
		iz = np.where(np.ceil(izf) < self.nz - 1, np.ceil(izf), self.nz-2).astype(np.int)
		
		# Calculate distance of point from gridlines.
		#		 xd = (ixf - floor(ixf)).*this.dx;
		#		 yd = (iyf - floor(iyf)).*this.dy;
		#		 zd = (izf - floor(izf)).*this.dz;
		xd = (ixf - np.floor(ixf))
		yd = (iyf - np.floor(iyf))
		zd = (izf - np.floor(izf))
		
		Q111 = super(EField3D, self).getPotential(ix	, iy	, iz	)
		Q112 = super(EField3D, self).getPotential(ix	, iy	, iz+1)
		Q121 = super(EField3D, self).getPotential(ix	, iy+1, iz	)
		Q122 = super(EField3D, self).getPotential(ix	, iy+1, iz+1)
		Q211 = super(EField3D, self).getPotential(ix+1, iy	, iz	)
		Q212 = super(EField3D, self).getPotential(ix+1, iy	, iz+1)
		Q221 = super(EField3D, self).getPotential(ix+1, iy+1, iz	)
		Q222 = super(EField3D, self).getPotential(ix+1, iy+1, iz+1)
		
		i1 = (xd*Q211 + (1-xd)*Q111)
		i2 = (xd*Q221 + (1-xd)*Q121)
		j1 = (xd*Q212 + (1-xd)*Q112)
		j2 = (xd*Q222 + (1-xd)*Q122)
		
		k1 = (yd*i2 + (1-yd)*i1)
		k2 = (yd*j2 + (1-yd)*j1)
		
		return (zd*k2 + (1-zd)*k1)
		
	def getField3(self, pos):
		# GRADIENT Calculate the potential gradient at r,x.
		# The gradient is calculated from the central-difference
		# approximation finite differences.
		x = pos[:, 0]
		y = pos[:, 1]
		z = pos[:, 2]
		
		hx = self.dx/2.0
		hy = self.dy/2.0
		hz = self.dz/2.0
		
		px2 = self.getPotential(x+hx, y, z)
		px1 = self.getPotential(x-hx, y, z)
		py2 = self.getPotential(x, y+hy, z)
		py1 = self.getPotential(x, y-hy, z)
		pz2 = self.getPotential(x, y, z+hz)
		pz1 = self.getPotential(x, y, z-hz)
		
		dfx = (px2-px1)/self.dx
		dfy = (py2-py1)/self.dy
		dfz = (pz2-pz1)/self.dz
		return np.array([dfx, dfy, dfz]).T
		

	def inArray3(self, pos):
		x = pos[:,0]
		y = pos[:,1]
		z = pos[:,2]
		return (x > self.xmin) & (x < self.xmax) & (y > self.ymin) & (y < self.ymax) & (z > self.zmin) & (z < self.zmax)
		
	def fastAdjust(self, n, v):
		self.pas[n].voltage = v
		
	def isElectrode3(self, pos):
		# ISELECTRODE Test if point r, x is within an electrode.
		# Returns true if (r, x) is inside an electrode.
		
		x = pos[:, 0]
		y = pos[:, 1]
		z = pos[:, 2]
		assert y.shape == x.shape, 'r and x arrays are different sizes.'
		
		# Integer part of potential array index.
		ixf = (x-self.x0)/self.dx - 1
		iyf = (y-self.y0)/self.dy - 1
		izf = abs(z-self.z0)/self.dz - 1
		
		# Integer part of potential array index.
		ix = np.where(np.ceil(ixf) < self.nx - 1, np.ceil(ixf), self.nx-2).astype(np.int)
		iy = np.where(np.ceil(iyf) < self.ny - 1, np.ceil(iyf), self.ny-2).astype(np.int)
		iz = np.where(np.ceil(izf) < self.nz - 1, np.ceil(izf), self.nz-2).astype(np.int)
		
		
		ix[ix < 0] = 0
		iy[iy < 0] = 0
		iz[iz < 0] = 0
		
		return self.electrode_map[ix, iy, iz].flatten()
		
		
	def plotPotential(self, plane=2):
		# PLOTPOTENTIAL plot the potential in a new figure
		# Potential is plotted as a colour contour map, electrode
		# positions are indicated by white contours.
		if plane == 1:
			# here we plot a cut in the y=0-plane
			x  = np.linspace(self.xmin, self.xmax, self.nx)
			z  = np.linspace(self.zmin, self.zmax, self.nz)
			ptot = np.zeros((self.nx, self.nz))
			ind = int(-1.*self.ymin/self.dy)
			for p in self.pas:
				ptot += p.voltage*p.potential[:, 10, :]
			plt.imshow(ptot.T, extent=[self.xmin, self.xmax, self.zmin, self.zmax], aspect='equal', interpolation='none');
			plt.contour(x, z, self.electrode_map[:, 10, :].T, 1, colors='k', )

		else:
			# by default we plot a cut in the z=zmin-plane
			x  = np.linspace(self.xmin, self.xmax, self.nx)
			y  = np.linspace(self.ymin, self.ymax, self.ny)
			ptot = np.zeros((self.nx, self.ny))
			for p in self.pas:
				ptot += p.voltage*p.potential[:, :, 0]
			plt.imshow(ptot.T, extent=[self.xmin, self.xmax, self.ymin, self.ymax], aspect='equal', interpolation='none');
			plt.contour(x, y, self.electrode_map[:, :, 0].T, 1, colors='k', )
			plt.xlabel('x');
			plt.ylabel('r');

