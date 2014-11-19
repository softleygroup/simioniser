from __future__ import print_function # for python3-compatibility

from .readSimion import simion, accelerator
import numpy as np
from matplotlib import pyplot as plt

import ctypes
from ctypes import c_double, c_ulong, c_uint
c_double_p = ctypes.POINTER(c_double)


class EField2D(simion):
	def __init__(self, filename, voltages, scale, use_accelerator = False, prune_electrodes = False):
		super(EField2D, self).__init__(filename, voltages, prune_electrodes)
		
		self.dx = 1./scale
		self.dr = 1./scale
		
		self.nr = self.ny # treat y direction as r
		
		self.xmax = self.nx*self.dx
		self.rmax = self.nr*self.dr
		
		if use_accelerator:
			a = accelerator()
			a.set_npas(len(voltages))
			a.set_pasize(self.nx, self.nr, self.dx, self.dr)
			for n, p in enumerate(self.pas):
				a.add_pa(n, p.potential.ctypes.data_as(c_double_p), voltages[n])
			
			self.fastAdjustAll = lambda V: a.fastAdjustAll(V.ctypes.data_as(c_double_p))
			self.fastAdjust = lambda n, V: a.fastAdjust(n, V)
			
			def helper(xx, yy):
				dE = np.zeros((xx.shape[0], 2), dtype=np.double)
				a.getFieldGradient(xx.shape[0], xx.ctypes.data_as(c_double_p), yy.ctypes.data_as(c_double_p), dE.ctypes.data_as(c_double_p))
				return dE
			self.getFieldGradient = helper
			
			self.getField3 = a.getField3
			
			#del EField2D.getField 			# to prevent anyone from accidentally trying to call these
			#del EField2D.getPotential
		
	def getPotential(self, x, r):
		r = abs(r)
		
		ixf = x/self.dx - 1
		irf = r/self.dr - 1
		
		# Integer part of potential array index.
		ir = np.where(np.ceil(irf) < self.nr - 1, np.ceil(irf), self.nr-2).astype(np.int)
		ix = np.where(np.ceil(ixf) < self.nx - 1, np.ceil(ixf), self.nx-2).astype(np.int)
		
		ir[ir < 0] = 0
		ix[ix < 0] = 0
		
		# if isscalar(r) && isscalar(x)
		Q11 = super(EField2D, self).getPotential(ix,	 ir,	 0)
		Q12 = super(EField2D, self).getPotential(ix+1, ir,	 0)
		Q21 = super(EField2D, self).getPotential(ix,	 ir+1, 0)
		Q22 = super(EField2D, self).getPotential(ix+1, ir+1, 0)
		
		
		# Calculate distance of point from gridlines.
		r1 = (irf - np.floor(irf))
		r2 = 1-r1
		x1 = (ixf - np.floor(ixf))
		x2 = 1-x1
		
		# Linear interpolation function.
		return ((Q11*r2*x2) + (Q21*r1*x2) + (Q12*r2*x1) + (Q22*x1*r1))
		
	
	def getField3(self, pos):
		# GRADIENT Calculate the potential gradient at r,x.
		#	 The gradient is calculated from the centred-difference
		#	 approximation finite differences.
		
		r = np.sqrt(pos[:, 1]**2+pos[:, 2]**2)
		x = pos[:, 0]
		
		hr = self.dr/2.
		hx = self.dx/2.
		
		p1 = self.getPotential(x-hx, r)
		p2 = self.getPotential(x+hx, r)
		p3 = self.getPotential(x, r-hr)
		p4 = self.getPotential(x, r+hr);
		
		
		dfr = (p4-p3)/self.dr
		dfx = (p2-p1)/self.dx
		
		dfy = dfr*np.sin(np.arctan2(pos[:, 1], pos[:, 2]))
		dfz = dfr*np.cos(np.arctan2(pos[:, 1], pos[:, 2]))
		return np.array([dfx, dfy, dfz]).T
		
	
	def getField(self, x, r):
		# GRADIENT Calculate the potential gradient at r,x.
		#	 The gradient is calculated from the centred-difference
		#	 approximation finite differences.
		
		hx = self.dx/2.
		hr = self.dr/2.
		
		
		p1 = self.getPotential(x-hx, r)
		p2 = self.getPotential(x+hx, r)
		p3 = self.getPotential(x, r-hr)
		p4 = self.getPotential(x, r+hr);
		
		dfx = (p2-p1)/self.dx
		dfr = (p4-p3)/self.dr
		
		return np.array([dfx, dfr]).T
	
	def getFieldGradient(self, x, r):
		# based on following formula:
		# Fx: x-component of force
		# Ex: x-component of field
		# U: potential
		# Fx \propto dx |E| = dx sqrt(Ex^2+Ey^2+Ez^2)
		#	   = (Ex dx Ex + Ey dx Ey + Ez dx Ez)/|E|
		# now dx \vec(E) = (E(x+hx, y, z) - E(x-hx, y, z))/this.dx
		# in the code E(x+hx...) is called dx2, minus version dx1
		# other compnents equivalently
		hx = self.dx/2.
		hr = self.dr/2.
		
		E0 = self.getField(x, r)
		normE = np.sqrt(np.sum(E0**2, 1))
		
		# otherwise return is NaN
#		if normE == 0:
#			raise RuntimeError
		
		dx2 = self.getField(x+hx, r)
		dx1 = self.getField(x-hx, r)
		dEx = np.diag(E0.dot((dx2.T-dx1.T)/self.dx))/normE
		
		dy2 = self.getField(x, r+hr)
		dy1 = self.getField(x, r-hr)
		dEy = np.diag(E0.dot((dy2.T-dy1.T)/self.dr))/normE
		return np.array([dEx, dEy]).T


	def inArray3(self, pos):
		r = np.sqrt(pos[:,1]**2+pos[:,2]**2)
		x = pos[:,0]
		return self.inArray(x, r)
	
	def inArray(self, x, r):
		return (r >= 0) & (x > 0)# & (r < self.rmax) & (x < self.xmax)
		
		
	def isElectrode3(self, pos):
		# ISELECTRODE Test if point r, x is within an electrode.
		#	 Returns true if (r, x) is inside an electrode.
		
		r = np.sqrt(pos[:, 1]**2 + pos[:, 2]**2)
		x = pos[:, 0]
		
		return self.isElectrode(x, r)
	
	def isElectrode(self, x, r):
		assert r.shape == x.shape, 'r and x arrays are different sizes'
		
		# Fractional potential array index.
		irf = r/self.dr
		ixf = x/self.dx
		
		# Integer part of potential array index.
		ir = np.where(np.ceil(irf) < self.nr - 1, np.ceil(irf), self.nr-2).astype(np.int)
		ix = np.where(np.ceil(ixf) < self.nx - 1, np.ceil(ixf), self.nx-2).astype(np.int)
		
		ir[ir < 0] = 0
		ix[ix < 0] = 0
		
		return self.electrode_map[ix, ir].flatten()
		
		
	def plotPotential(self):
		# PLOTPOTENTIAL plot the potential in a new figure
		# Potential is plotted as a colour contour map, electrode
		# positions are indicated by white contours.
		r  = np.linspace(0, self.rmax, self.nr)
		x  = np.linspace(0, self.xmax, self.nx)
		ptot = np.zeros((self.nx, self.nr))
		for p in self.pas:
			ptot += p.voltage*p.potential[:, :, 0]
		plt.imshow(ptot.T, extent=[0, self.xmax, 0, self.rmax], aspect='auto', interpolation='none');
		plt.contour(x, r, self.electrode_map[:, :, 0].T, 1, colors='k', )
#		hold on
#		contour (r, x, this.elec, 1, 'w-', 'LineWidth', 2);
#		set (h, 'LineStyle', 'None');
		plt.xlabel('x');
		plt.ylabel('r');

