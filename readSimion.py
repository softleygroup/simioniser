from __future__ import print_function # for python3-compatibility

import numpy as np
import os

import ctypes
from ctypes import c_double, c_ulong, c_uint
c_double_p = ctypes.POINTER(c_double)


class simion(object):
	def __init__(self, filename, voltages, prune_electrodes, verbose=False):
		self.verbose=verbose
		directory, fname = os.path.split(filename)
		files = sorted([x for x in os.listdir(directory) if x.startswith(fname) and x.endswith('patxt')])
		
		assert len(files) == len(voltages), 'Incorrect number of potentials specified!'
		
		self.pas = [0]*len(files)
		for i, f in enumerate(files):
			self.pas[i] = patxt(os.path.join(directory, f), voltages[i])
			
		# these are the same for all pas, so they go in here
		self.nx = int(self.pas[0].parameters['nx'])
		self.ny = int(self.pas[0].parameters['ny'])
		self.nz = int(self.pas[0].parameters['nz'])
		
		# make a global electrode map
		self.electrode_map = np.zeros((self.nx, self.ny, self.nz), dtype = np.bool)
		for p in self.pas:
			self.electrode_map |= p.isElectrode
		# or-ing these is unnecessary, as each patxt contains the full electrode set
		# self.electrode_map = self.pas[0].isElectrode[:]
		
		# remove all meshes from electrode map, as we don't want to collide on those anyways...
		# this is stupidly slow, though. there must be a better way of doing this
		if not prune_electrodes:
			return
		print('pruning meshes from electrode map. sorry, this takes a moment...')
		ind = np.where(self.electrode_map[:, :, :])
		em = self.electrode_map
		ns = [self.nx, self.ny, self.nz]
		for ax in range(3):
			if ns[ax] > 1:
				parity = np.zeros_like(ind[0], dtype=np.bool)
				dx = int(ax == 0)
				dy = int(ax == 1)
				dz = int(ax == 2)
				for k, i in enumerate(zip(*ind)):
					if i[ax] > 0 and i[ax] < ns[ax] - 1:
						parity[k] = em[i[0] - dx, i[1] - dy, i[2] - dz] | em[i[0] + dx, i[1] + dy, i[2] + dz]
					elif i[ax] > 0:
						parity[k] = em[i[0] - dx, i[1] - dy, i[2] - dz]
					elif i[ax] < ns[ax] - 1:
						parity[k] = em[i[0] + dx, i[1] + dy, i[2] + dz]
				ind2 = np.where(~parity)[0]
				em[ind[0][ind2], ind[1][ind2], ind[2][ind2]] = 0
		
	def fastAdjust(self, n, v):
		self.pas[n].setVoltage(v)
		
	def fastAdjustAll(self, potentials):
		assert len(potentials) == len (self.pas), 'Number of Voltages passed to fastAdjustAll must equal number of electrodes in potential array'
		for i, p in enumerate(potentials):
			self.pas[i].setVoltage(p)
		
	def getPotential(self, ix, iy, iz):
		value = 0
		for p in self.pas:
			value += p.getPotential(ix, iy, iz)
		return value
		
		
class accelerator(object):
	def __init__(self, ndim=2, verbose=False):
		self.verbose=verbose
		localdir = os.path.dirname(os.path.realpath(__file__)) + '/'
		
		if ndim == 3:
			target = 'accelerator3D'
		else:
			target = 'accelerator2D'
		if not os.path.exists(localdir + target + '.so') or os.stat(localdir + target + '.c').st_mtime > os.stat(localdir + target + '.so').st_mtime: # we need to recompile
			from subprocess import call
			
			COMPILE = ['PROF'] # 'PROF', 'FAST', both or neither
			# include branch prediction generation. compile final version with only -fprofile-use
			commonopts = ['-c', '-fPIC', '-Ofast', '-march=native', '-std=c99', '-fno-exceptions', '-fomit-frame-pointer']
			profcommand = ['gcc', '-fprofile-arcs', '-fprofile-generate', target + '.c']
			profcommand[1:1] = commonopts
			fastcommand = ['gcc', '-fprofile-use', target + '.c']
			fastcommand[1:1] = commonopts
			
			print()
			print()
			print('===================================')
			print('compilation target: ', target)
			if 'PROF' in COMPILE:
				if call(profcommand, cwd = localdir) != 0:
					print('COMPILATION FAILED!')
					raise RuntimeError
				call(['gcc', '-shared', '-fprofile-generate', target + '.o', '-o', target + '.so'], cwd = localdir)
				print('COMPILATION: PROFILING RUN')
			if 'FAST' in COMPILE:
				call(fastcommand, cwd=localdir)
				call(['gcc', '-shared', target + '.o', '-o', target + '.so'], cwd = localdir)
				print('COMPILATION: FAST RUN')
			if not ('PROF' in COMPILE or 'FAST' in COMPILE):
				print('DID NOT RECOMPILE C SOURCE')
			print('===================================')
			print()
			print()
		elif self.verbose:
			print('library up to date, not recompiling accelerator')
		
		
		self.acc = ctypes.cdll.LoadLibrary(localdir + target + '.so')
		
		self.acc.set_npas.argtypes = [c_uint]
		self.acc.set_npas.restype = None
		self.acc.add_pa.argtypes = [c_uint, c_double_p, c_double]
		self.acc.add_pa.restype = None
		if ndim == 3:
			self.acc.set_pasize.argtypes = [c_uint, c_uint, c_uint, c_double, c_double, c_double, c_double, c_double, c_double]
		else:
			self.acc.set_pasize.argtypes = [c_uint, c_uint, c_double, c_double]
		self.acc.set_pasize.restype = None
		
		if ndim == 2:
			self.acc.getFieldGradient.argtypes = [c_uint, c_double_p, c_double_p, c_double_p]
			self.acc.getFieldGradient.restype = None
			self.getFieldGradient = self.acc.getFieldGradient
		else:
			self.getFieldGradient = None
		self.acc.fastAdjustAll.argtypes = [c_double_p]
		self.acc.fastAdjustAll.restype = None
		
		self.acc.fastAdjust.argtypes = [c_uint, c_double]
		self.acc.fastAdjustAll.restype = None
		
					
		def helper(pos):
			res = np.zeros_like(pos, dtype=np.double)
			self.acc.getField3(pos.shape[0], pos.ctypes.data_as(c_double_p), res.ctypes.data_as(c_double_p))
			return res
		self.getField3 = helper

		
		self.set_npas = self.acc.set_npas
		self.add_pa = self.acc.add_pa
		self.set_pasize = self.acc.set_pasize
			
		self.fastAdjustAll = self.acc.fastAdjustAll
		self.fastAdjust = self.acc.fastAdjust
		
		
class patxt(object):
	def __init__(self, filename, V, verbose=False):
		self.verbose=verbose
		self.voltage = V
		self.filename = filename
		self.parameters = {}
		# read header data
		with open(filename, 'r') as ifile:
			# read header, save parameters in dict
			for l in range(18):
				line = ifile.readline()
				if line.startswith(' '):
					self.parameters[line.split()[0]] = line.split()[1]
			
		# make some parameters accessible by shorthand
		nx = int(self.parameters['nx'])
		ny = int(self.parameters['ny'])
		nz = int(self.parameters['nz'])
		
		# now read the actual data
		if os.path.isfile(filename + '.npy'): # TODO: check date, not just existence
			if self.verbose:
				print('loading ', filename, 'from cache')
			data = np.load(filename + '.npy')
		else:
			data = np.genfromtxt(filename, delimiter = ' ', skip_header = 18, skip_footer = 2, usecols = (3, 4))
			np.save(filename, data)
		
		assert data.shape[0] == nx*ny*nz
		
		self.isElectrode = data[:, 0].reshape((nx, ny, nz), order='F').astype(np.bool)
		self.potential = data[:, 1].reshape((nx, ny, nz), order='F').astype(np.double)
		self.potential = np.ascontiguousarray(self.potential)
		self.potential /= 10000.0
		
	def setVoltage(self, V):
		self.voltage = V
	
	def getPotential(self, ix, iy, iz):
		return self.voltage*self.potential[ix, iy, iz]