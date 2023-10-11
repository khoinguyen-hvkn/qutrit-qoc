import numpy as np

def ket(i):
  if(i == 0):
    return np.array([[1], [0], [0], [0]])
  if(i == 1):
    return np.array([[0], [1], [0], [0]])
  if(i == 2):
    return np.array([[0], [0], [1], [0]])
  if(i == 3):
    return np.array([[0], [0], [0], [1]])

def bra(i):
  if(i == 0):
    return np.array([[1, 0, 0, 0]])
  if(i == 1):
    return np.array([[0, 1, 0, 0]])
  if(i == 2):
    return np.array([[0, 0, 1, 0]])
  if(i == 3):
    return np.array([[0, 0, 0, 1]])

def Pi(i):
  return ket(i)*bra(i)

def sigma_x(i, j):
  return ket(j)@bra(i) + ket(i)@bra(j)

def sigma_y(i, j):
  return 1j*(ket(j)@bra(i) - ket(i)@bra(j))

def sigma_z(i, j):
  return ket(i)@bra(i) - ket(j)@bra(j)