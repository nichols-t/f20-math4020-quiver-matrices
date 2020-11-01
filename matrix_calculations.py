# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import math
import networkx as nx
import sys

# This file contains source code for explorations of Ausland-Reiter transform matrices B for small quviers.
# Example quivers are defined first, with functions to perform the necessary calculations following.

# Define tolerance to round numbers to, may be used in np.around to round
# floating point errors to 0.
MAX_DEC = 10

############################################################
# EXAMPLES
# These are just a bunch of data structures that I made
# as example quivers to study. 
# Notice that each quiver is technically a dictionary, and
# contains keys for:
#   name (string) - a human-readable name for the quiver
#   vertices (list of int) - list of vertices; this is just
#                            integers 0 through |V|-1 where
#                            |V| is the number of vertices.
#   edges (list of int 2-tuples) - edges in the format
#                                  (tail, head).
############################################################

# SOME DYNKIN TYPE QUIVERS
a3 = {
    "vertices": range(3),
    "edges":[(0, 1), (1, 2)],
    "name": "A_3"
}

a4 = {
    "vertices": range(4),
    "edges": [(0, 1), (1, 2), (2, 3)],
    "name": "A_4"
}

a5 = {
    "vertices": range(5),
    "edges": [(0, 1), (1, 2), (2, 3), (3, 4)],
    "name": "A_5"
}

a6 = {
    "vertices": range(6),
    "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
    "name": "A_6"
}

d4 = {
    "vertices": range(4),
    "edges": [(0,1), (1, 2), (2, 4), (2, 3)],
    "name": "D_4"
}

d5 = {
    "vertices": range(5),
    "edges": [(0, 1), (1, 2), (2, 3), (2, 4)],
    "name": "D_5"
}

d6 = {
    "vertices": range(6),
    "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5)],
    "name": "D_6"
}

d7 = {
    "vertices": range(7),
    "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (4, 6)],
    "name": "D_7"
}

e6 = {
    "vertices": range(6),
    "edges": [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5)],
    "name": "E_6"
}

e7 = {
    "vertices": range(7),
    "edges": [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6)],
    "name": "E_7"
}

e8 = {
    "vertices": range(8),
    "edges": [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (6, 7)],
    "name": "E_8"
}

e9 = {
    "vertices": range(9),
    "edges": [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (6, 7), (7, 8)],
    "name": "E_9"
}

e10 ={
    "vertices": range(10),
    "edges": [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)],
    "name": "E_10"
}


e11 ={
    "vertices": range(11),
    "edges": [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)],
    "name": "E_11"
}

e12 ={
    "vertices": range(12),
    "edges": [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)],
    "name": "E_12"
}

e13 ={
    "vertices": range(13),
    "edges": [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12)],
    "name": "E_13"
}

# OTHER EXAMPLES
ex = {
    "name": "Ex",
    "vertices": range(15),
    "edges": [(0, 1), (2, 1), (1, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (10, 12), (10, 13), (10, 14)]
}

# Complete graph; this turns out to have rather boring eigenvals.
complete4 = {
    "name": "Complete Order 4",
    "vertices": range(4),
    "edges": [(0, 1), (0, 2), (0, 3), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2)]
}

# A cycle of length 4, but not an oriented cycle. This is A_3 Extended.
ac_4cycle = {
    "name": "Acyclic Weakly Connected 'Cycle'",
    "vertices": range(4),
    "edges": [(0, 1), (1, 2), (2, 3), (0, 3)]
}

# A "pendulum"-looking graph. 
ac_pendulum = {
    "name": "Acyclic Pendulum",
    "vertices": range(4),
    "edges": [(0, 1), (1, 2), (1, 3), (3, 2)]
}

ac_pendulum_2 = {
    "name": "Acyclic Pendulum",
    "vertices": range(5),
    "edges": [(0, 1), (1, 2), (2, 3), (2, 4), (3, 4)]
}

ac_pendulum_3 = {
    "name": "Acyclic Pendulum",
    "vertices": range(6),
    "edges": [(0, 1), (1, 2), (2, 3), (2, 4), (3, 4), (4, 5)]
}

ac_pendulum_4 = {
    "name": "Acyclic Pendulum",
    "vertices": range(7),
    "edges": [(0, 1), (1, 2), (2, 3), (2, 4), (3, 4), (4, 5), (3, 6)]
}

# Variant on A_4
double_a4 = {
    "name": "A4 with bidirectional edges",
    "vertices": range(4),
    "edges": [(0, 1), (1, 2), (2, 3), (3, 2), (2, 1), (1, 0)]
}

# I drew this one that looked a little bit like the letter "M"
m4 = {
    "name": "M-shape Acyclic",
    "vertices": range(4),
    "edges": [(0, 1), (0, 2), (2, 3), (1, 3)]
}

# "Pendulum" shape, but longer.
big_pendulum = {
    "name": "Acyclic Pendulum",
    "vertices": range(7),
    "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6)]
}

# "Pendulum"  shape, but the "bottom" is wider.
wide_pendulum = {
    "name": "Wide Pendulum",
    "vertices": range(6),
    "edges": [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5)]
}

# A_7 extended, being sure not to have oriented cycle.
a7_ext_ac = {
    "name": "A_7 Extended with Acyclic Orientation",
    "vertices": range(8),
    "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (0, 7)]
}

ac_pendulum_2_U_a2 = {
  "name": "Pendulum graphs union",
  "vertices": range(7),
  "edges": [(0, 1), (1, 2), (2, 3), (2, 4), (3, 4), (3, 5), (5, 6)]
}

ac_pendulum_2_U_d4 = {
  "name": "Pendulum graphs union",
  "vertices": range(8),
  "edges": [(0, 1), (1, 2), (2, 3), (2, 4), (3, 4), (3, 5), (5, 6), (5, 7)]
}

ac_pendulum_2_U_ac_pendulum = {
  "name": "Pendulum graphs union",
  "vertices": range(8),
  "edges": [(0, 1), (1, 2), (2, 3), (2, 4), (3, 4), (3, 5), (5, 6), (6, 7), (5, 7)]
}

tri = {
  "name": "Tri",
  "vertices": range(3),
  "edges": [(0, 1) , (1, 2), (0, 2)]
}

quad = {
  "name": "Quad",
  "vertices": range(4),
  "edges": [(0, 1) , (1, 2), (0, 2), (0, 4)]
}

############################################################
# FUNCTIONS
# These functions do a bunch of calculations. See each 
# function's documentation for specific details.
############################################################

def head(edge):
  """Returns the head of the edge given as a 2-tuple.

  Arguments:
    edge (2-tuple): an edge specified as (out, in)
  
  Returns:
    head (int) - the head of the edge.
  """
  return edge[1]

def tail(edge):
  """Returns the tail of the edge given as a 2-tuple.

  Arguments:
    edge (2-tuple): an edge specified as (out, in)
  
  Returns:
    tail (int) - the tail of the edge.
  """
  return edge[0]

def E(q):
  """Transforms a quiver object into its corresponding Euler matrix.

  Arguments: 
    q (quiver) = a quiver with vertices (list of int) and edges (list of 2-tuples).

  Returns: 
    euler_matrix (numpy matrix) - the Euler matrix of q.
  """

  # We're going to use a 2D python array at first.
  euler_array = [[0 for x in range(len(q["vertices"]))] for y in range(len(q["vertices"]))]
  for out_vertex in range(np.size(q["vertices"])):
    for in_vertex in range(np.size(q["vertices"])):
      # Set Kronecker symbol
      kronecker = 0
      if out_vertex == in_vertex:
        kronecker = 1

      # iterate through edges and find edges with appropriate head and tail.
      edges = [edge for edge in q["edges"] if head(edge) == in_vertex and tail(edge) == out_vertex]
      euler_array[out_vertex][in_vertex] = kronecker - len(edges)

  return np.matrix(euler_array)

def B(q):
  """Calculates the matrix B from the given quiver.

  Arguments:
    q (quiver) - the quiver to calculate the transformation matrix B from.
  
  Returns: 
    B (numpy matrix)
  """
  e = E(q)
  et = e.transpose()
  e_inv = np.linalg.inv(e)
  b = -1 * et * e_inv

  return b

# Calculate a string representing the characteristic polynomial of a matrix.
def tex_char_poly(A, var="\\lambda"):
  """Returns a LaTeX string for the characteristic polynomial of a given matrix
  A

  Arguments: 
    A (matrix) - the matrix to calculate the characteristic polynomial of.
    var (string, optional) - string to use as the variable name.

  Returns:
    poly_string (string) - the characteristic polynomial in LaTeX form.
  """

  coeffs = [int(np.around(x, MAX_DEC)) for x in np.poly(A)]
  N = len(coeffs)
  latex_string = ""

  for i in range(0, N):
    # Can just round this to integers, since we know it will be.
    c = coeffs[i]
    v_power = var + "^{" + str(N - i - 1) + "}"
    
    # Set the sign for th next term. We do this even if this term will
    # not be displayed.
    op = "+" if i != 0 else ""
    if (c < 0):
        op = "-"
    # Need to handle the constant term differently
    if (i != N - 1):
      if (c != 0):
        # Sign is handled, so take absolute value of c
        c = abs(c)
        # For non-constant terms, omit a coefficient of 1 or -1
        if (c == 1):
          c = ""
        if (i == N - 2):
          # If power is 1, don't show it.
          v_power = var
        # Determine if next character should have pos/neg connector.
        latex_string += op + " " + str(c) + v_power + " "
    else:
      if (c != 0):
        latex_string += op + " " + str(c)

  # I think if the constant term is 0 we'd have a trailing +; just in case,
  # remove it before returning.
  return latex_string.strip("+ -")

# Overlaying roots of unity over the eigenvalues is occasionally illuminating.
def nth_roots_unity (n):
  """Calculates the nth roots of unity.

  Arguments:
    n (int) - some positive integer to calculate the roots of unity of.
  Returns:
    roots (numpy array) - the nth roots of unity.
  """
  p = [1] + [0] * (n - 1) + [-1]
  roots = np.roots(p)
  return roots

def mahler_measure_from_eigens(eigens):
  """Calculates the mahler measure of a characteristic polynomial that has
  the given eigenvalues as roots.

  Arguments:
    eigens (numpy array) - 1D array containing the eigenvalues, which may be
    complex numbers.

  Returns:
    measure (real number) - the Mahler measure of the polynomial given by 
    (x - eigens[0])(x - eigens[1])...(x-eigens[np.size(eigens) - 1]).
  """

  mahler = np.prod([np.max([1, np.absolute(x)]) for x in np.around(eigens, MAX_DEC)])
  return mahler

def eigens_from_quiver(q):
  """Given a quiver q, returns the eigenvalues of A = (E^T)^-1 where E is the Euler matrix for the
  quiver.

  Arguments:
    q (dictionary) - a dictionary with keys vertices (list of int) and edges (list of 2-int tuples).

  Returns: 
    eigens (list of complex numbers) - the eigenvalues of the matrix A.
  """

  eigens = np.linalg.eigvals(B(q))

  return eigens 

def format_subplot(ax, eigens):
  """Given a matplotlib ax object, call some formatting methods.

  Arguments:
    ax (matplotlib.plot.figure.subplot) - the subplot object.
    eigens (numpy array) - the eigenvalues being plotted, used to set axis bounds.
  """

  # Create horizontal lines to denote the x + 0i and 0 + yi lines for clarity.
  ax.axhline(color='black', zorder=-100001)
  ax.axvline(color='black', zorder=-100000)
  # Set aspect ratio to 1 to avoid weirdness
  ax.set_aspect(1.0/ax.get_data_ratio())
  #Label axes
  ax.set_xlabel('Real Part')
  ax.set_ylabel('Imaginary Part')
  # Set axis ticks; notice that the stopping point is ceil(max).1 for both ranges.
  # Otherwise, it will not render the final tick properly.
  e_max_r = np.max([eigen.real for eigen in eigens])
  e_min_r = np.min([eigen.real for eigen in eigens])
  e_max_i = np.max([eigen.imag for eigen in eigens])
  e_min_i = np.min([eigen.imag for eigen in eigens])

  ax.yaxis.set_ticks(np.arange(math.floor(e_min_i), math.ceil(e_max_i) + 0.1, 0.5))
  ax.xaxis.set_ticks(np.arange(math.floor(e_min_r), math.ceil(e_max_r) + 0.1, 0.5))
  ax.yaxis.set_major_formatter(FormatStrFormatter('%si'))
  ax.yaxis.grid(which='both')
  ax.xaxis.grid(which='both')
  ax.grid(True)

def plot_eigenvals(eigens, ax):
  """ Given the eigenvalues and subplot, plot the eigenvalues on the subplot.

  Arguments:
    eigens (list of complex numbers) - the eigenvalues to plot.
    ax (matplotlib subplot) - subplot to graph eigenvals on.
  """

  X = np.around([x.real for x in eigens], MAX_DEC)
  Y = np.around([x.imag for x in eigens], MAX_DEC)
  ax.scatter(X,Y, color='#DC3220', zorder=-10)

  # Also plot the unit circle
  #plot unit circle
  t = np.linspace(0,2*math.pi,101)
  ax.plot(np.cos(t), np.sin(t), color='#005AB5')

def plot_quivers_eigenvals(quiver_list):
  """Plots the eigenvalues of A for every quiver in the list and draws the graph next to it.

  Arguments:
    quiver_list (list of quiver) - the quivers to plot.
  """
  fig, axs = plt.subplots(len(quiver_list), 1, figsize=(10, 5))
  for n in range(len(axs)):
    # Get the eigenvals
    q = quiver_list[n]
    eigens = eigens_from_quiver(q)
    eigen_ax = axs[n]
    format_subplot(eigen_ax, q)
    plot_eigenvals(eigens, eigen_ax)

  # Margin is a bit too tight, so give each plot some more room.
  plt.subplots_adjust(bottom=-2)
  plt.show()

def plot_quiver_eigenvals(quiver):
  """Plots the eigenvalues of the Auslander-Reiten transform matrix B for a single quiver.

  Arguments:
    quiver (quiver) - the quiver to calculate B for and plot eigenvalues of.
  """
  fig, ax = plt.subplots(figsize=(5, 5))
  eigens = eigens_from_quiver(quiver)
  format_subplot(ax, eigens)
  plot_eigenvals(eigens, ax)

  plt.show()

def to_pmatrix(mat):
  """Converts some matrix into a string that can be pasted into a LaTeX document.

  Returns:
    bmat (string) = a string of the form a11 & a12 & ... \\ a21 & a22 & ... \\
    for pasting into LaTeX.
  """
  bmat = ''
  np_mat = np.array(mat)
  for row in np_mat:
    for n in row:
      bmat += str(int(n))
      bmat += ' & '
    bmat += '\\\\ '
  return bmat

# This is sort of the "final" function that will display everything nicely. 
def get_nums(quiver,  var="\\lambda"):
  """Prints some information for the given quiver. Includes the Euler Matrix
  and the matrix B in a form suitable for pasting into a LaTeX doc, as well as
  the Mahler measure of the quiver and a plot of the eigenvalues of B.
  """

  # print(to_pmatrix(E(quiver)) + "\n")
  print(to_pmatrix(B(quiver)) + "\n")
  print(tex_char_poly(B(quiver), var) + "\n")
  print(eigens_from_quiver(quiver))

  print(mahler_measure_from_eigens(eigens_from_quiver(quiver)))
  plot_quiver_eigenvals(quiver)

# Use this function on whichever quiver you want to graph and get matrices for.
try:
  var = sys.argv[2]
except IndexError:
  var = "\\lambda"
get_nums(globals()[sys.argv[1]], var)