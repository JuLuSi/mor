from petsc4py import PETSc
import firedrake
from firedrake import TrialFunction, TestFunction, assemble, dx, inner, grad
import numpy as np
import scipy.sparse as sp


def petsc2sp(A):
    """Creates scipy sparse matrix/numpy array from a PETSc matrix/vector.
    
    :arg A: PETSc matrix/vector A
    :returns: Scipy sparse matrix/numpy array
    """
    if A.Type == PETSc.Vec.Type:
        return A.array.reshape(1, A.local_size)
    else:
        if A.getInfo()['nz_used'] is 0.0:
            return sp.csr_matrix(A.size, shape=A.local_size)
        else:
            Asp = A.getValuesCSR()[::-1]
            return sp.csr_matrix(Asp, shape=A.local_size)


class MORProjector(object):
    """Model Order Reduction Projector

    The main object to handle storage and projection using basis functions.

    :arg default_type: (optional) type for the assumed vector or matrix.
    """
    def __init__(self, default_type='petsc'):
        self.default_type = default_type
        self.snaps = []
        self.snap_mat = None
        self.n_basis = 0
        self.basis_mat = None

    def take_snapshot(self, u):
        """Store a snapshot

        :arg u: a :class:`firedrake.Function`
        """
        self.snaps.append(u.copy(deepcopy=True))

    def _snapshots_to_matrix(self):
        """Convert the snapshots to a matrix"""
        # DOFs x timesteps
        self.snap_mat = np.zeros((self.snaps[-1].function_space().dof_count, len(self.snaps)))

        for snap, i in zip(self.snaps, range(len(self.snaps))):
            with snap.dat.vec as vec:
                self.snap_mat[:, i] = vec.array

        return self.snap_mat.shape[1]

    def compute_basis(self, n_basis, inner_product="L2", time_scaling=False, delta_t=None):
        """

        :arg n_basis: Number of basis.
        :arg inner_product: Type of inner product (L2 or H1).
        :arg time_scaling: Use time scaling.
        :arg delta_t: :class:`numpy.ndarray` with used timesteps to scale.
        :return: Estimated error.
        """
        # Build inner product matrix
        V = self.snaps[-1].function_space()
        if inner_product == "L2":
            ip_form = inner(TrialFunction(V), TestFunction(V)) * dx
        elif inner_product == "H1":
            ip_form = inner(TrialFunction(V), TestFunction(V)) * dx + inner(grad(TrialFunction(V)),
                                                                            grad(TestFunction(V))) * dx

        ip_mat = assemble(ip_form, mat_type="aij").M.handle

        M = self._snapshots_to_matrix()

        # This matrix is symmetric positive semidefinite
        corr_mat = np.matmul(self.snap_mat.transpose(), petsc2sp(ip_mat).dot(self.snap_mat))

        # Build time scaling diagonal matrix
        if time_scaling is True and delta_t is not None:
            D = np.zeros((len(self.snaps), len(self.snaps)))
            for i in range(len(self.snaps)):
                D[i, i] = np.sqrt(delta_t[i])
            D[0, 0] = np.sqrt(delta_t[0] / 2.0)
            D[-1, -1] = np.sqrt(delta_t[-1] / 2.0)

            # D'MD
            corr_mat = np.matmul(D.transpose(), np.matmul(corr_mat, D))

        self.n_basis = n_basis

        # Compute eigenvalues, all real and non negative
        w, v = np.linalg.eigh(corr_mat)

        idx = np.argsort(w)[::-1]
        w = w[idx]
        v = v[:, idx]

        # Skip negative entries
        idx_neg = np.argwhere(w < 0)
        if len(idx_neg) > 0:
            # Reduce number of basis to min(n_basis, first_negative_eigenvalue)
            n_basis = np.minimum(n_basis, idx_neg[0][0])

        psi_mat = np.zeros((self.snaps[-1].function_space().dof_count, n_basis))

        for i in range(n_basis):
            psi_mat[:, i] = self.snap_mat.dot(v[:, i]) / np.sqrt(w[i])

        ratio = np.sum(w[:n_basis]) / np.sum(w[:M])

        self.basis_mat = PETSc.Mat().create(PETSc.COMM_WORLD)
        self.basis_mat.setType('dense')
        self.basis_mat.setSizes([self.snap_mat.shape[0], n_basis])
        self.basis_mat.setUp()

        self.basis_mat.setValues(range(self.snap_mat.shape[0]),
                                 range(n_basis),
                                 psi_mat.reshape((self.snap_mat.shape[0], n_basis)))

        self.basis_mat.assemble()

        return ratio

    def get_basis_mat(self):
        """Return the basis mat"""
        assert self.basis_mat is not None
        return self.basis_mat

    def project_operator(self, oper, oper_type='petsc'):
        """Project an operator.

        :param oper: :class:`firedrake.matrix.Matrix`.
        :param oper_type: Type of returned operator.
        :return: Projected operator.
        """
        if type(oper) is firedrake.matrix.Matrix:
            A = oper.M.handle
        else:
            A = oper

        Ap = A.PtAP(self.basis_mat)

        if oper_type == 'petsc':
            return Ap
        elif oper_type == 'scipy':
            return petsc2sp(Ap).todense()

    def project_function(self, f, func_type='petsc'):
        """Project a function from full space to reduced space.

        :param f: :class:`firedrake.Function`.
        :param func_type: Type of returned function.
        :return: Projected function.
        """
        fp, _ = self.basis_mat.createVecs()
        if type(f) is firedrake.Function:
            with f.dat.vec as vec:
                self.basis_mat.multTranspose(vec, fp)
        else:
            self.basis_mat.multTranspose(f, fp)

        if func_type == 'petsc':
            return fp
        elif func_type == 'scipy':
            return fp.array

    def recover_function(self, fp, func_type='petsc'):
        """Recover project function from reduced space to full space.

        :param fp: Function of type func_type.
        :param func_type: Type of the input function.
        :return: Recovered function.
        """
        if func_type == 'scipy':
            fp_petsc, _ = self.basis_mat.createVecs()
            fp_petsc.setValues(range(fp.shape[0]), fp)

        f = firedrake.Function(self.snaps[-1].function_space())
        with f.dat.vec as vec:
            if func_type == 'scipy':
                self.basis_mat.mult(fp_petsc, vec)
            else:
                self.basis_mat.mult(fp, vec)

        return f
