import numpy as np
import scipy.optimize
import warnings
import quaternion
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class BinghamSampler:
    """
    Sample points on a Bingham distribution using rejection sampling.
    """

    def __init__(self, dim=3):
        self.dim = dim
        self.q = dim + 1

    def __call__(self, As, sampling_N):
        As = As[np.newaxis, :, :] if As.ndim == 2 else As
        results = [self._sample_bingham(A, sampling_N) for A in As]
        return np.array(results)

    def _sample_bingham(self, A, sampling_N=10000):
        # Eigen decomposition for A, ensure A is PSD
        Lambda, eig_vecs = np.linalg.eigh(-A)  # No need to re-negate A later
        Lambda -= np.min(Lambda)  # Shift eigenvalues to make them non-negative
        A = eig_vecs @ np.diag(Lambda) @ eig_vecs.T

        b = self._optimize_b(Lambda)
        Omega = np.eye(self.q) + 2 * A / b
        invsqrt_Omega = np.linalg.cholesky(np.linalg.inv(Omega))

        M_star = np.exp(-(self.q - b) / 2.0) * (self.q / b) ** (self.q / 2)

        result = np.empty((0, self.q))
        while result.shape[0] < sampling_N:
            X = self._sampling_from_ACG(invsqrt_Omega, 2 * (sampling_N - result.shape[0]))
            ratio = np.exp(-np.einsum('bi,ij,bj->b', X, A, X)) / (np.einsum('bi,ij,bj->b', X, Omega, X) ** (-self.q / 2))
            uniforms = np.random.rand(X.shape[0])
            accepted = X[uniforms < ratio / M_star]
            result = np.vstack((result, accepted))
        return result[:sampling_N]

    def _optimize_b(self, Lambda):
        # Optimize b using the condition sum(1 / (b + 2*lambda_i)) = 1
        func = lambda b: (np.sum(1 / (b + 2 * Lambda)) - 1) ** 2
        dfunc_db = lambda b: -2 * np.sum(1 / (b + 2 * Lambda) ** 2) * (np.sum(1 / (b + 2 * Lambda)) - 1)

        # Use scipy's optimize function for root finding
        b_opt = scipy.optimize.newton(func, 1.0, fprime=dfunc_db, tol=1e-6)
        return b_opt

    def _sampling_from_ACG(self, invsqrt_Omega, N):
        y = invsqrt_Omega @ np.random.randn(self.q, N)
        return (y / np.linalg.norm(y, axis=0)).T

    def genA(self, variance=3):
        # Generate a random symmetric matrix A
        A = 1 - 2 * np.random.randn(self.q, self.q) * variance
        return A.T @ A


class BinghamDistribution:
    """Simplified Bingham distribution in SO(3) using symmetric matrix A."""

    def __init__(self, A):
        """Constructor with a symmetric matrix A.

        Args:
            A (numpy.array): Symmetric matrix (4x4).
        """
        if not np.allclose(A, A.T):
            warnings.warn("Input matrix A is not symmetric. It will be symmetrized.", UserWarning)
            A = 0.5 * (A + A.T)
        self.A = A
        self.Z, self.M = np.linalg.eigh(self.A)
        self.sampler = BinghamSampler(dim=3)
        self.sample_buf = None

    def density(self, q):
        """Calculate the density of a quaternion under the Bingham distribution."""
        self.check_normalize(q)
        q = np.array([q.w, q.x, q.y, q.z])
        return np.exp(q.T @ self.A @ q)

    def mode(self):
        """Find the mode of the Bingham distribution."""
        eig_val, eig_vec = np.linalg.eigh(self.A)
        return eig_vec[:, np.argmin(eig_val)]

    def check_normalize(self, q):
        """Check if a quaternion is normalized, and normalize if it is not."""
        if not np.isclose(quaternion.norm(q), 1):
            q = q / quaternion.norm(q)
        return q

    def update_sample(self, N_sample=None):
        """Update quaternion samples on Bingham distribution."""
        if N_sample is None:
            self.sample_buf = quaternion.as_quat_array(self.sampler(self.A, 1)[0][0])
        else:
            self.sample_buf = quaternion.as_quat_array(self.sampler(self.A, N_sample)[0])
        return self.sample_buf

    def sample(self, N_sample=None):
        """Return quaternion samples on Bingham distribution."""
        if self.sample_buf is None or (N_sample is not None and self.sample_buf.shape[0] != N_sample):
            self.update_sample(N_sample)
        return self.sample_buf

    def __repr__(self):
        """Representation of BinghamDistribution."""
        A_str = "\n".join(["[" + ", ".join(f"{item:.2e}" for item in row) + "]" for row in self.A])
        return f"BinghamDistribution(A=\n{A_str}\n)"
        
        
def draw_bingham(bdistr, quat_gt, num_samples=500, probability=0.7, ax=None, elev=20, azim=80):
    """Draw samples from a Bingham distribution
    and return an image object.

    Args:
        bdistr (BinghamDistribution): Object of BinghamDistribution
        quat_gt (numpy.array): Ground truth rotation
        num_samples (int): The number of samples to draw.
        probability (float): Probability of Bingham distribution

    Returns:
        numpy.ndarray: Numpy array of a IMAGE_WIDTH x IMAGE_HEIGHT GRB image.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    draw_bingham_distribution(ax, bdistr, quat_gt, num_samples, probability, elev, azim)


def draw_bingham_distribution(ax, bdistr, quat_gt, num_samples=500, probability=0.7, elev=20, azim=80):
    """Draw samples from a Bingham distribution

    Args:
        ax (axes): Object of axes
        bdistr (BinghamDistribution): Object of BinghamDistribution
        quat_gt (numpy.array): Ground truth rotation
        num_samples (int): The number of samples to draw.
        probability (float): Probability of Bingham distribution
    """
    M, zs = bdistr.M, bdistr.Z
    if (M is not None) and (zs is not None):
        quaternions = bdistr.sample(num_samples)
        rotations = np.zeros([num_samples, 3, 3])

        # Convert quaternion to rotation matrix.
        for idx, quat in enumerate(quaternions):
            rotation = quaternion.as_rotation_matrix(quat)
            rotations[idx, :, :] = rotation

        axes = quaternion.as_rotation_matrix(quaternion.as_quat_array(np.array(M[:, -1])))
    else:
        rotations = None
        axes = None

    if quat_gt is not None:
        axes_gt = quaternion.as_rotation_matrix(quaternion.as_quat_array(quat_gt))
    else:
        axes_gt = None
    draw_so3s(ax, rotations, axes, axes_gt, elev, azim)


def make_sphere():
    """
    Make a unit sphere pointcloud.

    Returns:
        tuple (numpy.array, numpy.array, numpy.array):
        a tuple of three arrays where
        each array contains x, y, z components of the sphere surface
        positions.
    """
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def draw_so3s(ax, rotations, axes=None, axes_gt=None, elev=20, azim=80):
    """Draw 3D points for the tips of x, y, z axes over the given rotations.
    Optionally, a coordinate system can be drew by providing @p axes.

    Args:
        ax (axes): Object of axes
        rotations (numpy.array):
            An np.ndarray whose shape is (N, 3, 3)
            where N denotes the number of
            rotation matrices, which are stored in the last two dimensions.
        axes (numpy.array):
            An orthonormal np.ndarray whose shape is (3, 3) and each column
            represents x, y, z axis of a 3D coordinate system respectively.
        axes_gt (numpy.array):
            Ground truth rotation, as same type as axes
    """
    point_of_view = np.array([0, 0, -1])
    def arrow3d(ax, R, along="z", length=1.0, width=0.03, head=0.33, headwidth=4, **kw):
        # TODO: need to review OR rewrite
        arrow_pts = [
            [0, 0],
            [width, 0],
            [width, (1 - head) * length],
            [headwidth * width, (1 - head) * length],
            [0, length],
        ]
        arrow_pts = np.array(arrow_pts)

        r, theta = np.meshgrid(arrow_pts[:, 0], np.linspace(0, 2 * np.pi, 30))
        z = np.tile(arrow_pts[:, 1], r.shape[0]).reshape(r.shape)
        x = r * np.sin(theta)
        y = r * np.cos(theta)

        if along == "x":
            R_swap = np.eye(3)[[2, 1, 0]]
        if along == "y":
            R_swap = np.eye(3)[[0, 2, 1]]
        if along == "z":
            R_swap = np.eye(3)[[0, 1, 2]]

        b1 = np.dot(R_swap, np.c_[x.flatten(), y.flatten(), z.flatten()].T)
        b2 = np.dot(R, b1).T
        x = b2[:, 0].reshape(r.shape)
        y = b2[:, 1].reshape(r.shape)
        z = b2[:, 2].reshape(r.shape)
        ax.plot_surface(x, y, z, **kw)

    def draw_axes(axes, **kw):
        """
        The axes should be drawn from the back to the front,
        i.e., in the order of how far the tip of the axis is
        from the viewpoint (1,-1,1) \in R^3.
        """
        dist_from_axistip_to_viewpoint = np.linalg.norm(axes - point_of_view.reshape(-1, 1), axis=0)

        # The farthest should comes first, the nearest last.
        draw_order = np.argsort(-dist_from_axistip_to_viewpoint)
        for i in draw_order:
            arrow3d(ax, axes, along=["x", "y", "z"][i], color=["red", "green", "blue"][i], **kw)

    ax.grid(False)
    ax.set_box_aspect((1, 1, 1))
    
    # red, green, blue
    colors = ["red", "green", "blue"]
    lighter_colors = ["#ff8080", "#80c080", "#8080ff"]

    if rotations is not None:
        for i, color in enumerate(colors):
            axis_vectors = rotations[:, :, i]
            is_back = (axis_vectors @ np.array([0, 0, -1])) < 0  # View from back
            ax.scatter(*axis_vectors[is_back].T, color=lighter_colors[i], alpha=0.5, marker=".")
            ax.scatter(*axis_vectors[~is_back].T, color=color, alpha=0.5, marker=".", label=f"{color}-axis")
    
    # Draw the sphere
    x, y, z = make_sphere()
    ax.plot_wireframe(x, y, z, rstride=5, cstride=5, color="gray", linewidth=0.2)

    def draw_axes(ax, axes, alpha=1.0, **kwargs):
        # Determine the order to draw the axis arrows based on depth
        order = np.argsort([np.linalg.norm(ax - np.array([0, 0, -1])) for ax in axes.T])
        for i in order:
            arrow3d(ax, axes, along=["x", "y", "z"][i], color=colors[i], alpha=alpha, **kwargs)

    if axes is not None:
        draw_axes(ax, axes)
        
    if axes_gt is not None:
        draw_axes(ax, axes_gt, alpha=0.2)
    
    # Set axis properties
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax.zaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    
    ax.set_xlabel('X-axis', fontsize=14)
    ax.set_ylabel('Y-axis', fontsize=14)
    ax.set_zlabel('Z-axis', fontsize=14)

    # Modify the size of the ticks
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='z', labelsize=12)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.view_init(elev, azim)

    
    plt.show()  # Show the plot with all the changes