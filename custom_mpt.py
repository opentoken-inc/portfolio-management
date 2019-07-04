def allocate(mu_vec, cov_mat, lb_vec, ub_vec):
    """
    Implements the "critical line method" of determining the set of efficient
    (mean-variance optimized) portfolios give a risk parameter r.

    Specifically, given the following:
        N := A set of n assets {1, 2, ..., n}
        w := nx1 vector of portfolio weights
        l := nx1 vector of lower bounds on portfolio weights
             by default, this can be a vector of 0s
        u := nx1 vector of upper bounds on portfolio weights
             by default, this can be a vector of 1s
        F := A subset of all assets N that do not lie on either bound (l or u)
        B := A subset of all assets N that do lie on either bound (l or u)

    Choose the allocation weight vector w that maximizes:

    Subject to the following constraints:
    """

    def init(mu_vec, lb_vec, ub_vec):
        """
        Find the first "turning point". This is just the subset of assets with
        highest returns such that  the sum of the upper bounds on portfolio
        weights is exactly 1. In the case where u is a vector of 1s and l is a
        vector of 0s, w is the 100% allocation towards the asset with the
        highest mean.
        """
        i = -1
        w = np.copy(lb_vec)
        size = mu_vec.shape[0]
        a = np.array(list(zip(range(size, mu_vec))),
                     dtype=[('id', int),('mu', float)])
        sorted_a = np.sort(a, order='mu')[::-1]  # Sort in reverse
        while sum(w) < 1:
            i += 1
            w[sorted_a[i][0]] = ub_vec[sorted_a[i][0]]
        w[sorted_a[i][0]] += 1 - sum(w)  # Subtract the excess
        return ([sorted_a[i][0]], w)

    def find_F():
        return

    # Initialize optimization parameters
    w_vec = []
    lambda_vec = []
    gamma_vec = []
    f_vec = []


    return
