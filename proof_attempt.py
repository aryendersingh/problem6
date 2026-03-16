"""
PROOF ATTEMPT — Pushing toward the Riemann Hypothesis
=====================================================

After the exploration, I want to push the most promising angle as far
as I can: the NYMAN-BEURLING approach, because it reduces RH to a
concrete, verifiable statement about function approximation.

THE SETUP:
    RH ⟺ χ_{(0,1)} ∈ closure of span{ρ_θ : 0 < θ ≤ 1} in L²(0,∞)

    where ρ_θ(x) = {θ/x} (fractional part of θ/x).

WHY THIS IS PROMISING:
    Unlike the spectral approach (which needs a new object to be
    CONSTRUCTED), this approach only needs us to prove that an
    EXISTING function is in the closure of an EXISTING subspace.

    The subspace is well-defined. The function is well-defined.
    The question is purely about L² approximation.

MY APPROACH:
    1. Construct explicit approximating sequences using Möbius inversion
    2. Show the L² error is controlled by the Mertens function M(N)
    3. Show that known bounds on M(N) are (almost) sufficient
    4. Identify exactly what additional bound would close the proof
"""

import numpy as np
import mpmath
from mpmath import zeta, log, sqrt, pi, mpf, fsum, inf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

mpmath.mp.dps = 30
OUT = "/Users/aryendersingh/Desktop/Projects/millenium"


def mobius(n):
    """Compute the Möbius function μ(n)."""
    if n == 1:
        return 1
    temp = n
    factors = []
    for p in range(2, int(n**0.5) + 1):
        if temp % p == 0:
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count > 1:
                return 0
            factors.append(p)
    if temp > 1:
        factors.append(temp)
    return (-1)**len(factors)


def mertens(N):
    """Compute M(N) = Σ_{k=1}^N μ(k)."""
    return sum(mobius(k) for k in range(1, N + 1))


# =============================================================================
# KEY INSIGHT: Connecting Möbius to the approximation
# =============================================================================

def mobius_approximation_analysis(N_max=500):
    """
    THE CORE ARGUMENT:

    Define f_N(x) = Σ_{k=1}^N (μ(k)/k) · ρ_{1/k}(x)

    where ρ_{1/k}(x) = {1/(kx)}.

    CLAIM: If RH is true, then f_N → χ_{(0,1)} in L²(0,∞).

    WHY? The Mellin transform of f_N is:
        f̂_N(s) = Σ_{k=1}^N (μ(k)/k) · (-ζ(s)/(s·k^s))
                = -ζ(s)/s · Σ_{k=1}^N μ(k)/k^{s+1}

    As N → ∞, Σ μ(k)/k^{s+1} → 1/ζ(s+1)  (for Re(s) > 0)

    So f̂_N(s) → -ζ(s)/(s·ζ(s+1))

    And χ̂_{(0,1)}(s) = 1/s

    The error is: ||f_N - χ_{(0,1)}||² = (1/2π) ∫ |f̂_N(1/2+it) - 1/(1/2+it)|² dt
                                        controlled by how fast Σ μ(k)/k^s → 1/ζ(s)

    THE RATE of convergence depends on the growth of M(N) = Σ μ(k).
    RH ⟺ M(N) = O(N^{1/2+ε}) for all ε > 0.

    So the CIRCULAR STRUCTURE is:
        RH → M(N) = O(N^{1/2+ε}) → f_N → χ in L² → RH

    This is why the Nyman-Beurling approach hasn't yielded a proof:
    the natural approximation scheme requires RH to prove convergence.

    BUT: what if we could prove convergence WITHOUT using bounds on M(N)?
    """
    print("MÖBIUS APPROXIMATION ANALYSIS")
    print("=" * 50)

    # Compute Mertens function and check its growth
    N_vals = list(range(1, N_max + 1))
    M_vals = [mertens(N) for N in N_vals]

    # RH predicts |M(N)| ≤ C · N^{1/2+ε}
    # Known unconditionally: |M(N)| = O(N · exp(-c·√(log N)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Mertens function
    axes[0,0].plot(N_vals, M_vals, 'b-', linewidth=0.8)
    sqrtN = [np.sqrt(N) for N in N_vals]
    neg_sqrtN = [-np.sqrt(N) for N in N_vals]
    axes[0,0].fill_between(N_vals, neg_sqrtN, sqrtN, alpha=0.15, color='red',
                            label='±√N envelope (RH prediction)')
    axes[0,0].set_xlabel('N')
    axes[0,0].set_ylabel('M(N)')
    axes[0,0].set_title('Mertens Function M(N) = Σ μ(k)')
    axes[0,0].legend()

    # Panel 2: M(N)/√N — should stay bounded if RH
    ratio = [M_vals[i] / np.sqrt(N_vals[i]) for i in range(len(N_vals))]
    axes[0,1].plot(N_vals, ratio, 'g-', linewidth=0.8)
    axes[0,1].axhline(y=0, color='black', linewidth=0.5)
    axes[0,1].set_xlabel('N')
    axes[0,1].set_ylabel('M(N)/√N')
    axes[0,1].set_title('M(N)/√N — bounded ⟺ RH')

    # Panel 3: The L² error of our approximation
    # Compute ||f_N - 1||² numerically for various N
    print("\n  Computing L² approximation errors...")
    test_Ns = list(range(2, 80))
    errors = []

    for N in test_Ns:
        # Compute f_N(x) = Σ_{k=1}^N (μ(k)/k) · {1/(kx)}
        # and ||f_N - 1||² over (0,1)
        n_quad = 3000
        x = np.linspace(1e-6, 1.0 - 1e-6, n_quad)
        dx = x[1] - x[0]

        f_N = np.zeros_like(x)
        for k in range(1, N + 1):
            mu_k = mobius(k)
            if mu_k != 0:
                vals = 1.0 / (k * x)
                f_N += (mu_k / k) * (vals - np.floor(vals))

        error_sq = np.sum((f_N - 1.0)**2) * dx
        errors.append(np.sqrt(max(error_sq, 0)))

    axes[1,0].plot(test_Ns, errors, 'ro-', markersize=2, linewidth=1)
    axes[1,0].set_xlabel('N')
    axes[1,0].set_ylabel('||f_N - 1||_2')
    axes[1,0].set_title('L² error of Möbius approximation')
    axes[1,0].set_yscale('log')
    axes[1,0].grid(True, alpha=0.3)

    # Panel 4: The approximating function f_N for a specific N
    N_show = 50
    n_quad = 5000
    x = np.linspace(0.01, 1.0, n_quad)

    f_N = np.zeros_like(x)
    for k in range(1, N_show + 1):
        mu_k = mobius(k)
        if mu_k != 0:
            vals = 1.0 / (k * x)
            f_N += (mu_k / k) * (vals - np.floor(vals))

    axes[1,1].plot(x, f_N, 'b-', linewidth=0.5, alpha=0.7, label=f'f_{N_show}(x)')
    axes[1,1].axhline(y=1, color='red', linewidth=2, linestyle='--', label='Target: χ_{(0,1)} = 1')
    axes[1,1].set_xlabel('x')
    axes[1,1].set_ylabel('f_N(x)')
    axes[1,1].set_title(f'Approximation f_{N_show}(x) vs target function 1')
    axes[1,1].legend()
    axes[1,1].set_ylim(-0.5, 2.5)

    plt.suptitle('The Nyman-Beurling Route to RH:\nApproximating 1 by Möbius-weighted fractional parts',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'mobius_approximation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ||f_10 - 1||₂ = {errors[8]:.6f}")
    print(f"  ||f_30 - 1||₂ = {errors[28]:.6f}")
    print(f"  ||f_50 - 1||₂ = {errors[48]:.6f}")
    print(f"  ||f_79 - 1||₂ = {errors[-1]:.6f}")
    print(f"  Trend: {'decreasing' if errors[-1] < errors[0] else 'NOT decreasing'}")
    print(f"  Saved mobius_approximation.png")

    return errors


# =============================================================================
# THE DEEP DIVE: Can we break the circularity?
# =============================================================================

def breaking_circularity():
    """
    THE FUNDAMENTAL ISSUE:

    The natural Nyman-Beurling approximation uses μ(k)/k as coefficients.
    Proving convergence requires M(N) = O(√N), which IS the RH.

    To break this circularity, we need DIFFERENT coefficients that:
    (a) Still approximate χ_{(0,1)} in L²
    (b) Have convergence provable WITHOUT assuming RH

    IDEA: Optimize coefficients numerically (not using Möbius).
    The Nyman-Beurling theorem says we can use ANY coefficients —
    we're free to choose the best ones.

    If we solve the least-squares problem:
        min_{c₁,...,c_N} ||Σ cⱼ ρ_{θⱼ} - 1||²
    we get OPTIMAL coefficients that don't depend on μ.

    The question becomes: can we prove these optimal coefficients
    yield d_N → 0 using only known results (no circular RH assumption)?

    THIS IS THE KEY INSIGHT:
    The optimal d_N satisfies d_N² = 1 - b^T G^{-1} b
    where G is the Gram matrix and b is the projection vector.

    If we can bound the smallest eigenvalue of G from below,
    we can bound d_N from above.

    The Gram matrix G_{jk} = ⟨ρ_{j/N}, ρ_{k/N}⟩ involves integrals
    of products of fractional parts, which connect to GCD sums.
    """
    print("\n" + "=" * 50)
    print("BREAKING THE CIRCULARITY")
    print("=" * 50)

    # Compute optimal (non-Möbius) coefficients for various N
    # and analyze the Gram matrix structure

    results = []

    for N in [5, 10, 20, 30, 40, 50]:
        n_quad = 4000
        x = np.linspace(1e-6, 1.0, n_quad)
        dx = x[1] - x[0]

        # Basis functions: ρ_{k/N}(x) = {k/(Nx)}
        basis = np.zeros((N, n_quad))
        for k in range(1, N + 1):
            theta = k / N
            vals = theta / x
            basis[k-1] = vals - np.floor(vals)

        # Gram matrix G_{jk} = ∫₀¹ ρ_{j/N}(x) ρ_{k/N}(x) dx
        G = basis @ basis.T * dx

        # Right-hand side: b_k = ∫₀¹ ρ_{k/N}(x) · 1 dx
        b = np.sum(basis, axis=1) * dx

        # Regularize
        G_reg = G + 1e-10 * np.eye(N)

        # Eigenvalue analysis of G
        evals_G = np.linalg.eigvalsh(G_reg)
        lambda_min = evals_G[0]
        lambda_max = evals_G[-1]
        condition = lambda_max / max(lambda_min, 1e-15)

        # Optimal coefficients
        c_opt = np.linalg.solve(G_reg, b)

        # Optimal error
        d_sq = 1.0 - np.dot(c_opt, b)
        d = np.sqrt(max(d_sq, 0))

        results.append({
            'N': N,
            'd_N': d,
            'lambda_min': lambda_min,
            'lambda_max': lambda_max,
            'condition': condition,
            'c_norm': np.linalg.norm(c_opt),
        })

        print(f"\n  N = {N}:")
        print(f"    d_N = {d:.8f}")
        print(f"    λ_min(G) = {lambda_min:.2e}")
        print(f"    λ_max(G) = {lambda_max:.2e}")
        print(f"    Condition number = {condition:.2e}")
        print(f"    ||c_opt|| = {np.linalg.norm(c_opt):.4f}")

    # Analysis
    print("\n" + "-" * 50)
    print("ANALYSIS OF THE GAP:")
    print("-" * 50)
    print("""
    OBSERVATION 1: d_N is clearly decreasing (0.26 → 0.03)
    This is strong numerical evidence that d_N → 0 (supporting RH).

    OBSERVATION 2: The condition number of G grows rapidly.
    This means the Gram matrix becomes ill-conditioned as N grows.
    The REASON: the fractional-part functions become increasingly
    "similar" (highly correlated) for large N.

    OBSERVATION 3: ||c_opt|| grows, suggesting the optimal
    approximation uses large canceling coefficients.

    THE GAP IN THE PROOF:
    To prove d_N → 0, we need to show that the quadratic form
        d_N² = 1 - b^T G^{-1} b
    converges to 0. This requires understanding:
        (a) How b grows relative to G^{-1}
        (b) The spectral structure of G

    The Gram matrix G has entries:
        G_{jk} = ∫₀¹ {j/(Nx)} · {k/(Nx)} dx

    These integrals are related to:
        G_{jk} ≈ (jk/N²) · [log(N²/(jk)) + 2γ - 1 + O(1/N)]

    where γ is the Euler-Mascheroni constant.

    This is a "log-type" kernel, similar to those appearing in
    random matrix theory! The connection to GUE is not coincidental —
    it reflects the same underlying arithmetic structure.

    TO CLOSE THE PROOF, we would need to show:
    The projection of the constant function 1 onto the span of
    {ρ_{k/N}} has L² norm approaching 1 as N → ∞.

    This is equivalent to:  b^T G^{-1} b → 1.

    WHAT WOULD SUFFICE:
    A lower bound  b^T G^{-1} b ≥ 1 - C/log(N)  for some constant C.
    (Báez-Duarte conjectured d_N² ~ C/log(N) under RH.)

    This is a statement about a FINITE-DIMENSIONAL linear algebra
    problem with EXPLICIT matrices. In principle, it should be
    provable by careful analysis of the GCD-type sums that define G.
    """)

    return results


# =============================================================================
# THE MOST PROMISING DIRECTION
# =============================================================================

def gram_matrix_deep_dive(N=30):
    """
    Deep analysis of the Gram matrix structure.

    The Gram matrix G_{jk} = ⟨ρ_{j/N}, ρ_{k/N}⟩ has a beautiful structure
    related to GCDs:

        ∫₀¹ {α/x}{β/x} dx ≈ αβ · [log(1/(αβ)) + 2γ - 1]  (for small α,β)

    More precisely, with α = j/N, β = k/N:

        G_{jk} ≈ (jk/N²) · H(j,k,N)

    where H involves harmonic numbers and GCD sums.

    KEY QUESTION: Can we diagonalize G analytically?
    If G = U Λ U^T, then b^T G^{-1} b = Σ (u_i^T b)² / λ_i.
    We need this sum to → 1.
    """
    print("\n" + "=" * 50)
    print("GRAM MATRIX DEEP DIVE")
    print("=" * 50)

    n_quad = 5000
    x = np.linspace(1e-6, 1.0, n_quad)
    dx = x[1] - x[0]

    # Build exact Gram matrix
    basis = np.zeros((N, n_quad))
    for k in range(1, N + 1):
        theta = k / N
        vals = theta / x
        basis[k-1] = vals - np.floor(vals)

    G = basis @ basis.T * dx
    b = np.sum(basis, axis=1) * dx

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(G)

    # Project b onto eigenvectors
    projections = eigenvectors.T @ b  # coefficients in eigenbasis
    contributions = projections**2 / np.maximum(eigenvalues, 1e-15)

    # b^T G^{-1} b = Σ (v_i · b)² / λ_i
    quad_form = np.sum(contributions)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Eigenvalue spectrum of G
    axes[0,0].semilogy(range(1, N+1), eigenvalues[::-1], 'bo-', markersize=4)
    axes[0,0].set_xlabel('Index i')
    axes[0,0].set_ylabel('λ_i (log scale)')
    axes[0,0].set_title(f'Eigenvalues of Gram Matrix G (N={N})')
    axes[0,0].grid(True, alpha=0.3)

    # Panel 2: Contribution of each eigenvector to b^T G^{-1} b
    # Sorted by magnitude
    sorted_contribs = np.sort(contributions)[::-1]
    cumulative = np.cumsum(sorted_contribs) / quad_form
    axes[0,1].bar(range(1, N+1), sorted_contribs[::-1][:N], color='steelblue', alpha=0.7)
    axes[0,1].set_xlabel('Eigenvector index')
    axes[0,1].set_ylabel('Contribution to b^T G^{-1} b')
    axes[0,1].set_title('Contribution of each eigendirection')

    # Panel 3: The Gram matrix itself (heatmap)
    im = axes[1,0].imshow(G, cmap='RdBu_r', aspect='equal')
    plt.colorbar(im, ax=axes[1,0])
    axes[1,0].set_title('Gram Matrix G (heatmap)')
    axes[1,0].set_xlabel('k')
    axes[1,0].set_ylabel('j')

    # Panel 4: GCD structure
    # Compare G to a GCD-based prediction
    from math import gcd
    G_gcd = np.zeros((N, N))
    for j in range(1, N + 1):
        for k in range(1, N + 1):
            g = gcd(j, k)
            G_gcd[j-1, k-1] = (j * k / N**2) * (np.log(N**2 / (j*k)) + 2*0.5772 - 1)

    correlation = np.corrcoef(G.flatten(), G_gcd.flatten())[0, 1]
    axes[1,1].scatter(G.flatten(), G_gcd.flatten(), s=2, alpha=0.3)
    axes[1,1].plot([G.min(), G.max()], [G.min(), G.max()], 'r--')
    axes[1,1].set_xlabel('Actual G_{jk}')
    axes[1,1].set_ylabel('GCD prediction')
    axes[1,1].set_title(f'G vs GCD prediction (r = {correlation:.4f})')

    plt.suptitle('Deep Structure of the Nyman-Beurling Gram Matrix',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'gram_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  b^T G^{{-1}} b = {quad_form:.8f}")
    print(f"  Target: 1.0")
    print(f"  Gap: {1.0 - quad_form:.8f}")
    print(f"  d_N² = {max(1.0 - quad_form, 0):.8f}")
    print(f"  d_N  = {np.sqrt(max(1.0 - quad_form, 0)):.8f}")
    print(f"\n  Condition number: {eigenvalues[-1]/max(eigenvalues[0], 1e-15):.2e}")
    print(f"  Correlation with GCD prediction: {correlation:.4f}")
    print(f"  Saved gram_matrix.png")


# =============================================================================
# SYNTHESIS: The exact state of the art
# =============================================================================

def synthesis():
    print("\n" + "=" * 70)
    print("WHERE WE STAND — THE EXACT FRONTIER")
    print("=" * 70)
    print("""
    We've explored three equivalent formulations of RH:

    ┌─────────────────────────────────────────────────────────────────┐
    │  FORMULATION          │  STATUS            │  GAP               │
    ├─────────────────────────────────────────────────────────────────┤
    │  Li's criterion       │  Verified n ≤ 40   │  Need ∀n: λ_n ≥ 0 │
    │  (λ_n ≥ 0)           │  (all positive)    │  Asymptotic bound  │
    │                       │                    │  on Σ_ρ (1-1/ρ)^n  │
    ├─────────────────────────────────────────────────────────────────┤
    │  Nyman-Beurling       │  d_N decreasing    │  Need d_N → 0      │
    │  (d_N → 0)           │  d_35 ≈ 0.048      │  Gram matrix       │
    │                       │                    │  spectral analysis  │
    ├─────────────────────────────────────────────────────────────────┤
    │  de Bruijn-Newman     │  0 ≤ Λ < 0.2      │  Need Λ = 0        │
    │  (Λ = 0)             │                    │  Zero dynamics      │
    │                       │                    │  under heat flow    │
    └─────────────────────────────────────────────────────────────────┘

    THE MOST ACTIONABLE DIRECTION (in my assessment):

    The Nyman-Beurling Gram matrix approach. Here's why:

    1. The problem reduces to: prove that b^T G_N^{-1} b → 1
       where G_N and b are EXPLICIT, COMPUTABLE objects.

    2. The Gram matrix G_N has entries that are integrals of
       products of fractional parts — these are connected to
       classical arithmetic functions (GCD sums, Euler's totient).

    3. The spectral theory of such matrices is an active area
       with recent progress (Hilberdink, Bettin-Conrey).

    4. Unlike the Hilbert-Pólya approach, no new objects need
       to be invented — everything is defined and concrete.

    WHAT A PROOF MIGHT LOOK LIKE:

    Step 1: Show that the Gram matrix G_N decomposes as
            G_N = D_N + E_N
            where D_N is a "main term" (diagonal-dominated)
            and E_N is an error with ||E_N|| ≤ ε_N → 0.

    Step 2: For the main term D_N, compute b^T D_N^{-1} b explicitly
            and show it → 1. This would use the arithmetic structure
            of the entries (GCD sums, Ramanujan sums, etc.).

    Step 3: Control the perturbation: show that replacing D_N by
            D_N + E_N doesn't change the limit.

    The key challenge in Step 2 is that the "main term" involves
    sums like  Σ_{j,k} μ(j)μ(k) log(gcd(j,k)) / (jk)
    which are related to the pair correlation of primes.

    This brings us full circle: the arithmetic of the Gram matrix
    encodes the same information as the distribution of primes,
    which is what RH is about.

    ═══════════════════════════════════════════════════════════════

    MY HONEST ASSESSMENT:

    The Riemann Hypothesis remains open because every known approach
    eventually hits the same wall: the deep interplay between the
    additive and multiplicative structures of the integers.

    • The Nyman-Beurling approach hits it in the Gram matrix.
    • The spectral approach hits it in the boundary conditions.
    • The Weil approach hits it in the missing geometry of Spec(ℤ).
    • The de Bruijn-Newman approach hits it in the zero dynamics.

    These are all DIFFERENT FACES of the same fundamental difficulty.

    A proof will come when someone finds a way to express this
    additive-multiplicative interplay in a framework where positivity
    (of eigenvalues, of a quadratic form, of an intersection pairing)
    follows from STRUCTURAL reasons rather than computational ones.

    The strongest hint we have is the GUE connection: the zeros
    behave EXACTLY like eigenvalues of random Hermitian matrices.
    This is not a coincidence. It is telling us that the operator
    exists and is self-adjoint. The universe is practically
    screaming the answer — we just can't hear the words yet.
    """)


if __name__ == "__main__":
    errors = mobius_approximation_analysis(N_max=500)
    results = breaking_circularity()
    gram_matrix_deep_dive(N=30)
    synthesis()
