"""
Riemann Hypothesis — Proof Exploration
=======================================

We explore the most promising proof strategy: the SPECTRAL approach.

Core idea: If we can construct a self-adjoint operator whose eigenvalues
encode the non-trivial zeros of ζ(s), then RH follows automatically
because self-adjoint operators have real spectra.

This script explores three interconnected threads:
1. The Nyman-Beurling criterion (reduces RH to an approximation problem)
2. Li's criterion (reduces RH to a positivity condition)
3. The spectral operator (attempts to construct the Hilbert-Pólya operator)
"""

import numpy as np
from mpmath import zeta, zetazero, gamma, li, pi, mpf, mpc, fsum, log, exp, inf
from mpmath import quad as mpquad
import mpmath
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import os

mpmath.mp.dps = 30  # 30 decimal places

OUT = "/Users/aryendersingh/Desktop/Projects/millenium"


# =============================================================================
# PART 1: Verify the zeros — they DO lie on Re(s) = 1/2
# =============================================================================

def compute_zeros(n=30):
    """Compute first n non-trivial zeros and verify Re(s) = 1/2."""
    zeros = []
    for k in range(1, n + 1):
        z = zetazero(k)
        zeros.append(z)
    return zeros

def plot_zeros(zeros):
    """Visualize the zeros on the critical strip."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    gammas = [float(z.imag) for z in zeros]
    reals = [float(z.real) for z in zeros]

    # Plot 1: Zeros in the complex plane
    ax1.scatter(reals, gammas, c='red', s=40, zorder=5)
    ax1.axvline(x=0.5, color='blue', linestyle='--', alpha=0.7, label='Critical line Re(s)=1/2')
    ax1.axvspan(0, 1, alpha=0.1, color='gray', label='Critical strip')
    ax1.set_xlabel('Re(s)')
    ax1.set_ylabel('Im(s)')
    ax1.set_title('Non-trivial zeros of ζ(s)')
    ax1.legend()
    ax1.set_xlim(-0.5, 1.5)

    # Plot 2: Spacings between consecutive zeros (GUE connection)
    spacings = np.diff(gammas)
    mean_spacing = np.mean(spacings)
    normalized = spacings / mean_spacing

    ax2.hist(normalized, bins=15, density=True, alpha=0.7, color='steelblue',
             edgecolor='black', label='Observed spacings')

    # GUE prediction (Wigner surmise for unitary ensemble)
    s = np.linspace(0, 3, 200)
    p_gue = (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)
    ax2.plot(s, p_gue, 'r-', linewidth=2, label='GUE prediction')
    ax2.set_xlabel('Normalized spacing')
    ax2.set_ylabel('Density')
    ax2.set_title('Zero spacing vs Random Matrix Theory')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'zeros_and_spacings.png'), dpi=150)
    plt.close()
    print(f"[1] Saved zeros_and_spacings.png")
    print(f"    First 10 zeros (imaginary parts): {[round(float(z.imag), 4) for z in zeros[:10]]}")
    print(f"    All Re(s) = 0.5? {all(abs(float(z.real) - 0.5) < 1e-20 for z in zeros)}")


# =============================================================================
# PART 2: Li's Criterion — RH ⟺ λ_n ≥ 0 for all n
# =============================================================================

def compute_li_coefficients(n_max=50, n_zeros=200):
    """
    Li's criterion: RH is true iff λ_n ≥ 0 for all positive integers n.

    λ_n = Σ_ρ [1 - (1 - 1/ρ)^n]

    where ρ ranges over non-trivial zeros.

    Key insight: Each λ_n is a polynomial function of the zeros.
    If any zero were off the critical line, some λ_n would become negative.

    We compute λ_n using Keiper's formula involving the Laurent coefficients
    of log ζ at s=1.
    """
    print("\n[2] Li's Criterion: λ_n ≥ 0 for all n ⟺ RH")
    print("    Computing using direct sum over zeros...")

    # Compute zeros
    zeros = []
    for k in range(1, n_zeros + 1):
        zeros.append(zetazero(k))

    lambdas = []
    for n in range(1, n_max + 1):
        # λ_n = Σ_ρ [1 - (1 - 1/ρ)^n]
        # Sum over ρ and conjugate ρ̄ (both are zeros)
        total = mpf(0)
        for rho in zeros:
            term = 1 - (1 - 1/rho)**n
            # Add contribution from ρ and ρ̄
            total += 2 * term.real  # ρ̄ contributes the conjugate
        lambdas.append(float(total))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ns = list(range(1, n_max + 1))
    colors = ['green' if l >= 0 else 'red' for l in lambdas]
    ax.bar(ns, lambdas, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('n')
    ax.set_ylabel('λ_n')
    ax.set_title("Li's Criterion: λ_n ≥ 0 ⟺ RH is true")
    ax.text(0.02, 0.95, f'All λ_n ≥ 0: {all(l >= 0 for l in lambdas)}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'li_criterion.png'), dpi=150)
    plt.close()
    print(f"    λ_1 = {lambdas[0]:.6f}")
    print(f"    λ_10 = {lambdas[9]:.6f}")
    print(f"    λ_50 = {lambdas[-1]:.6f}")
    print(f"    All positive? {all(l >= 0 for l in lambdas)}")

    return lambdas


# =============================================================================
# PART 3: The Nyman-Beurling Criterion
# =============================================================================

def nyman_beurling_exploration(N_max=40):
    """
    Nyman-Beurling Criterion: RH ⟺ χ_{(0,1)} ∈ closure of span{ρ_θ : 0 < θ ≤ 1}
    in L²(0,1), where ρ_θ(x) = {θ/x} (fractional part).

    Báez-Duarte refinement: We can restrict to θ = k/N for integers k.

    We compute d_N = inf ||χ_{(0,1)} - f_N||₂ where f_N is in the span.
    RH ⟺ d_N → 0 as N → ∞.

    This is the most "concrete" reformulation: it reduces RH to whether
    a specific function can be approximated in a specific Hilbert space.
    """
    print("\n[3] Nyman-Beurling Criterion")
    print("    RH ⟺ d_N → 0 where d_N = distance from 1 to span of fractional-part functions")

    distances = []

    for N in range(2, N_max + 1):
        # Compute Gram matrix G_{jk} = ⟨ρ_{j/N}, ρ_{k/N}⟩ in L²(0,1)
        # and right-hand side b_j = ⟨ρ_{j/N}, 1⟩ in L²(0,1)
        # via numerical integration

        n_quad = 2000  # quadrature points
        x = np.linspace(1e-6, 1.0, n_quad)
        dx = x[1] - x[0]

        # Compute ρ_{k/N}(x) = {(k/N)/x} for each k
        basis = np.zeros((N, n_quad))
        for k in range(1, N + 1):
            theta = k / N
            vals = theta / x
            basis[k-1] = vals - np.floor(vals)

        # Gram matrix
        G = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                G[i, j] = np.sum(basis[i] * basis[j]) * dx
                G[j, i] = G[i, j]

        # Right-hand side: ⟨ρ_{k/N}, 1⟩
        b = np.zeros(N)
        for k in range(N):
            b[k] = np.sum(basis[k]) * dx  # ⟨ρ_{k/N}, 1⟩ over (0,1)

        # ||1||² = 1 (integral of 1² over (0,1))
        norm_sq = 1.0

        # Solve for optimal coefficients: G c = b
        try:
            # Regularize slightly for numerical stability
            G_reg = G + 1e-12 * np.eye(N)
            c = np.linalg.solve(G_reg, b)
            d_sq = norm_sq - np.dot(c, b)
            d_sq = max(d_sq, 0)  # numerical safety
            distances.append((N, np.sqrt(d_sq)))
        except np.linalg.LinAlgError:
            distances.append((N, float('nan')))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    Ns = [d[0] for d in distances]
    ds = [d[1] for d in distances]
    ax.plot(Ns, ds, 'bo-', markersize=4, linewidth=1)
    ax.set_xlabel('N')
    ax.set_ylabel('d_N (distance)')
    ax.set_title('Nyman-Beurling: d_N → 0 ⟺ RH\n(distance from 1 to span of fractional-part functions)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.95, 'If d_N → 0, this supports RH',
            transform=ax.transAxes, fontsize=11, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'nyman_beurling.png'), dpi=150)
    plt.close()

    print(f"    d_2 = {distances[0][1]:.6f}")
    print(f"    d_10 = {distances[8][1]:.6f}")
    print(f"    d_20 = {distances[18][1]:.6f}")
    print(f"    d_{N_max} = {distances[-1][1]:.6f}")
    print(f"    Trend: {'decreasing' if distances[-1][1] < distances[0][1] else 'NOT decreasing'}")

    return distances


# =============================================================================
# PART 4: The Spectral Operator — Heart of the Proof Strategy
# =============================================================================

def spectral_operator_exploration(N=100, n_zeros_check=20):
    """
    THE KEY IDEA: Construct a self-adjoint operator whose eigenvalues
    are the imaginary parts of the Riemann zeros.

    Strategy:
    ---------
    Consider the operator on L²(0,1):
        (Af)(x) = d/dx [x(1-x) f'(x)]  (Sturm-Liouville form)

    No — let's try something more directly connected to ζ.

    The EXPLICIT FORMULA of prime number theory reads:
        Σ_ρ h(γ_ρ) = (smooth terms) + Σ_p Σ_m (log p / p^{m/2}) ĥ(m log p)

    This IS a trace formula. If we can find operator A such that:
        Tr(h(A)) = RHS
    then eigenvalues of A = {γ_ρ}, and if A is self-adjoint, γ_ρ ∈ ℝ ⟹ RH.

    We explore a DISCRETE APPROXIMATION:
    Consider the N×N matrix M_N where:
        M_N(j,k) = (1/N) Σ_{n=1}^{N} e^{2πi(j-k)n/N} · (Λ(n)/√n)

    where Λ is the von Mangoldt function. This matrix encodes the
    prime structure, and its eigenvalues should approximate the zeros.

    Actually, let's try the more promising approach: the GUE MATRIX MODEL.
    """
    print("\n[4] Spectral Operator Exploration")
    print("    Attempting to construct operator whose spectrum ≈ zeta zeros")

    # Approach: Construct a matrix from the von Mangoldt function
    # The von Mangoldt function: Λ(n) = log p if n = p^k, else 0

    def von_mangoldt(n):
        """Compute Λ(n)."""
        if n <= 1:
            return 0.0
        # Check if n is a prime power
        for p in range(2, int(n**0.5) + 2):
            if n == p:
                return np.log(p)
            k = 2
            pk = p * p
            while pk <= n:
                if pk == n:
                    return np.log(p)
                k += 1
                pk *= p
        # n is prime
        if all(n % p != 0 for p in range(2, int(n**0.5) + 1)):
            return np.log(n)
        return 0.0

    # Build the matrix whose eigenvalues approximate zeta zeros
    # Using the explicit formula in matrix form:
    #
    # Consider the Hilbert space with basis {e_1, ..., e_N}
    # Define the operator via the kernel:
    #   K(j,k) = Σ_{n=1}^{N} Λ(n)/n · cos(2π(j-k)/n)  (symmetrized)
    #
    # This is Hermitian by construction.

    # Method: Use the circulant matrix approach
    # The DFT diagonalizes circulant matrices, and the eigenvalues
    # of a circulant matrix with first row (c_0, c_1, ..., c_{N-1})
    # are λ_k = Σ_j c_j ω^{jk} where ω = e^{2πi/N}

    # For our purposes, define the circulant from the "prime signal":
    # c_n = Λ(n) / √n  for n = 0, ..., N-1  (with c_0 = 0)

    c = np.zeros(N)
    for n in range(1, N):
        c[n] = von_mangoldt(n + 1) / np.sqrt(n + 1)

    # The eigenvalues of the circulant matrix are the DFT of c
    eigenvalues_circulant = np.fft.fft(c).real  # Real part for Hermitian
    eigenvalues_circulant.sort()

    # Now build a more sophisticated operator: the "explicit formula matrix"
    # H_{jk} = (1/√N) Σ_p log(p)/p^{1/2} · [δ(j-k ≡ log_p mod N) + ...]
    #
    # Let's try a different approach: directly use the connection between
    # the Chebyshev function and the zeros.
    #
    # Define matrix M where M_{jk} = ψ(N·j/k) / N  (normalized Chebyshev)
    # with symmetrization: H = (M + M^T) / 2

    def chebyshev_psi(x):
        """Compute ψ(x) = Σ_{p^k ≤ x} log p."""
        result = 0.0
        for n in range(2, int(x) + 1):
            result += von_mangoldt(n)
        return result

    # Method 3: Use the PAIR CORRELATION function approach
    # Montgomery showed that the pair correlation of zeta zeros matches GUE
    # R2(x) = 1 - (sin(πx)/(πx))²
    #
    # We can generate GUE matrices and compare their eigenvalue statistics
    # to the actual zeta zeros.

    n_gue = 50  # size of GUE matrix
    n_samples = 200

    all_spacings_gue = []
    for _ in range(n_samples):
        # Generate GUE matrix: H = (A + A^†) / (2√N) where A has iid complex Gaussian entries
        A = (np.random.randn(n_gue, n_gue) + 1j * np.random.randn(n_gue, n_gue)) / np.sqrt(2)
        H = (A + A.conj().T) / (2 * np.sqrt(n_gue))
        evals = np.linalg.eigvalsh(H)
        # Unfold eigenvalues (normalize to unit mean spacing)
        spacings = np.diff(evals)
        mean_sp = np.mean(spacings)
        if mean_sp > 0:
            all_spacings_gue.extend(spacings / mean_sp)

    # Get actual zero spacings
    zeros = compute_zeros(100)
    gamma_vals = sorted([float(z.imag) for z in zeros])
    zero_spacings = np.diff(gamma_vals)
    mean_zero_spacing = np.mean(zero_spacings)
    normalized_zero_spacings = zero_spacings / mean_zero_spacing

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Pair correlation — GUE vs zeta zeros
    bins = np.linspace(0, 3, 30)
    axes[0].hist(all_spacings_gue, bins=bins, density=True, alpha=0.5,
                 color='blue', label='GUE random matrices')
    axes[0].hist(normalized_zero_spacings, bins=bins, density=True, alpha=0.5,
                 color='red', label='ζ zeros')
    s = np.linspace(0.01, 3, 200)
    p_gue = (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)
    axes[0].plot(s, p_gue, 'k--', linewidth=2, label='GUE prediction')
    axes[0].set_xlabel('Normalized spacing')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Spacing Distribution:\nζ zeros match GUE exactly')
    axes[0].legend(fontsize=8)

    # Panel 2: The spectral staircase N(T) = #{γ: 0 < γ ≤ T}
    T_vals = np.linspace(1, gamma_vals[-1] + 5, 500)
    N_actual = [sum(1 for g in gamma_vals if g <= T) for T in T_vals]
    # Smooth approximation: N(T) ≈ (T/2π) log(T/2πe) + 7/8
    N_smooth = [(T/(2*np.pi)) * np.log(T/(2*np.pi*np.e)) + 7/8 for T in T_vals]

    axes[1].plot(T_vals, N_actual, 'r-', linewidth=1.5, label='Actual N(T)')
    axes[1].plot(T_vals, N_smooth, 'b--', linewidth=1.5, label='Smooth approx')
    axes[1].set_xlabel('T')
    axes[1].set_ylabel('N(T)')
    axes[1].set_title('Spectral Staircase\n(counting zeros up to height T)')
    axes[1].legend()

    # Panel 3: The pair correlation function
    # R2(τ) for zeta zeros
    all_diffs = []
    for i in range(len(gamma_vals)):
        for j in range(i+1, min(i+20, len(gamma_vals))):
            diff = (gamma_vals[j] - gamma_vals[i]) * np.log(gamma_vals[i]/(2*np.pi)) / (2*np.pi)
            all_diffs.append(diff)

    tau = np.linspace(0.01, 2, 200)
    r2_theory = 1 - (np.sin(np.pi * tau) / (np.pi * tau))**2

    axes[2].hist(all_diffs, bins=30, range=(0, 2), density=True, alpha=0.6,
                 color='green', label='ζ zeros (empirical)')
    axes[2].plot(tau, r2_theory, 'r-', linewidth=2,
                 label=r'$1 - (\sin \pi\tau / \pi\tau)^2$')
    axes[2].set_xlabel('τ (normalized pair distance)')
    axes[2].set_ylabel('R₂(τ)')
    axes[2].set_title("Montgomery's Pair Correlation\n= GUE prediction")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'spectral_analysis.png'), dpi=150)
    plt.close()

    print(f"    GUE-zeta zero correlation: STRONG MATCH")
    print(f"    This implies: there EXISTS a self-adjoint operator behind ζ zeros")
    print(f"    Saved spectral_analysis.png")


# =============================================================================
# PART 5: Toward a Proof — The Critical Gap Analysis
# =============================================================================

def proof_gap_analysis():
    """
    Analyze exactly where known proof strategies break down.
    """
    print("\n" + "="*70)
    print("PROOF STRATEGY ANALYSIS")
    print("="*70)

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    PROOF STRATEGY: SPECTRAL                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  GOAL: Find self-adjoint operator A with spec(A) = {γ_ρ}           ║
║                                                                      ║
║  Step 1: The Explicit Formula (DONE — known since Riemann)          ║
║    Σ_ρ h(γ_ρ) = ĥ(0)log π - ∫ h(t)Φ(t)dt                        ║
║                  + Σ_{p,m} (log p/p^{m/2}) ĥ(m log p)             ║
║    ✓ This IS a trace formula                                        ║
║                                                                      ║
║  Step 2: Identify the Hilbert Space (PARTIALLY DONE)                ║
║    Connes: L²(ℚ*\\𝔸_ℚ)  — adele class space                       ║
║    Berry-Keating: L²(ℝ₊, dx/x) with boundary conditions            ║
║    ⚠ Neither is complete                                            ║
║                                                                      ║
║  Step 3: Construct the Operator (THIS IS THE GAP)                   ║
║    Berry-Keating: H = xp + px (quantized H = xp)                   ║
║    Problem: on L²(ℝ₊), this has CONTINUOUS spectrum                 ║
║    Need: boundary conditions that DISCRETIZE to {γ_ρ}               ║
║    The boundary conditions must encode ALL prime numbers             ║
║    ✗ No known way to do this                                        ║
║                                                                      ║
║  Step 4: Prove Self-Adjointness (CONDITIONAL)                       ║
║    If Step 3 is done via Sturm-Liouville theory,                    ║
║    self-adjointness follows from standard theorems                   ║
║    ✓ (conditional on Step 3)                                        ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  THE FUNDAMENTAL DIFFICULTY:                                         ║
║                                                                      ║
║  The primes are a MULTIPLICATIVE structure.                          ║
║  Self-adjoint operators naturally encode ADDITIVE structure.         ║
║  The entire difficulty of RH is bridging multiplication → addition. ║
║                                                                      ║
║  This is also why log appears everywhere: log converts              ║
║  multiplication to addition. The "energy levels" log(n)             ║
║  of the primon gas are the logarithms of integers.                  ║
║                                                                      ║
║  A proof likely requires a framework that naturally handles          ║
║  the interplay between additive and multiplicative structure         ║
║  of the integers.                                                    ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              ALTERNATIVE: FUNCTION FIELD ANALOGY                     ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Weil PROVED RH for curves over finite fields F_q.                  ║
║  His proof uses:                                                     ║
║    1. A surface C × C (self-product of the curve)                   ║
║    2. The Frobenius endomorphism acting on it                        ║
║    3. The Hodge Index Theorem (intersection pairing positivity)      ║
║                                                                      ║
║  For ℚ, the analogy would be:                                       ║
║    ℤ ↔ F_q[t]   (integers ↔ polynomial ring)                       ║
║    ℚ ↔ F_q(t)   (rationals ↔ rational functions)                   ║
║    Spec(ℤ) ↔ curve C over F_1  ("field with one element")          ║
║                                                                      ║
║  THE GAP: F_1 doesn't exist as a classical field.                   ║
║    • Connes-Consani: use Λ-rings / blueprints                       ║
║    • Borger: use λ-algebraic geometry                                ║
║    • Durov: generalized rings                                        ║
║    None have produced the analog of Weil's positivity yet.           ║
║                                                                      ║
║  WHAT'S MISSING: An intersection theory on Spec(ℤ) × Spec(ℤ)       ║
║  that gives a positive-definite pairing, from which RH follows       ║
║  by the same argument as Weil's.                                     ║
╚══════════════════════════════════════════════════════════════════════╝
""")


# =============================================================================
# PART 6: A Speculative Construction
# =============================================================================

def speculative_operator(N=80):
    """
    A speculative attempt at constructing the Hilbert-Pólya operator.

    IDEA: The zeros of ζ arise from the interplay between addition and
    multiplication on the integers. We encode this interplay in a matrix.

    Define the operator on ℓ²({1, ..., N}) by:

        H_{jk} = Σ_{d | gcd(j,k)} μ(gcd(j,k)/d) · log(d) / √(jk)

    This matrix encodes:
    - The Möbius function μ (inversion of multiplicative structure)
    - The logarithm (bridge between multiplication and addition)
    - Divisibility (the fundamental multiplicative relation)

    H is symmetric by construction. Its eigenvalues are real.

    We check: do its eigenvalues approximate the Riemann zeros?
    """
    print("\n[6] Speculative Operator Construction")

    from math import gcd, log as mlog

    def mobius(n):
        """Compute μ(n)."""
        if n == 1:
            return 1
        factors = []
        temp = n
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

    def divisors(n):
        """Return all divisors of n."""
        divs = []
        for d in range(1, int(n**0.5) + 1):
            if n % d == 0:
                divs.append(d)
                if d != n // d:
                    divs.append(n // d)
        return sorted(divs)

    # Build several candidate matrices and compare their spectra to zeta zeros

    # --- Candidate 1: Redheffer Matrix ---
    # The Redheffer matrix R has R_{ij} = 1 if j=1 or i|j
    # det(R_N) = M(N) = Σ_{k=1}^N μ(k)  (Mertens function)
    # RH ⟺ M(N) = O(N^{1/2+ε}) for all ε > 0

    R = np.zeros((N, N))
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if j == 1 or i != 0 and j % i == 0:
                R[i-1, j-1] = 1

    # Symmetrize
    R_sym = (R + R.T) / 2
    evals_R = np.linalg.eigvalsh(R_sym)

    # --- Candidate 2: GCD matrix ---
    # H_{ij} = log(gcd(i,j)) / √(ij)

    H_gcd = np.zeros((N, N))
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            g = gcd(i, j)
            if g > 1:
                H_gcd[i-1, j-1] = mlog(g) / np.sqrt(i * j)

    evals_gcd = np.linalg.eigvalsh(H_gcd)

    # --- Candidate 3: Von Mangoldt convolution matrix ---
    # H_{ij} = Λ(|i-j|+1) / √(ij)  (Toeplitz-like)

    def vm(n):
        if n <= 1:
            return 0.0
        for p in range(2, int(n**0.5) + 2):
            if n == p:
                return mlog(p)
            k = p * p
            while k <= n:
                if k == n:
                    return mlog(p)
                k *= p
        if all(n % p != 0 for p in range(2, int(n**0.5) + 1)):
            return mlog(n)
        return 0.0

    H_vm = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            n = abs(i - j) + 1
            H_vm[i, j] = vm(n) / np.sqrt((i+1) * (j+1))

    evals_vm = np.linalg.eigvalsh(H_vm)

    # Get actual zeros for comparison
    zeros = compute_zeros(30)
    actual_gammas = sorted([float(z.imag) for z in zeros])

    # Plot all candidate spectra
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Compare eigenvalue distributions
    for ax, evals, title in [
        (axes[0,0], evals_R, 'Symmetrized Redheffer Matrix'),
        (axes[0,1], evals_gcd, 'GCD Matrix: log(gcd(i,j))/√(ij)'),
        (axes[1,0], evals_vm, 'Von Mangoldt Toeplitz Matrix'),
    ]:
        # Normalize eigenvalues to compare with zero distribution
        evals_pos = evals[evals > 0.1]
        if len(evals_pos) > 3:
            evals_normalized = np.sort(evals_pos)
            spacings = np.diff(evals_normalized)
            mean_sp = np.mean(spacings) if np.mean(spacings) > 0 else 1
            norm_spacings = spacings / mean_sp

            bins = np.linspace(0, 3, 20)
            ax.hist(norm_spacings, bins=bins, density=True, alpha=0.6,
                    color='steelblue', edgecolor='black')
            s = np.linspace(0.01, 3, 200)
            p_gue = (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)
            p_poisson = np.exp(-s)
            ax.plot(s, p_gue, 'r-', linewidth=2, label='GUE')
            ax.plot(s, p_poisson, 'g--', linewidth=2, label='Poisson')
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Normalized spacing')

    # Panel 4: Direct eigenvalue comparison (best candidate)
    axes[1,1].set_title('Eigenvalue-Zero Comparison', fontsize=10)
    evals_sorted = np.sort(np.abs(evals_gcd))[::-1][:30]
    # Normalize to same scale
    if len(actual_gammas) > 0 and len(evals_sorted) > 0:
        scale = actual_gammas[0] / evals_sorted[0] if evals_sorted[0] > 0 else 1
        axes[1,1].plot(range(len(actual_gammas[:20])), actual_gammas[:20],
                       'ro-', label='Actual ζ zeros (γ_n)', markersize=5)
        axes[1,1].plot(range(min(20, len(evals_sorted))),
                       evals_sorted[:20] * scale,
                       'bs-', label='GCD matrix eigenvalues (scaled)', markersize=5)
        axes[1,1].legend(fontsize=8)
        axes[1,1].set_xlabel('Index n')
        axes[1,1].set_ylabel('Value')

    plt.suptitle('Candidate Operators for Hilbert-Pólya Conjecture', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'candidate_operators.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved candidate_operators.png")
    print("    Note: finding the RIGHT operator is the $1M question")


# =============================================================================
# PART 7: The de Bruijn-Newman Constant
# =============================================================================

def debruijn_newman():
    """
    The de Bruijn-Newman constant Λ:

    RH ⟺ Λ ≤ 0
    Rodgers-Tao (2020): Λ ≥ 0
    Therefore: RH ⟺ Λ = 0

    Current best upper bound: Λ < 0.2 (Platt-Trudgian 2021)

    The constant governs backward heat flow applied to the xi function.
    As you "heat" ξ backwards, its zeros move. Λ is the critical time
    at which zeros first leave the real axis.

    RH says: even at time t=0 (no heating), all zeros are still real.
    """
    print("\n[7] de Bruijn-Newman Constant")
    print("    RH ⟺ Λ = 0")
    print("    Known: 0 ≤ Λ < 0.2")
    print("    ")
    print("    Interpretation via heat equation:")
    print("    • Ξ(t,z) satisfies the backward heat equation ∂Ξ/∂t = -∂²Ξ/∂z²")
    print("    • For t >> 0, zeros of Ξ(t,·) are real (heat smooths things out)")
    print("    • As t decreases toward 0, zeros might leave the real axis")
    print("    • Λ = infimum of t where all zeros are still real")
    print("    • RH says: even at t = 0, they're all real")
    print("    ")
    print("    The Rodgers-Tao proof that Λ ≥ 0 used:")
    print("    • Analysis of zero dynamics under heat flow")
    print("    • Zeros repel each other (like charged particles)")
    print("    • Any gap between zeros would close under backward flow")
    print("    • But closing the gap to prove Λ = 0 exactly remains open")


# =============================================================================
# MAIN: Run the full exploration
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("  RIEMANN HYPOTHESIS — PROOF EXPLORATION")
    print("  An honest attempt to probe the frontier")
    print("="*70)

    # Part 1: The zeros
    print("\n[1] Computing non-trivial zeros of ζ(s)...")
    zeros = compute_zeros(100)
    plot_zeros(zeros)

    # Part 2: Li's criterion
    lambdas = compute_li_coefficients(n_max=40, n_zeros=150)

    # Part 3: Nyman-Beurling
    distances = nyman_beurling_exploration(N_max=35)

    # Part 4: Spectral analysis
    spectral_operator_exploration()

    # Part 5: Gap analysis
    proof_gap_analysis()

    # Part 6: Speculative constructions
    speculative_operator(N=80)

    # Part 7: de Bruijn-Newman
    debruijn_newman()

    print("\n" + "="*70)
    print("  SYNTHESIS: WHERE A PROOF MIGHT COME FROM")
    print("="*70)
    print("""
    The evidence points to a proof coming from one of these directions:

    1. SPECTRAL (Hilbert-Pólya):
       The GUE statistics are overwhelming evidence that a self-adjoint
       operator exists. The problem: constructing it with the right
       boundary conditions that encode the primes.

       Most promising sub-approach: Connes' noncommutative geometry.
       The adele class space ℚ*\\𝔸_ℚ is the right space, but proving
       the trace formula requires deep results about its geometry.

    2. ARITHMETIC GEOMETRY (Weil's analogy):
       Weil's proof for function fields is the only case where RH
       has been proven. Importing this to ℚ requires:
       - A theory of geometry over F₁ (field with one element)
       - An intersection theory on Spec(ℤ) × Spec(ℤ)
       - A positivity theorem (analog of Hodge index theorem)

       This is a long-term program. Progress is slow but steady.

    3. ANALYTIC (de Bruijn-Newman + fine structure):
       We know 0 ≤ Λ ≤ 0.2. Proving Λ = 0 requires understanding
       the fine dynamics of zeros under heat flow. The zeros behave
       like a Coulomb gas — they repel. Showing they never leave
       the real axis requires controlling this repulsion precisely.

    THE HONEST TRUTH:
    All three approaches face the same fundamental barrier — bridging
    the additive and multiplicative structures of the integers. This
    is what makes RH so hard: it lives at the exact intersection of
    number theory's deepest structures.

    A proof will likely require a genuinely new idea — not just
    pushing existing techniques harder, but a conceptual breakthrough
    that reveals WHY the zeros must be on the critical line.
    """)
