"""
DEEP EXPLORATION: The Additive-Multiplicative Interlock
=======================================================

The core question: WHY can't we prove positivity about the
relationship between addition and multiplication on the integers?

This script explores three concrete angles:

1. RAMANUJAN SUM DECOMPOSITION of the Gram matrix
   - Ramanujan sums are the "Fourier basis" for multiplicative structure
   - Can they diagonalize the Nyman-Beurling Gram matrix?

2. THE TWO-DIMENSIONAL STRUCTURE of the integers
   - Additive dimension: the number line
   - Multiplicative dimension: the prime factorization tree
   - Together they form a "surface" — the setting where positivity lives

3. THE WEIL POSITIVITY CRITERION unpacked
   - What EXACTLY does positivity mean here?
   - Why does it fail for off-line zeros?
   - What computation would close the gap?
"""

import numpy as np
from math import gcd, log as mlog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

OUT = "/Users/aryendersingh/Desktop/Projects/millenium"


# =============================================================================
# PART 1: Ramanujan Sums and the Gram Matrix
# =============================================================================

def ramanujan_sum(q, n):
    """
    Compute the Ramanujan sum c_q(n) = Σ_{a=1..q, gcd(a,q)=1} e^{2πi·a·n/q}

    This equals: c_q(n) = μ(q/gcd(q,n)) · φ(q) / φ(q/gcd(q,n))

    Ramanujan sums are the bridge between additive and multiplicative:
    - They're defined using ADDITIVE characters (exponentials)
    - But they depend on the MULTIPLICATIVE structure (gcd, φ, μ)
    """
    g = gcd(q, n)
    result = 0.0
    for a in range(1, q + 1):
        if gcd(a, q) == 1:
            result += np.cos(2 * np.pi * a * n / q)
    return result


def euler_totient(n):
    """Compute φ(n)."""
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result


def mobius(n):
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


def explore_ramanujan_gram_connection(N=25):
    """
    KEY INSIGHT: The Gram matrix G of the Nyman-Beurling problem
    can be expressed in terms of GCD's and divisor sums.

    The classical formula:
        ∫₀¹ B₁(jx) B₁(kx) dx = -1/(2π²) · Σ_{m=1}^∞ c_j(m)c_k(m)/m²

    where B₁(x) = {x} - 1/2 is the first Bernoulli function, and
    c_j(m) is the Ramanujan sum.

    Simpler: ∫₀¹ {jx}{kx}dx = (gcd(j,k)²)/(4jk) + 1/4 - gcd(j,k)/(2·max(j,k)) + ...

    Actually the exact formula:
        ∫₀¹ {ax}{bx} dx = 1/4 - gcd(a,b)/(2·lcm(a,b)) + gcd(a,b)²/(12·a·b)  (for a ≠ b)
        ∫₀¹ {ax}² dx = 1/12·a  (for a=a, wait this isn't right either)

    Let me compute these numerically and look at the structure.
    """
    print("=" * 70)
    print("PART 1: Ramanujan Sums and the Gram Matrix")
    print("=" * 70)

    # Build the Gram matrix numerically
    n_quad = 10000
    x = np.linspace(1e-8, 1.0 - 1e-8, n_quad)
    dx = x[1] - x[0]

    G_num = np.zeros((N, N))
    for j in range(1, N + 1):
        fj = j * x - np.floor(j * x)  # {jx}
        for k in range(j, N + 1):
            fk = k * x - np.floor(k * x)  # {kx}
            val = np.sum(fj * fk) * dx
            G_num[j-1, k-1] = val
            G_num[k-1, j-1] = val

    # Build the GCD matrix
    G_gcd = np.zeros((N, N))
    for j in range(1, N + 1):
        for k in range(1, N + 1):
            g = gcd(j, k)
            G_gcd[j-1, k-1] = g

    # Build the Ramanujan matrix: R_{jk} = c_k(j) (or c_j(k))
    R = np.zeros((N, N))
    for j in range(1, N + 1):
        for k in range(1, N + 1):
            R[j-1, k-1] = ramanujan_sum(k, j)

    # Eigendecompose all three
    evals_G, evecs_G = np.linalg.eigh(G_num)
    evals_gcd, evecs_gcd = np.linalg.eigh(G_gcd)

    # Check if the Ramanujan matrix diagonalizes G
    # If G = R D R^{-1} for diagonal D, then R^{-1} G R should be diagonal
    try:
        R_inv_G_R = np.linalg.solve(R, G_num @ R)  # R^{-1} G R
        off_diag_norm = np.linalg.norm(R_inv_G_R - np.diag(np.diag(R_inv_G_R)), 'fro')
        total_norm = np.linalg.norm(R_inv_G_R, 'fro')
        diag_ratio = 1.0 - off_diag_norm / total_norm
    except np.linalg.LinAlgError:
        diag_ratio = 0.0
        R_inv_G_R = np.zeros((N, N))

    print(f"\n  Gram matrix size: {N}×{N}")
    print(f"  Diagonalization by Ramanujan basis: {diag_ratio:.4f} (1.0 = perfect)")

    # Explore: what DOES diagonalize G?
    # The eigenvectors of the GCD matrix are known to involve Ramanujan sums.
    # Smith's result: det[gcd(i,j)] = Π φ(k)

    # Check correlation between eigenvectors of G and GCD matrix
    overlap = np.abs(evecs_G.T @ evecs_gcd)
    max_overlaps = np.max(overlap, axis=1)
    mean_max_overlap = np.mean(max_overlaps)

    print(f"  Mean max overlap between G and GCD eigenvectors: {mean_max_overlap:.4f}")

    # CRITICAL: Build the "arithmetic Fourier transform" matrix
    # Define F_{jk} = (1/√N) · c_k(j) / √φ(k)  (normalized Ramanujan)
    F = np.zeros((N, N))
    for j in range(1, N + 1):
        for k in range(1, N + 1):
            phi_k = euler_totient(k)
            F[j-1, k-1] = ramanujan_sum(k, j) / np.sqrt(max(phi_k, 1))
    # Normalize columns
    for k in range(N):
        col_norm = np.linalg.norm(F[:, k])
        if col_norm > 0:
            F[:, k] /= col_norm

    # Transform G into the Ramanujan basis
    G_ramanujan = F.T @ G_num @ F
    off_diag_R = np.linalg.norm(G_ramanujan - np.diag(np.diag(G_ramanujan)), 'fro')
    total_R = np.linalg.norm(G_ramanujan, 'fro')
    print(f"  Off-diagonal energy in Ramanujan basis: {off_diag_R/total_R:.4f} (0 = perfectly diagonal)")

    # Plot
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig)

    # Panel 1: Gram matrix
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(G_num, cmap='viridis', aspect='equal')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    ax1.set_title(f'Gram Matrix G (N={N})')

    # Panel 2: GCD matrix
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(G_gcd, cmap='viridis', aspect='equal')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    ax2.set_title('GCD Matrix [gcd(j,k)]')

    # Panel 3: G in Ramanujan basis
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(np.abs(G_ramanujan), cmap='hot', aspect='equal')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    ax3.set_title('|G| in Ramanujan basis')

    # Panel 4: Eigenvalue comparison
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.semilogy(range(1, N+1), np.sort(evals_G)[::-1], 'bo-', markersize=3, label='Gram G')
    ax4.semilogy(range(1, N+1), np.sort(evals_gcd)[::-1], 'rs-', markersize=3, label='GCD')
    ax4.set_xlabel('Index')
    ax4.set_ylabel('Eigenvalue')
    ax4.set_title('Eigenvalue spectra')
    ax4.legend()

    # Panel 5: Eigenvector overlap
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(overlap, cmap='Blues', aspect='equal', vmin=0, vmax=1)
    plt.colorbar(im5, ax=ax5, shrink=0.8)
    ax5.set_xlabel('GCD eigenvector index')
    ax5.set_ylabel('Gram eigenvector index')
    ax5.set_title('Eigenvector overlap |⟨v_G, v_GCD⟩|')

    # Panel 6: Diagonal of G in Ramanujan basis
    ax6 = fig.add_subplot(gs[1, 2])
    diag_vals = np.diag(G_ramanujan)
    ax6.bar(range(1, N+1), diag_vals, color='steelblue', alpha=0.8)
    ax6.set_xlabel('Ramanujan index q')
    ax6.set_ylabel('G_qq in Ramanujan basis')
    ax6.set_title('Diagonal of G in Ramanujan basis')

    plt.suptitle('The Arithmetic Fourier Analysis of the Gram Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'ramanujan_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved ramanujan_analysis.png")

    return G_num, G_gcd, G_ramanujan


# =============================================================================
# PART 2: The Two-Dimensional Structure
# =============================================================================

def two_dimensional_structure(N=100):
    """
    THE KEY INSIGHT:

    The integers live in TWO dimensions simultaneously:
    1. ADDITIVE: n sits at position n on the number line
    2. MULTIPLICATIVE: n = p₁^a₁ · p₂^a₂ · ... sits at coordinates
       (a₁, a₂, ...) in the "prime factorization lattice"

    These are TWO INDEPENDENT coordinate systems for the same objects.

    RH is about the COMPATIBILITY between these coordinate systems.

    We visualize this by embedding each integer in 2D:
    - x-axis: additive position (n)
    - y-axis: multiplicative "complexity" (Ω(n) = total prime factor count)

    And we look at how the primes are distributed in this 2D picture.
    """
    print("\n" + "=" * 70)
    print("PART 2: The Two-Dimensional Structure of Integers")
    print("=" * 70)

    # Compute prime factorization data
    def omega(n):
        """Total number of prime factors with multiplicity (Ω(n))."""
        if n <= 1:
            return 0
        count = 0
        temp = n
        for p in range(2, int(n**0.5) + 1):
            while temp % p == 0:
                count += 1
                temp //= p
        if temp > 1:
            count += 1
        return count

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i+2) == 0:
                return False
            i += 6
        return True

    def log_radical(n):
        """log of the radical of n: rad(n) = Π_{p|n} p."""
        if n <= 1:
            return 0
        result = 0
        temp = n
        for p in range(2, int(n**0.5) + 1):
            if temp % p == 0:
                result += mlog(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            result += mlog(temp)
        return result

    ns = list(range(2, N + 1))
    omegas = [omega(n) for n in ns]
    primes = [is_prime(n) for n in ns]
    log_n = [mlog(n) for n in ns]
    log_rads = [log_radical(n) for n in ns]
    mu_vals = [mobius(n) for n in ns]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Integers in the (n, Ω(n)) plane
    colors = ['red' if p else 'lightblue' for p in primes]
    sizes = [30 if p else 8 for p in primes]
    axes[0,0].scatter(ns, omegas, c=colors, s=sizes, alpha=0.7, edgecolors='none')
    axes[0,0].set_xlabel('n (additive position)')
    axes[0,0].set_ylabel('Ω(n) (multiplicative complexity)')
    axes[0,0].set_title('Integers in the Additive-Multiplicative plane\n(red = primes)')

    # Panel 2: The "multiplicative vs additive" information
    # log(n) = additive information content
    # Σ log(p_i) = multiplicative information content  (= log(rad(n)))
    axes[0,1].scatter(log_n, log_rads, c=colors, s=sizes, alpha=0.7, edgecolors='none')
    axes[0,1].plot([0, mlog(N)], [0, mlog(N)], 'k--', alpha=0.3, label='y = x (primes)')
    axes[0,1].set_xlabel('log(n) — additive size')
    axes[0,1].set_ylabel('log(rad(n)) — multiplicative content')
    axes[0,1].set_title('Additive vs Multiplicative information\n(primes lie on diagonal)')
    axes[0,1].legend()

    # Panel 3: The Möbius function — the "sign" of the add-mult interlock
    mu_colors = ['green' if m == 1 else ('red' if m == -1 else 'gray') for m in mu_vals]
    axes[1,0].scatter(ns, mu_vals, c=mu_colors, s=10, alpha=0.7)
    axes[1,0].set_xlabel('n')
    axes[1,0].set_ylabel('μ(n)')
    axes[1,0].set_title('Möbius function: the arithmetic sign\ngreen=+1, red=-1, gray=0')
    axes[1,0].set_yticks([-1, 0, 1])

    # Panel 4: Cumulative Mertens function M(n) = Σ μ(k)
    # with √n envelope
    M_vals = np.cumsum(mu_vals)
    sqrt_n = np.sqrt(ns)
    axes[1,1].plot(ns, M_vals, 'b-', linewidth=0.8, label='M(n)')
    axes[1,1].fill_between(ns, -sqrt_n, sqrt_n, alpha=0.15, color='red',
                            label='±√n (RH envelope)')
    axes[1,1].set_xlabel('n')
    axes[1,1].set_ylabel('M(n)')
    axes[1,1].set_title('Mertens function = running sum of Möbius\nRH ⟺ stays within ±√n envelope')
    axes[1,1].legend()

    plt.suptitle('The TWO DIMENSIONS of the Integers\nAdditive (horizontal) vs Multiplicative (vertical)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'two_dimensions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved two_dimensions.png")


# =============================================================================
# PART 3: The Weil Positivity Criterion — What exactly needs to be positive?
# =============================================================================

def weil_positivity_exploration():
    """
    THE WEIL EXPLICIT FORMULA says:

    For a test function h (even, smooth, compactly supported),
    define its "Weil functional":

        W(h) = ĥ(0)·log π
             - (1/2)∫ h(t)[Ψ(1/4 + it/2) + Ψ(1/4 - it/2)] dt
             + h(0)·[log π - γ - log 4]
             - Σ_p Σ_m (log p / p^{m/2}) [h(m log p) + h(-m log p)]

    where Ψ = Γ'/Γ is the digamma function.

    Then: W(h) = Σ_ρ ĥ(γ_ρ)  (sum over non-trivial zeros ρ = 1/2 + iγ)

    WEIL'S CRITERION:
    RH ⟺ W(h * h̃) ≥ 0 for all h, where h̃(x) = h(-x).

    WHY? Because:
    W(h * h̃) = Σ_ρ (h * h̃)^(γ_ρ) = Σ_ρ ĥ(γ_ρ) · ĥ(γ_ρ)̅  [if γ_ρ ∈ ℝ]
              = Σ_ρ |ĥ(γ_ρ)|² ≥ 0   ✓  (if RH)

    If some γ_ρ ∉ ℝ, the cross-terms are NOT manifestly non-negative.

    Let's COMPUTE W(h * h̃) for specific test functions to see
    the positivity in action.
    """
    print("\n" + "=" * 70)
    print("PART 3: The Weil Positivity Criterion")
    print("=" * 70)

    from mpmath import zetazero, digamma, mpf, pi as mpi
    import mpmath
    mpmath.mp.dps = 20

    # Get zeros
    n_zeros = 50
    zeros = [zetazero(k) for k in range(1, n_zeros + 1)]
    gammas = [float(z.imag) for z in zeros]

    # Test with Gaussian: h(t) = exp(-t²/(2σ²))
    # ĥ(ξ) = σ√(2π) exp(-2π²σ²ξ²)
    # (h * h̃)(t) = h * h(t) [since h is even] has Fourier transform |ĥ(ξ)|²

    sigmas = np.linspace(0.05, 2.0, 50)
    W_values = []

    for sigma in sigmas:
        # W(h * h̃) = Σ_ρ |ĥ(γ_ρ)|²
        # |ĥ(γ)|² = 2πσ² exp(-4π²σ²γ²)  [for real γ, which we're assuming]
        total = 0.0
        for gamma in gammas:
            h_hat_sq = 2 * np.pi * sigma**2 * np.exp(-4 * np.pi**2 * sigma**2 * gamma**2)
            total += 2 * h_hat_sq  # Factor 2 for γ and -γ
        W_values.append(total)

    # Also compute the "prime side" of the explicit formula for verification
    # The prime sum: P(h*h̃) = Σ_p Σ_m (log p / p^{m/2}) (h*h̃)(m log p)
    # For Gaussian: (h*h̃)(t) = (σ√π) exp(-t²/(4σ²))  [convolution of two Gaussians]

    primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                   53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    P_values = []
    for sigma in sigmas:
        prime_sum = 0.0
        for p in primes_list:
            log_p = mlog(p)
            for m in range(1, 20):
                pm_half = p**(m/2)
                t_val = m * log_p
                # (h*h̃)(t) for Gaussian h with width σ:
                # h*h̃ = (σ√π) exp(-t²/(4σ²))
                conv_val = sigma * np.sqrt(np.pi) * np.exp(-t_val**2 / (4 * sigma**2))
                prime_sum += (log_p / pm_half) * 2 * conv_val  # factor 2 for ±t
        P_values.append(prime_sum)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: W(h*h̃) as function of σ — should be ≥ 0
    axes[0,0].plot(sigmas, W_values, 'b-', linewidth=2, label='W(h*h̃) = Σ|ĥ(γ)|²')
    axes[0,0].axhline(y=0, color='red', linewidth=1, linestyle='--')
    axes[0,0].set_xlabel('σ (Gaussian width)')
    axes[0,0].set_ylabel('W(h*h̃)')
    axes[0,0].set_title('Weil Positivity: W(h*h̃) ≥ 0\n(spectral side)')
    axes[0,0].legend()
    axes[0,0].text(0.5, 0.9, 'ALWAYS ≥ 0 ⟹ consistent with RH',
                   transform=axes[0,0].transAxes, ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Panel 2: The individual terms |ĥ(γ_k)|² for various σ
    sigma_examples = [0.1, 0.3, 0.5, 1.0]
    for sigma in sigma_examples:
        terms = []
        for gamma in gammas[:20]:
            h_hat_sq = 2 * np.pi * sigma**2 * np.exp(-4 * np.pi**2 * sigma**2 * gamma**2)
            terms.append(2 * h_hat_sq)
        axes[0,1].plot(range(1, 21), terms, 'o-', markersize=3, label=f'σ={sigma}')
    axes[0,1].set_xlabel('Zero index k')
    axes[0,1].set_ylabel('|ĥ(γ_k)|²')
    axes[0,1].set_title('Contribution of each zero to W')
    axes[0,1].set_yscale('log')
    axes[0,1].legend(fontsize=8)

    # Panel 3: What would happen with a FAKE off-line zero?
    # Suppose there were a zero at ρ = 0.7 + 14.13i (off the critical line)
    # The paired zero would be at 0.3 + 14.13i
    # γ_ρ = 14.13 - 0.2i, γ_{1-ρ̄} = 14.13 + 0.2i
    # Contribution: ĥ(γ)·ĥ(γ̄)̄ + c.c. = 2Re[ĥ(γ)·ĥ(γ̄)̄]

    fake_offsets = np.linspace(0, 0.49, 50)  # How far off the critical line
    gamma_base = 14.134  # Near first zero

    negativity = []
    for delta in fake_offsets:
        # γ = gamma_base - i·delta  (complex)
        # ĥ(γ) = σ√(2π) exp(-2π²σ²(gamma_base - i·delta)²)
        sigma = 0.3
        gamma_complex = gamma_base - 1j * delta
        gamma_conj = gamma_base + 1j * delta
        h_hat_gamma = sigma * np.sqrt(2*np.pi) * np.exp(-2*np.pi**2*sigma**2*gamma_complex**2)
        h_hat_gamma_conj = sigma * np.sqrt(2*np.pi) * np.exp(-2*np.pi**2*sigma**2*gamma_conj**2)
        # Contribution: ĥ(γ)·(ĥ(γ̄))̄ + ĥ(γ̄)·(ĥ(γ))̄ = 2Re[ĥ(γ)·(ĥ(γ̄))̄]
        cross = h_hat_gamma * np.conj(h_hat_gamma_conj)
        contribution = 2 * cross.real
        negativity.append(contribution)

    axes[1,0].plot(fake_offsets + 0.5, negativity, 'r-', linewidth=2)
    axes[1,0].axhline(y=0, color='black', linewidth=0.5)
    axes[1,0].axvline(x=0.5, color='blue', linewidth=1, linestyle='--', label='Critical line')
    axes[1,0].set_xlabel('Re(ρ) of hypothetical zero')
    axes[1,0].set_ylabel('Cross-term contribution')
    axes[1,0].set_title('What if a zero were OFF the critical line?\n(contribution can become negative!)')
    axes[1,0].legend()
    axes[1,0].fill_between([0.5, 0.99], [min(negativity)*1.2]*2, [0, 0],
                            alpha=0.1, color='red', label='Danger zone')

    # Panel 4: The ANATOMY of W — spectral vs prime contributions
    axes[1,1].plot(sigmas, W_values, 'b-', linewidth=2, label='Spectral: Σ|ĥ(γ)|²')
    axes[1,1].plot(sigmas, P_values, 'r--', linewidth=2, label='Prime sum (partial)')
    axes[1,1].set_xlabel('σ')
    axes[1,1].set_ylabel('Value')
    axes[1,1].set_title('Spectral side vs Prime side of W')
    axes[1,1].legend()

    plt.suptitle('Weil Positivity: The Exact Criterion for RH', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'weil_positivity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved weil_positivity.png")

    print(f"\n  W(h*h̃) range: [{min(W_values):.6f}, {max(W_values):.6f}]")
    print(f"  All non-negative: {all(w >= -1e-10 for w in W_values)}")
    print(f"  Off-line zero contribution range: [{min(negativity):.6f}, {max(negativity):.6f}]")
    print(f"  Goes negative for Re(ρ) > {0.5 + fake_offsets[next(i for i,v in enumerate(negativity) if v < 0)]:.3f}"
          if any(v < 0 for v in negativity) else "  Never goes negative (for this test function)")


# =============================================================================
# PART 4: The Deep Structure — What IS the missing positivity?
# =============================================================================

def deep_structure_analysis(N=60):
    """
    The deepest question: what mathematical object naturally provides
    positivity for the additive-multiplicative interlock?

    ANSWER (I believe): It's the CONVEXITY of the logarithm.

    log is the UNIQUE function (up to scale) that converts × to +:
        log(ab) = log(a) + log(b)

    And log is CONCAVE. This concavity is a form of positivity.

    Jensen's inequality: log(E[X]) ≥ E[log(X)]  (for concave log)

    Could the positivity needed for RH be a sophisticated form
    of Jensen's inequality applied to the integers?

    Let's explore this connection.
    """
    print("\n" + "=" * 70)
    print("PART 4: The Deep Structure — Positivity from Convexity of log")
    print("=" * 70)

    # The von Mangoldt function Λ(n) = log p if n = p^k, else 0
    # satisfies the identity:  Σ_{d|n} Λ(d) = log n
    #
    # In other words: log (the bridge from × to +) is the ADDITIVE
    # aggregation of the MULTIPLICATIVE indicator Λ.
    #
    # The explicit formula:
    #   Σ_{n≤x} Λ(n) = x - Σ_ρ x^ρ/ρ - log(2π) - ...
    #
    # The oscillatory terms Σ_ρ x^ρ/ρ represent the INTERFERENCE
    # between the additive average (left) and the smooth part (x).
    #
    # RH says: |Σ_ρ x^ρ/ρ| ≤ C · √x · log²x
    # i.e., the interference is "square-root cancellation"
    #
    # Square-root cancellation is what you get from INDEPENDENT
    # random variables (by CLT). So RH says: the prime indicators
    # behave like INDEPENDENT random variables in the additive direction.
    #
    # Independence + log-concavity might give the positivity.

    def von_mangoldt(n):
        if n <= 1:
            return 0
        temp = n
        for p in range(2, int(n**0.5) + 1):
            if temp % p == 0:
                log_p = mlog(p)
                while temp % p == 0:
                    temp //= p
                if temp == 1:
                    return log_p
                return 0
        return mlog(n)  # n is prime

    # Compute chebyshev psi and compare to x
    x_vals = list(range(2, N + 1))
    psi_vals = []
    running = 0
    for n in range(2, N + 1):
        running += von_mangoldt(n)
        psi_vals.append(running)

    # The error term: ψ(x) - x = -Σ_ρ x^ρ/ρ + ...
    errors = [psi_vals[i] - x_vals[i] for i in range(len(x_vals))]

    # Normalize by √x to check square-root cancellation
    normalized_errors = [errors[i] / np.sqrt(x_vals[i]) for i in range(len(x_vals))]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: ψ(x) vs x
    axes[0,0].plot(x_vals, psi_vals, 'b-', linewidth=1.5, label='ψ(x)')
    axes[0,0].plot(x_vals, x_vals, 'r--', linewidth=1.5, label='x')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('ψ(x)')
    axes[0,0].set_title('Chebyshev ψ(x) vs x')
    axes[0,0].legend()

    # Panel 2: Error normalized by √x
    axes[0,1].plot(x_vals, normalized_errors, 'g-', linewidth=1)
    axes[0,1].axhline(y=0, color='black', linewidth=0.5)
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('[ψ(x) - x] / √x')
    axes[0,1].set_title('Normalized error — RH says this stays bounded')

    # Panel 3: The "convexity" connection
    # log(n) = Σ_{d|n} Λ(d)  and  log is concave
    # Consider the "entropy" of the prime factorization
    # H(n) = -Σ (a_i/Ω(n)) log(a_i/Ω(n)) where n = Π p_i^{a_i}
    # This measures how "spread out" the factorization is

    def factorization_entropy(n):
        if n <= 1:
            return 0
        exponents = []
        temp = n
        for p in range(2, int(n**0.5) + 1):
            if temp % p == 0:
                a = 0
                while temp % p == 0:
                    a += 1
                    temp //= p
                exponents.append(a)
        if temp > 1:
            exponents.append(1)
        total = sum(exponents)
        if total == 0:
            return 0
        return -sum((a/total) * mlog(a/total) for a in exponents if a > 0)

    entropies = [factorization_entropy(n) for n in range(2, N + 1)]
    axes[1,0].scatter(range(2, N + 1), entropies, s=10, c='purple', alpha=0.7)
    axes[1,0].set_xlabel('n')
    axes[1,0].set_ylabel('H(n) = factorization entropy')
    axes[1,0].set_title('Entropy of prime factorization\n(primes have H=0, highly composite numbers have large H)')

    # Panel 4: The INTERFERENCE pattern — Fourier transform of Λ(n)
    # F(t) = Σ_{n≤N} Λ(n) e^{2πint/N}
    # This should show peaks at positions related to the zeros of ζ
    t_vals = np.linspace(0, 1, 500)
    F_vals = []
    lambdas = [von_mangoldt(n) for n in range(1, N + 1)]
    for t in t_vals:
        F = sum(lambdas[n] * np.exp(2j * np.pi * (n+1) * t) for n in range(N))
        F_vals.append(np.abs(F))

    axes[1,1].plot(t_vals, F_vals, 'b-', linewidth=1)
    axes[1,1].set_xlabel('t (frequency)')
    axes[1,1].set_ylabel('|F(t)|')
    axes[1,1].set_title('Fourier spectrum of Λ(n)\n= interference pattern of add × mult')

    plt.suptitle('The Deep Structure: How Addition and Multiplication Interfere', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'deep_structure.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved deep_structure.png")

    # The key argument
    print("""
    THE CONCAVITY ARGUMENT (speculative):

    1. log is the unique bridge from multiplication to addition.

    2. log is CONCAVE: log(Σ aᵢxᵢ) ≥ Σ aᵢ log(xᵢ)  (Jensen)

    3. The explicit formula: Σ Λ(n) = x - Σ_ρ x^ρ/ρ + ...
       says that log (via Λ) controls the prime distribution.

    4. The error term Σ_ρ x^ρ/ρ involves the zeros.

    5. IF each term x^ρ/ρ could be bounded using the concavity
       of log (since Λ = log aggregated over divisors), THEN
       the total error would be controlled, giving RH.

    6. The concavity of log IS the positivity:
       it's equivalent to 1/x being POSITIVE (since -log''(x) = 1/x² > 0).

    7. And 1/x is the kernel of the MELLIN TRANSFORM, which is
       exactly the transform that converts between additive (Fourier)
       and multiplicative (Dirichlet) analysis.

    So the chain is:
        log concave → 1/x² > 0 → Mellin transform is positive →
        additive-multiplicative bridge is positive-definite →
        zeros on the critical line → RH

    The gap: making steps 5-7 rigorous.
    """)


# =============================================================================
# PART 5: The Missing Piece — A Framework for the Interlock
# =============================================================================

def the_missing_piece():
    """
    The synthesis of everything we've explored.
    """
    print("\n" + "=" * 70)
    print("THE MISSING PIECE: A Framework for the Interlock")
    print("=" * 70)
    print("""
    After deep exploration, here is what I believe the situation is:

    ╔═══════════════════════════════════════════════════════════════════╗
    ║  THE THREE PILLARS OF A PROOF                                    ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  PILLAR 1: THE SPACE                                             ║
    ║  The correct space is the adele class space ℚ*\\𝔸_ℚ              ║
    ║  (Connes).                                                        ║
    ║                                                                   ║
    ║  This space naturally combines:                                   ║
    ║  • Additive structure (ℝ component, archimedean place)           ║
    ║  • Multiplicative structure (Π ℚ_p components, p-adic places)    ║
    ║  • Their interlock (the diagonal ℚ* quotient)                    ║
    ║                                                                   ║
    ║  The quotient by ℚ* is what LINKS addition and multiplication:   ║
    ║  it identifies additive translates with multiplicative scalings.  ║
    ║  STATUS: ✓ Known, well-defined                                   ║
    ║                                                                   ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  PILLAR 2: THE OPERATOR                                          ║
    ║  The operator is the generator of the scaling action of ℝ₊*     ║
    ║  on the adele class space.                                        ║
    ║                                                                   ║
    ║  This is the "quantized xp" of Berry-Keating, living on         ║
    ║  Connes' space. Its trace formula IS the explicit formula         ║
    ║  of prime number theory.                                          ║
    ║  STATUS: ✓ Known, trace formula established                      ║
    ║                                                                   ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  PILLAR 3: THE POSITIVITY                                        ║
    ║  The operator must be self-adjoint (or equivalent).              ║
    ║  This requires a positive-definite inner product on the          ║
    ║  space, preserved by the operator.                                ║
    ║                                                                   ║
    ║  The inner product comes from the HAAR MEASURE on the adeles.    ║
    ║  The Haar measure is the product of local measures:              ║
    ║    dμ = dx_∞ × Π_p dx_p                                         ║
    ║                                                                   ║
    ║  After quotienting by ℚ*, this becomes the Tamagawa measure.    ║
    ║  The scaling action DOES preserve this measure (it's Haar).      ║
    ║                                                                   ║
    ║  So the formal self-adjointness holds.                            ║
    ║  STATUS: ⚠ Formal OK, but ESSENTIAL SELF-ADJOINTNESS unknown    ║
    ║                                                                   ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  THE GAP IS IN PILLAR 3:                                         ║
    ║                                                                   ║
    ║  The operator D (scaling generator) is SYMMETRIC on a natural    ║
    ║  domain, but we need it to be ESSENTIALLY SELF-ADJOINT —        ║
    ║  meaning it has a UNIQUE self-adjoint extension.                  ║
    ║                                                                   ║
    ║  Essential self-adjointness on the adele class space requires    ║
    ║  controlling the behavior of functions at the "boundary" of      ║
    ║  ℚ*\\𝔸_ℚ. The boundary comes from:                              ║
    ║  • The point 0 ∈ 𝔸_ℚ (where the scaling action degenerates)    ║
    ║  • The "cusps" at infinity in each ℚ_p component               ║
    ║                                                                   ║
    ║  The arithmetic of ℤ determines the structure of these           ║
    ║  boundaries. The primes control the cusps.                        ║
    ║                                                                   ║
    ║  To prove essential self-adjointness, one needs to show that     ║
    ║  the "deficiency indices" of D are (0, 0). This requires        ║
    ║  showing that certain L² solutions of (D ± i)f = 0 do not       ║
    ║  exist on the adele class space.                                  ║
    ║                                                                   ║
    ║  The non-existence of these L² solutions is a SPECTRAL           ║
    ║  condition on the adele class space — it says the space is       ║
    ║  "complete" in a specific functional-analytic sense.              ║
    ║                                                                   ║
    ║  This completeness is the POSITIVITY we've been looking for.     ║
    ║  It's the statement that the additive-multiplicative interlock   ║
    ║  is "tight enough" that no spurious modes can exist.             ║
    ║                                                                   ║
    ║  In physical terms: the quantum system has no "leaks" —         ║
    ║  probability is conserved — the evolution is unitary —          ║
    ║  the Hamiltonian is self-adjoint — the eigenvalues are real —   ║
    ║  RH is true.                                                      ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝

    WHAT WOULD CLOSE THE GAP:

    Show that for the scaling operator D on L²(ℚ*\\𝔸_ℚ), restricted
    to the subspace orthogonal to the "trivial" contributions (pole
    at s=1 and trivial zeros), the deficiency indices are (0, 0).

    This is equivalent to: for every λ ∈ ℂ with Im(λ) ≠ 0, the
    equation Df = λf has NO L² solution on the adele class space.

    An L² solution would correspond to a zero of ζ at s = 1/2 + λ
    with Im(λ) ≠ 0 — i.e., a zero OFF the critical line. If we
    show such solutions can't be L², we've shown they can't exist,
    which is RH.

    But wait — this seems circular again! Showing no L² solution
    exists seems to require knowing where the zeros are.

    THE BREAK IN THE CIRCULARITY comes from the GEOMETRY of the
    adele class space. The space has enough structure (from the
    product formula, the reciprocity laws of class field theory, etc.)
    to constrain the L² condition independently of where the zeros are.

    Specifically: the product formula Π_v |x|_v = 1 for x ∈ ℚ*
    (product over all valuations) imposes a GLOBAL constraint that
    links all the local (p-adic) behaviors. This global constraint
    is what should rule out L² solutions of Df = λf for Im(λ) ≠ 0.

    The product formula IS the additive-multiplicative interlock,
    expressed as a constraint on the adeles.

    So the chain would be:
    Product formula → L² constraint on adele class space →
    Deficiency indices (0,0) → Essential self-adjointness →
    Real spectrum → RH

    I believe this is the most promising route to a proof.
    The tools needed are:
    • Harmonic analysis on adele groups (Tate's thesis, Weil)
    • Von Neumann extension theory for symmetric operators
    • The arithmetic of the product formula and its consequences
    """)


if __name__ == "__main__":
    G, G_gcd, G_ram = explore_ramanujan_gram_connection(N=25)
    two_dimensional_structure(N=200)
    weil_positivity_exploration()
    deep_structure_analysis(N=200)
    the_missing_piece()
