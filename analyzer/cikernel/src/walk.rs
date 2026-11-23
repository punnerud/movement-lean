//! The Schrödinger equation as diffusion with reaction (see analyzer/diffusion_mc.py).
//!
//! Walkers diffuse with a drift along grad(ln g), where g is a guess for the
//! wavefunction, and multiply or die at the rate shift - (H g)/g. With more than two
//! electrons the wavefunction changes sign, and walkers are kept on one side of the
//! surfaces where the guess is zero (a move across is rejected): the energy is then the
//! lowest one with those surfaces, an upper bound to the true one.

use rayon::prelude::*;

struct Rng([u64; 4]);

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    fn new(seed: u64, a: u64, b: u64) -> Rng {
        let mut s = seed ^ a.wrapping_mul(0xD605_BBB5_8C8A_BBFD) ^ b.wrapping_mul(0xA24B_AED4_963E_E407);
        Rng([splitmix(&mut s), splitmix(&mut s), splitmix(&mut s), splitmix(&mut s)])
    }
    fn next(&mut self) -> u64 {                               // xoshiro256++
        let s = &mut self.0;
        let out = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        out
    }
    fn uniform(&mut self) -> f64 {                            // in (0, 1)
        ((self.next() >> 11) as f64 + 0.5) / 9007199254740992.0
    }
    fn normal(&mut self) -> f64 {
        (-2.0 * self.uniform().ln()).sqrt() * (2.0 * std::f64::consts::PI * self.uniform()).cos()
    }
}

/// Most electrons the walker loop handles (the arrays are fixed in size).
pub const MAX_ELECTRONS: usize = 8;
const DIM: usize = 3 * MAX_ELECTRONS;

/// Most centres when the guess has extra terms (the distances are kept in fixed arrays).
pub const MAX_CENTRES: usize = 16;

/// g = det(up orbitals) * det(down orbitals) * exp(sum over pairs u(r_ij) + extra terms).
/// Orbital k of a spin is sum over centres c[k][A] f_A(r - R_A), with f_A(d) = exp(-zeta_A |d|)
/// (kind 0, s) or that times d_x, d_y or d_z (kinds 1, 2, 3: p functions, which let an orbital
/// lean to one side of a nucleus);
/// u(r) = a r / (1 + b r) with a = 1/2 for opposite spins and 1/4 for equal ones.
///
/// The extra terms are powers of the scaled distances s(r) = r / (1 + scale r), which run from
/// 0 to 1 / scale. Term t has powers (m, n, o) and one coefficient per centre A:
///   n = o = 0:   coeff[t][A] sum over electrons i of s(r_iA)^m
///   m = n = 0:   coeff[t][0] sum over pairs of s(r_ij)^o
///   otherwise:   coeff[t][A] sum over pairs of (s(r_iA)^m s(r_jA)^n + s(r_iA)^n s(r_jA)^m) s(r_ij)^o
/// The last kind lets the pull between two electrons depend on where they are relative to a
/// nucleus. Powers of 1 change the slopes at contact (the cusps) and are normally left out.
struct Guess<'a> {
    z: &'a [f64],
    r: &'a [f64],
    zeta: &'a [f64],
    n_up: usize,
    n_dn: usize,
    c_up: &'a [f64],
    c_dn: &'a [f64],
    b: f64,
    e_nuc: f64,
    powers: &'a [i32],
    coeff: &'a [f64],
    scale: f64,
    kind: &'a [i32],
}

/// s^m and its first and second derivatives with respect to r, for s = r / (1 + scale r).
fn power(r: f64, scale: f64, m: i32) -> (f64, f64, f64) {
    if m == 0 {
        return (1.0, 0.0, 0.0);
    }
    let q = 1.0 + scale * r;
    let (s, s1, s2) = (r / q, 1.0 / (q * q), -2.0 * scale / (q * q * q));
    let low = s.powi(m - 1);
    let lower = if m >= 2 { (m * (m - 1)) as f64 * s.powi(m - 2) } else { 0.0 };
    (low * s, m as f64 * low * s1, lower * s1 * s1 + m as f64 * low * s2)
}

/// Adds the extra terms of the guess to ln g, its gradient and its Laplacian.
fn extra_terms(g: &Guess, x: &[f64; DIM], ln: &mut f64, grad: &mut [f64; DIM], lap: &mut f64) {
    let (n_nuc, n) = (g.z.len(), g.n_up + g.n_dn);
    let mut unit = [[[0.0; 3]; MAX_CENTRES]; MAX_ELECTRONS];
    let mut dist = [[0.0; MAX_CENTRES]; MAX_ELECTRONS];
    for i in 0..n {
        for a in 0..n_nuc {
            let d = [x[3 * i] - g.r[3 * a], x[3 * i + 1] - g.r[3 * a + 1], x[3 * i + 2] - g.r[3 * a + 2]];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            dist[i][a] = r;
            unit[i][a] = [d[0] / r, d[1] / r, d[2] / r];
        }
    }
    for (t, p) in g.powers.chunks(3).enumerate() {
        let (m, nn, o) = (p[0], p[1], p[2]);
        let c = &g.coeff[t * n_nuc..(t + 1) * n_nuc];
        if nn == 0 && o == 0 {
            for a in 0..n_nuc {
                if c[a] == 0.0 {
                    continue;
                }
                for i in 0..n {
                    let (f, f1, f2) = power(dist[i][a], g.scale, m);
                    *ln += c[a] * f;
                    for k in 0..3 {
                        grad[3 * i + k] += c[a] * f1 * unit[i][a][k];
                    }
                    *lap += c[a] * (f2 + 2.0 * f1 / dist[i][a]);
                }
            }
            continue;
        }
        for i in 0..n {
            for j in 0..i {
                let s = [x[3 * i] - x[3 * j], x[3 * i + 1] - x[3 * j + 1], x[3 * i + 2] - x[3 * j + 2]];
                let r = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
                let s = [s[0] / r, s[1] / r, s[2] / r];
                let (q, q1, q2) = power(r, g.scale, o);
                if m == 0 && nn == 0 {
                    *ln += c[0] * q;
                    for k in 0..3 {
                        grad[3 * i + k] += c[0] * q1 * s[k];
                        grad[3 * j + k] -= c[0] * q1 * s[k];
                    }
                    *lap += 2.0 * c[0] * (q2 + 2.0 * q1 / r);
                    continue;
                }
                for a in 0..n_nuc {
                    if c[a] == 0.0 {
                        continue;
                    }
                    let (mi, mi1, mi2) = power(dist[i][a], g.scale, m);
                    let (ni, ni1, ni2) = power(dist[i][a], g.scale, nn);
                    let (mj, mj1, mj2) = power(dist[j][a], g.scale, m);
                    let (nj, nj1, nj2) = power(dist[j][a], g.scale, nn);
                    let p = mi * nj + ni * mj;
                    let (pi, pii) = (mi1 * nj + ni1 * mj, mi2 * nj + ni2 * mj);
                    let (pj, pjj) = (mi * nj1 + ni * mj1, mi * nj2 + ni * mj2);
                    *ln += c[a] * p * q;
                    let (mut ci, mut cj) = (0.0, 0.0);
                    for k in 0..3 {
                        grad[3 * i + k] += c[a] * (pi * q * unit[i][a][k] + p * q1 * s[k]);
                        grad[3 * j + k] += c[a] * (pj * q * unit[j][a][k] - p * q1 * s[k]);
                        ci += unit[i][a][k] * s[k];
                        cj += unit[j][a][k] * s[k];
                    }
                    *lap += c[a] * (q * (pii + 2.0 * pi / dist[i][a] + pjj + 2.0 * pj / dist[j][a])
                        + 2.0 * p * (q2 + 2.0 * q1 / r) + 2.0 * q1 * (pi * ci - pj * cj));
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Walker {
    x: [f64; DIM],
    ln: f64,
    sign: f64,
    drift: [f64; DIM],
    local: f64,
    weight: f64,
}

/// Determinant and inverse of the m x m matrix in `a` (row-major, rows of length MAX_ELECTRONS).
fn invert(a: &mut [[f64; MAX_ELECTRONS]; MAX_ELECTRONS], m: usize) -> (f64, [[f64; MAX_ELECTRONS]; MAX_ELECTRONS]) {
    let mut inv = [[0.0; MAX_ELECTRONS]; MAX_ELECTRONS];
    for (i, row) in inv.iter_mut().enumerate().take(m) {
        row[i] = 1.0;
    }
    let mut det = 1.0;
    for col in 0..m {
        let mut best = col;
        for row in col + 1..m {
            if a[row][col].abs() > a[best][col].abs() {
                best = row;
            }
        }
        if a[best][col] == 0.0 {
            return (0.0, inv);
        }
        if best != col {
            a.swap(best, col);
            inv.swap(best, col);
            det = -det;
        }
        let pivot = a[col][col];
        det *= pivot;
        for k in 0..m {
            a[col][k] /= pivot;
            inv[col][k] /= pivot;
        }
        for row in 0..m {
            if row != col {
                let f = a[row][col];
                if f != 0.0 {
                    for k in 0..m {
                        a[row][k] -= f * a[col][k];
                        inv[row][k] -= f * inv[col][k];
                    }
                }
            }
        }
    }
    (det, inv)
}

/// (ln |g|, sign of g, grad ln |g|, (H g)/g) at the electron positions x.
fn evaluate(g: &Guess, x: &[f64; DIM]) -> (f64, f64, [f64; DIM], f64) {
    let n_nuc = g.z.len();
    let (mut ln, mut sign, mut grad, mut lap, mut v) = (0.0, 1.0, [0.0; DIM], 0.0, g.e_nuc);
    for (first, m, c) in [(0, g.n_up, g.c_up), (g.n_up, g.n_dn, g.c_dn)] {
        if m == 0 {
            continue;
        }
        let mut value = [[0.0; MAX_ELECTRONS]; MAX_ELECTRONS];          // [electron][orbital]
        let mut slope = [[[0.0; 3]; MAX_ELECTRONS]; MAX_ELECTRONS];
        let mut second = [[0.0; MAX_ELECTRONS]; MAX_ELECTRONS];
        for i in 0..m {
            let p = 3 * (first + i);
            for a in 0..n_nuc {
                let d = [x[p] - g.r[3 * a], x[p + 1] - g.r[3 * a + 1], x[p + 2] - g.r[3 * a + 2]];
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let zeta = g.zeta[a];
                let e = (-zeta * r).exp();
                v -= g.z[a] / r;
                let axis = if g.kind.is_empty() { 0 } else { g.kind[a] as usize };
                for k in 0..m {
                    let w = c[k * n_nuc + a] * e;
                    if axis == 0 {
                        value[i][k] += w;
                        for q in 0..3 {
                            slope[i][k][q] -= w * zeta * d[q] / r;
                        }
                        second[i][k] += w * (zeta * zeta - 2.0 * zeta / r);
                    } else if w != 0.0 {
                        let t = d[axis - 1];
                        value[i][k] += w * t;
                        for q in 0..3 {
                            slope[i][k][q] -= w * zeta * t * d[q] / r;
                        }
                        slope[i][k][axis - 1] += w;
                        second[i][k] += w * t * (zeta * zeta - 4.0 * zeta / r);
                    }
                }
            }
        }
        let (det, inv) = invert(&mut value, m);
        if det == 0.0 {
            return (f64::NEG_INFINITY, 0.0, grad, 0.0);
        }
        ln += det.abs().ln();
        sign *= det.signum();
        for i in 0..m {
            let p = 3 * (first + i);
            let mut square = 0.0;
            for q in 0..3 {
                let t: f64 = (0..m).map(|k| slope[i][k][q] * inv[k][i]).sum();
                grad[p + q] = t;
                square += t * t;
            }
            lap += (0..m).map(|k| second[i][k] * inv[k][i]).sum::<f64>() - square;
        }
    }
    let n = g.n_up + g.n_dn;
    for i in 0..n {
        for j in 0..i {
            let s = [x[3 * i] - x[3 * j], x[3 * i + 1] - x[3 * j + 1], x[3 * i + 2] - x[3 * j + 2]];
            let r = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
            let a = if (i < g.n_up) == (j < g.n_up) { 0.25 } else { 0.5 };
            let q = 1.0 + g.b * r;
            let u1 = a / (q * q);
            ln += a * r / q;
            for k in 0..3 {
                grad[3 * i + k] += u1 * s[k] / r;
                grad[3 * j + k] -= u1 * s[k] / r;
            }
            lap += 2.0 * (-2.0 * a * g.b / (q * q * q) + 2.0 * u1 / r);
            v += 1.0 / r;
        }
    }
    if !g.powers.is_empty() {
        extra_terms(g, x, &mut ln, &mut grad, &mut lap);
    }
    let square: f64 = grad[..3 * n].iter().map(|t| t * t).sum();
    (ln, sign, grad, -0.5 * (lap + square) + v)
}

/// g = exp(sum over pairs u_ij(r_ij)) for particles of any mass and charge, with
/// u(r) = (a r + c r^2) / (1 + b r): slope a at contact (the cusp, charge_i charge_j times the
/// reduced mass), slope c / b far away. The force between a pair is charge_i charge_j / r plus
/// `linear`_ij r (a constant pull, as between quarks). For particles that can be told apart, or
/// two of a kind with opposite spins. With `odd` set to a pair (i, j), g is multiplied by
/// z_i - z_j: the lowest state that is odd under reflection, whose zero surface z_i = z_j is
/// exact by symmetry (the P state of two particles).
struct PairGuess<'a> {
    n: usize,
    inv_mass: &'a [f64],
    charge: &'a [f64],
    a: &'a [f64],
    b: &'a [f64],
    c: &'a [f64],
    linear: &'a [f64],
    odd: Option<(usize, usize)>,
}

fn evaluate_pairs(g: &PairGuess, x: &[f64; DIM]) -> (f64, f64, [f64; DIM], f64) {
    let (mut ln, mut grad, mut lap, mut v) = (0.0, [0.0; DIM], [0.0; MAX_ELECTRONS], 0.0);
    for i in 0..g.n {
        for j in 0..i {
            let s = [x[3 * i] - x[3 * j], x[3 * i + 1] - x[3 * j + 1], x[3 * i + 2] - x[3 * j + 2]];
            let r = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
            let (a, b, c) = (g.a[i * g.n + j], g.b[i * g.n + j], g.c[i * g.n + j]);
            let q = 1.0 + b * r;
            let top = a + 2.0 * c * r + b * c * r * r;
            let u1 = top / (q * q);
            let u2 = 2.0 * c / q - 2.0 * b * top / (q * q * q);
            ln += (a * r + c * r * r) / q;
            for k in 0..3 {
                grad[3 * i + k] += u1 * s[k] / r;
                grad[3 * j + k] -= u1 * s[k] / r;
            }
            lap[i] += u2 + 2.0 * u1 / r;
            lap[j] += u2 + 2.0 * u1 / r;
            v += g.charge[i] * g.charge[j] / r + g.linear[i * g.n + j] * r;
        }
    }
    let mut sign = 1.0;
    if let Some((i, j)) = g.odd {
        let z = x[3 * i + 2] - x[3 * j + 2];
        if z == 0.0 {
            return (f64::NEG_INFINITY, 0.0, grad, 0.0);
        }
        ln += z.abs().ln();
        sign = z.signum();
        grad[3 * i + 2] += 1.0 / z;
        grad[3 * j + 2] -= 1.0 / z;
        lap[i] -= 1.0 / (z * z);
        lap[j] -= 1.0 / (z * z);
    }
    let mut kinetic = 0.0;
    for i in 0..g.n {
        let square: f64 = grad[3 * i..3 * i + 3].iter().map(|t| t * t).sum();
        kinetic -= 0.5 * g.inv_mass[i] * (lap[i] + square);
    }
    (ln, sign, grad, kinetic + v)
}

/// The drift of each particle, shortened where it is very large (next to a zero surface of the
/// guess it grows like 1/distance, and a step would overshoot): v becomes
/// v (sqrt(1 + 2 v^2 t) - 1) / (v^2 t) with t = tau / mass, which is v itself for small v^2 t.
fn shorten(drift: &mut [f64; DIM], inv_mass: &[f64], tau: f64) {
    for (i, w) in inv_mass.iter().enumerate() {
        let v2t: f64 = drift[3 * i..3 * i + 3].iter().map(|t| t * t).sum::<f64>() * tau * w;
        if v2t > 1e-12 {
            let f = ((1.0 + 2.0 * v2t).sqrt() - 1.0) / v2t;
            for k in 0..3 {
                drift[3 * i + k] *= f;
            }
        }
    }
}

/// Settings for experiments on the walker loop (see `walk_options`): the rate limit as a multiple
/// of sqrt(particles / tau) (0: no limit), and whether a walker whose move was rejected reacts.
static CLAMP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0x3FC9_9999_9999_999A);   // 0.2
static REACT_REJECTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Changes the settings of the walker loop for the runs that follow. The defaults are 0.2 and 0.
#[no_mangle]
pub extern "C" fn walk_options(clamp: f64, react_rejected: i32) {
    CLAMP.store(clamp.to_bits(), std::sync::atomic::Ordering::Relaxed);
    REACT_REJECTED.store(react_rejected != 0, std::sync::atomic::Ordering::Relaxed);
}

/// The walker loop. `evaluate` gives (ln |g|, sign, grad ln |g|, (H g)/g); particle i diffuses
/// with the constant 1 / (2 mass_i); `start` draws a starting position. Reaction starts at step
/// `react_from`; before it the walkers sample g^2. Returns the mean (H g)/g over the later part
/// of the reaction-free steps, and the walkers as they are at the end.
#[allow(clippy::too_many_arguments)]
fn run<E, S>(
    evaluate: E, mut start: S, inv_mass: &[f64], tau: f64, walkers: usize, warm: usize, steps: usize,
    react_from: usize, seed: u64, out: &mut [f64],
) -> (f64, Vec<Walker>)
where
    E: Fn(&[f64; DIM]) -> (f64, f64, [f64; DIM], f64) + Sync,
    S: FnMut(&mut Rng) -> [f64; DIM],
{
    let particles = inv_mass.len();
    let mut rng = Rng::new(seed, 0, 0);
    let mut pop: Vec<Walker> = (0..walkers)
        .map(|_| loop {
            let x = start(&mut rng);
            let (ln, sign, mut drift, local) = evaluate(&x);
            shorten(&mut drift, inv_mass, tau);
            if sign != 0.0 && local.is_finite() {
                break Walker { x, ln, sign, drift, local, weight: 1.0 };
            }
        })
        .collect();
    let mut shift = pop.iter().map(|w| w.local).sum::<f64>() / walkers as f64;
    let (mut first, mut first_count) = (0.0, 0usize);
    // (H g)/g is infinite where the guess is wrong in an essential way (at its zero surfaces, at a
    // contact with the wrong cusp). A walker there would multiply without bound, so the rate is
    // limited to 0.2 sqrt(particles / tau) around the shift; the limit goes away as tau -> 0.
    let clamp = f64::from_bits(CLAMP.load(std::sync::atomic::Ordering::Relaxed));
    let react_rejected = REACT_REJECTED.load(std::sync::atomic::Ordering::Relaxed);
    let limit = if clamp > 0.0 { clamp * (particles as f64 / tau).sqrt() } else { f64::INFINITY };
    let mut next = Vec::with_capacity(walkers + walkers / 4);
    for step in 0..warm + steps {
        let react = step >= react_from;
        let now = shift;
        // diffusion with drift, accepted or rejected so that g^2 is sampled exactly at any step
        pop.par_chunks_mut(256).enumerate().for_each(|(chunk, part)| {
            let mut rng = Rng::new(seed, step as u64 + 1, chunk as u64 + 1);
            for w in part.iter_mut() {
                let mut x = w.x;
                let mut forward = 0.0;
                for (i, im) in inv_mass.iter().enumerate() {
                    let sigma = (tau * im).sqrt();
                    for k in 3 * i..3 * i + 3 {
                        let kick = sigma * rng.normal();
                        x[k] += tau * im * w.drift[k] + kick;
                        forward += kick * kick / im;
                    }
                }
                let (ln, sign, mut drift, local) = evaluate(&x);
                shorten(&mut drift, inv_mass, tau);
                let mut back = 0.0;
                for (i, im) in inv_mass.iter().enumerate() {
                    for k in 3 * i..3 * i + 3 {
                        back += (w.x[k] - x[k] - tau * im * drift[k]).powi(2) / im;
                    }
                }
                let old = now + (w.local - now).clamp(-limit, limit);
                // a move across a surface where the guess is zero is rejected
                let moved = sign == w.sign && rng.uniform().ln() < 2.0 * (ln - w.ln) + (forward - back) / (2.0 * tau);
                if moved {
                    *w = Walker { x, ln, sign, drift, local, weight: 1.0 };
                }
                let here = now + (w.local - now).clamp(-limit, limit);
                // reaction: multiply or die at the rate shift - local energy. A walker whose move was
                // rejected has not advanced in time and does not react, or one stuck at a singular
                // point of the guess would multiply for ever.
                w.weight = if react && (moved || react_rejected) { (-tau * (0.5 * (here + old) - now)).exp() } else { 1.0 };
            }
        });
        if !react {
            if step >= warm / 4 {
                first += pop.iter().map(|w| w.local).sum::<f64>() / walkers as f64;
                first_count += 1;
            }
            continue;
        }
        let total: f64 = pop.iter().map(|w| w.weight).sum();
        let estimate = pop.iter().map(|w| w.weight * (now + (w.local - now).clamp(-limit, limit))).sum::<f64>() / total;
        if step >= warm {
            out[step - warm] = estimate;
        } else {
            shift += 0.1 * (estimate - shift);
        }
        // hold the population at its size
        next.clear();
        for w in pop.iter() {
            let copies = (w.weight * walkers as f64 / total + rng.uniform()).floor() as usize;
            for _ in 0..copies {
                next.push(*w);
            }
        }
        if next.len() == walkers {
            std::mem::swap(&mut pop, &mut next);
        } else {
            for w in pop.iter_mut() {
                *w = next[(rng.next() % next.len() as u64) as usize];
            }
        }
    }
    (if first_count > 0 { first / first_count as f64 } else { f64::NAN }, pop)
}

/// # Safety
/// As `walk_run`.
#[allow(clippy::too_many_arguments)]
unsafe fn electron_guess<'a>(
    n_nuc: usize, z: *const f64, r: *const f64, zeta: *const f64, n_up: usize, n_dn: usize,
    c_up: *const f64, c_dn: *const f64, b: f64, n_terms: usize, powers: *const i32, coeff: *const f64, scale: f64, kind: *const i32,
) -> Guess<'a> {
    let z = std::slice::from_raw_parts(z, n_nuc);
    let r = std::slice::from_raw_parts(r, 3 * n_nuc);
    let mut e_nuc = 0.0;
    for a in 0..n_nuc {
        for c in 0..a {
            if z[a] * z[c] != 0.0 {                           // a centre with charge 0 only carries functions
                let d: f64 = (0..3).map(|k| (r[3 * a + k] - r[3 * c + k]).powi(2)).sum();
                e_nuc += z[a] * z[c] / d.sqrt();
            }
        }
    }
    Guess {
        z,
        r,
        zeta: std::slice::from_raw_parts(zeta, n_nuc),
        n_up,
        n_dn,
        c_up: std::slice::from_raw_parts(c_up, n_up * n_nuc),
        c_dn: std::slice::from_raw_parts(c_dn, n_dn * n_nuc),
        b,
        e_nuc,
        powers: if n_terms > 0 { std::slice::from_raw_parts(powers, 3 * n_terms) } else { &[] },
        coeff: if n_terms > 0 { std::slice::from_raw_parts(coeff, n_terms * n_nuc) } else { &[] },
        scale,
        kind: if kind.is_null() { &[] } else { std::slice::from_raw_parts(kind, n_nuc) },
    }
}

fn electron_start<'a>(r: &'a [f64], electrons: usize) -> impl FnMut(&mut Rng) -> [f64; DIM] + 'a {
    let n_nuc = r.len() / 3;
    move |rng: &mut Rng| {
        let mut x = [0.0; DIM];
        for i in 0..electrons {
            let a = (rng.next() % n_nuc as u64) as usize;
            for k in 0..3 {
                x[3 * i + k] = r[3 * a + k] + rng.normal();
            }
        }
        x
    }
}

/// Writes positions and (H g)/g of the walkers after a run without reaction to `out_x`
/// (3 * particles per walker) and `out_local`.
fn hand_back(pop: &[Walker], particles: usize, out_x: &mut [f64], out_local: &mut [f64]) {
    for (k, w) in pop.iter().enumerate() {
        out_x[3 * particles * k..3 * particles * (k + 1)].copy_from_slice(&w.x[..3 * particles]);
        out_local[k] = w.local;
    }
}

/// (H g)/g and ln |g| at `count` given positions.
fn locals<E>(evaluate: E, particles: usize, count: usize, x: &[f64], out_local: &mut [f64], out_ln: &mut [f64])
where
    E: Fn(&[f64; DIM]) -> (f64, f64, [f64; DIM], f64) + Sync,
{
    out_local.par_iter_mut().zip(out_ln.par_iter_mut()).enumerate().take(count).for_each(|(k, (local, ln))| {
        let mut p = [0.0; DIM];
        p[..3 * particles].copy_from_slice(&x[3 * particles * k..3 * particles * (k + 1)]);
        let e = evaluate(&p);
        *local = e.3;
        *ln = e.0;
    });
}

/// Electrons around fixed nuclei, no reaction: `steps` steps, then the walkers' positions and
/// (H g)/g. These are samples of g^2, for improving the guess.
///
/// # Safety
/// As `walk_run`; `out_x` holds 3 * electrons values per walker, `out_local` one.
#[no_mangle]
pub unsafe extern "C" fn walk_sample(
    n_nuc: usize, z: *const f64, r: *const f64, zeta: *const f64, n_up: usize, n_dn: usize,
    c_up: *const f64, c_dn: *const f64, b: f64,
    n_terms: usize, powers: *const i32, coeff: *const f64, scale: f64, kind: *const i32, tau: f64,
    walkers: usize, steps: usize, seed: u64, out_x: *mut f64, out_local: *mut f64,
) -> i32 {
    let electrons = n_up + n_dn;
    if electrons == 0 || electrons > MAX_ELECTRONS || n_nuc == 0 || (n_terms > 0 && n_nuc > MAX_CENTRES) {
        return 1;
    }
    let g = electron_guess(n_nuc, z, r, zeta, n_up, n_dn, c_up, c_dn, b, n_terms, powers, coeff, scale, kind);
    let pop = run(|x| evaluate(&g, x), electron_start(g.r, electrons), &vec![1.0; electrons], tau, walkers, steps, 0,
                  usize::MAX, seed, &mut []).1;
    hand_back(&pop, electrons, std::slice::from_raw_parts_mut(out_x, 3 * electrons * walkers),
              std::slice::from_raw_parts_mut(out_local, walkers));
    0
}

/// Electrons around fixed nuclei: (H g)/g and ln |g| at `count` given positions.
///
/// # Safety
/// As `walk_run`; `x` holds 3 * electrons values per position, the outputs one.
#[no_mangle]
pub unsafe extern "C" fn walk_local(
    n_nuc: usize, z: *const f64, r: *const f64, zeta: *const f64, n_up: usize, n_dn: usize,
    c_up: *const f64, c_dn: *const f64, b: f64,
    n_terms: usize, powers: *const i32, coeff: *const f64, scale: f64, kind: *const i32, count: usize, x: *const f64, out_local: *mut f64, out_ln: *mut f64,
) -> i32 {
    let electrons = n_up + n_dn;
    if electrons == 0 || electrons > MAX_ELECTRONS || n_nuc == 0 || (n_terms > 0 && n_nuc > MAX_CENTRES) {
        return 1;
    }
    let g = electron_guess(n_nuc, z, r, zeta, n_up, n_dn, c_up, c_dn, b, n_terms, powers, coeff, scale, kind);
    locals(|x| evaluate(&g, x), electrons, count, std::slice::from_raw_parts(x, 3 * electrons * count),
           std::slice::from_raw_parts_mut(out_local, count), std::slice::from_raw_parts_mut(out_ln, count));
    0
}

/// Electrons around fixed nuclei. Runs `warm + steps` time steps of length `tau` with `walkers`
/// walkers and writes the energy estimate of each of the last `steps` steps to `out`. The first
/// half of the warm-up has no reaction; the mean (H g)/g over its second half goes to `guess_alone`.
///
/// # Safety
/// `z` and `zeta` hold `n_nuc` values, `r` three coordinates per nucleus, `c_up` and `c_dn`
/// `n_up` and `n_dn` rows of `n_nuc` orbital coefficients, `powers` three values and `coeff`
/// `n_nuc` values per extra term, `kind` `n_nuc` values or null (all s), `out` `steps` values.
#[no_mangle]
pub unsafe extern "C" fn walk_run(
    n_nuc: usize, z: *const f64, r: *const f64, zeta: *const f64, n_up: usize, n_dn: usize,
    c_up: *const f64, c_dn: *const f64, b: f64,
    n_terms: usize, powers: *const i32, coeff: *const f64, scale: f64, kind: *const i32, tau: f64,
    walkers: usize, warm: usize, steps: usize, seed: u64, out: *mut f64, guess_alone: *mut f64,
) -> i32 {
    let electrons = n_up + n_dn;
    if electrons == 0 || electrons > MAX_ELECTRONS || n_nuc == 0 || (n_terms > 0 && n_nuc > MAX_CENTRES) {
        return 1;
    }
    let g = electron_guess(n_nuc, z, r, zeta, n_up, n_dn, c_up, c_dn, b, n_terms, powers, coeff, scale, kind);
    let r = g.r;
    let out = std::slice::from_raw_parts_mut(out, steps);
    let start = electron_start(r, electrons);
    *guess_alone = run(|x| evaluate(&g, x), start, &vec![1.0; electrons], tau, walkers, warm, steps, warm / 2, seed, out).0;
    0
}

/// Particles of any mass and charge, nothing held fixed. `a`, `b`, `c` and `linear` are n x n
/// tables (see `PairGuess`); `odd` is -1 or i * n + j for the pair whose z-difference multiplies
/// the guess; the walkers start within `spread` of the origin. Otherwise as `walk_run`.
///
/// # Safety
/// `mass` and `charge` hold `n` values, `a`, `b`, `c`, `linear` `n * n` each, `out` `steps` values.
#[no_mangle]
pub unsafe extern "C" fn walk_particles(
    n: usize, mass: *const f64, charge: *const f64, a: *const f64, b: *const f64, c: *const f64,
    linear: *const f64, odd: i64, spread: f64, tau: f64, walkers: usize, warm: usize, steps: usize, seed: u64, out: *mut f64,
    guess_alone: *mut f64,
) -> i32 {
    if n < 2 || n > MAX_ELECTRONS {
        return 1;
    }
    let inv_mass: Vec<f64> = std::slice::from_raw_parts(mass, n).iter().map(|m| 1.0 / m).collect();
    let g = PairGuess {
        n,
        inv_mass: &inv_mass,
        charge: std::slice::from_raw_parts(charge, n),
        a: std::slice::from_raw_parts(a, n * n),
        b: std::slice::from_raw_parts(b, n * n),
        c: std::slice::from_raw_parts(c, n * n),
        linear: std::slice::from_raw_parts(linear, n * n),
        odd: if odd < 0 { None } else { Some((odd as usize / n, odd as usize % n)) },
    };
    let out = std::slice::from_raw_parts_mut(out, steps);
    let start = |rng: &mut Rng| {
        let mut x = [0.0; DIM];
        for t in x.iter_mut().take(3 * n) {
            *t = spread * rng.normal();
        }
        x
    };
    *guess_alone = run(|x| evaluate_pairs(&g, x), start, &inv_mass, tau, walkers, warm, steps, warm / 2, seed, out).0;
    0
}
