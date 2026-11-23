//! The Schrödinger equation as diffusion with reaction (see analyzer/diffusion_mc.py).
//!
//! Walkers diffuse with a drift along grad(ln g), where g is a guess for the
//! wavefunction, and multiply or die at the rate shift - (H g)/g. The guess is
//! g = prod_i phi(r_i) * exp(u(r_12)), phi(r) = sum over nuclei exp(-zeta |r - R|),
//! u(r) = r / (2 (1 + b r)), for one or two electrons.

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

struct Guess<'a> {
    z: &'a [f64],
    r: &'a [f64],
    electrons: usize,
    zeta: f64,
    b: f64,
    e_nuc: f64,
}

#[derive(Clone, Copy)]
struct Walker {
    x: [f64; 6],
    ln: f64,
    drift: [f64; 6],
    local: f64,
    weight: f64,
}

/// (ln g, grad ln g, (H g)/g) at the electron positions x.
fn evaluate(g: &Guess, x: &[f64; 6]) -> (f64, [f64; 6], f64) {
    let (mut ln, mut grad, mut lap, mut v) = (0.0, [0.0; 6], 0.0, g.e_nuc);
    for i in 0..g.electrons {
        let (mut phi, mut gr, mut second) = (0.0, [0.0; 3], 0.0);
        for (a, z) in g.z.iter().enumerate() {
            let d = [x[3 * i] - g.r[3 * a], x[3 * i + 1] - g.r[3 * a + 1], x[3 * i + 2] - g.r[3 * a + 2]];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let e = (-g.zeta * r).exp();
            phi += e;
            for k in 0..3 {
                gr[k] -= g.zeta * e * d[k] / r;
            }
            second += e * (g.zeta * g.zeta - 2.0 * g.zeta / r);
            v -= z / r;
        }
        let mut square = 0.0;
        for k in 0..3 {
            grad[3 * i + k] = gr[k] / phi;
            square += grad[3 * i + k] * grad[3 * i + k];
        }
        lap += second / phi - square;
        ln += phi.ln();
    }
    if g.electrons == 2 {
        let s = [x[0] - x[3], x[1] - x[4], x[2] - x[5]];
        let r = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
        let q = 1.0 + g.b * r;
        let u1 = 0.5 / (q * q);
        ln += 0.5 * r / q;
        for k in 0..3 {
            grad[k] += u1 * s[k] / r;
            grad[3 + k] -= u1 * s[k] / r;
        }
        lap += 2.0 * (-g.b / (q * q * q) + 2.0 * u1 / r);
        v += 1.0 / r;
    }
    let square: f64 = grad.iter().map(|t| t * t).sum();
    (ln, grad, -0.5 * (lap + square) + v)
}

/// Runs `warm + steps` time steps of length `tau` with `walkers` walkers and writes the
/// energy estimate of each of the last `steps` steps to `out`. The first half of the
/// warm-up has no reaction; the mean (H g)/g over its second half goes to `guess_alone`.
///
/// # Safety
/// `z` holds `n_nuc` charges, `r` three coordinates per nucleus, `out` `steps` values.
#[no_mangle]
pub unsafe extern "C" fn walk_run(
    n_nuc: usize, z: *const f64, r: *const f64, electrons: usize, zeta: f64, b: f64, tau: f64,
    walkers: usize, warm: usize, steps: usize, seed: u64, out: *mut f64, guess_alone: *mut f64,
) -> i32 {
    if electrons == 0 || electrons > 2 || n_nuc == 0 {
        return 1;
    }
    let z = std::slice::from_raw_parts(z, n_nuc);
    let r = std::slice::from_raw_parts(r, 3 * n_nuc);
    let out = std::slice::from_raw_parts_mut(out, steps);
    let mut e_nuc = 0.0;
    for a in 0..n_nuc {
        for c in 0..a {
            let d: f64 = (0..3).map(|k| (r[3 * a + k] - r[3 * c + k]).powi(2)).sum();
            e_nuc += z[a] * z[c] / d.sqrt();
        }
    }
    let g = Guess { z, r, electrons, zeta, b, e_nuc };
    let n = 3 * electrons;
    let mut rng = Rng::new(seed, 0, 0);
    let mut pop: Vec<Walker> = (0..walkers)
        .map(|_| {
            let mut x = [0.0; 6];
            for i in 0..electrons {
                let a = (rng.next() % n_nuc as u64) as usize;
                for k in 0..3 {
                    x[3 * i + k] = r[3 * a + k] + rng.normal();
                }
            }
            let (ln, drift, local) = evaluate(&g, &x);
            Walker { x, ln, drift, local, weight: 1.0 }
        })
        .collect();
    let mut shift = pop.iter().map(|w| w.local).sum::<f64>() / walkers as f64;
    let (mut first, mut first_count) = (0.0, 0usize);
    let sigma = tau.sqrt();
    let mut next = Vec::with_capacity(walkers + walkers / 4);
    for step in 0..warm + steps {
        let react = step >= warm / 2;
        let now = shift;
        // diffusion with drift, accepted or rejected so that g^2 is sampled exactly at any step
        pop.par_chunks_mut(256).enumerate().for_each(|(chunk, part)| {
            let mut rng = Rng::new(seed, step as u64 + 1, chunk as u64 + 1);
            for w in part.iter_mut() {
                let mut x = w.x;
                let mut forward = 0.0;
                for k in 0..n {
                    let kick = sigma * rng.normal();
                    x[k] += tau * w.drift[k] + kick;
                    forward += kick * kick;
                }
                let (ln, drift, local) = evaluate(&g, &x);
                let mut back = 0.0;
                for k in 0..n {
                    back += (w.x[k] - x[k] - tau * drift[k]).powi(2);
                }
                let old = w.local;
                if rng.uniform().ln() < 2.0 * (ln - w.ln) + (forward - back) / (2.0 * tau) {
                    *w = Walker { x, ln, drift, local, weight: 1.0 };
                }
                // reaction: multiply or die at the rate shift - local energy
                w.weight = if react { (-tau * (0.5 * (w.local + old) - now)).exp() } else { 1.0 };
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
        let estimate = pop.iter().map(|w| w.weight * w.local).sum::<f64>() / total;
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
    *guess_alone = if first_count > 0 { first / first_count as f64 } else { f64::NAN };
    0
}
