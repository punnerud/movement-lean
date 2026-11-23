// The kernel itself, written for a string type `Bits` chosen by the module that includes this file
// (u64 for up to 64 orbitals, u128 for up to 128). See lib.rs.

use rayon::prelude::*;
use std::collections::HashMap;

pub struct Ci {
    n: usize,
    h: Vec<f64>,
    g: Vec<f64>,
    da: Vec<Bits>,
    db: Vec<Bits>,
    aid: Vec<u32>,
    bid: Vec<u32>,
    diag: Vec<f64>,
    /// for each alpha string: (beta string, determinant index), sorted by beta
    rows: Vec<Vec<(Bits, u32)>>,
    /// for each beta string: (alpha string, determinant index), sorted by alpha
    cols: Vec<Vec<(Bits, u32)>>,
    /// for each alpha string: the listed alpha strings one excitation away (id, hole, particle, sign)
    alpha_singles: Vec<Vec<(u32, u8, u8, f64)>>,
    g_max: f64,
    /// for each pair of orbitals that electrons leave, the pairs they can go to,
    /// strongest coupling first (the heat-bath ordering): see `sorted_couplings`
    opposite: Vec<(u8, u8, f64)>,
    same: Vec<(u8, u8, f64)>,
    plain_search: bool,
    /// partners remembered for the determinants where finding them is costly: see `ci_plan`
    cache_start: Vec<u64>,
    cache_len: Vec<u32>,
    cache_j: Vec<u32>,
    cache_v: Vec<f64>,
}

const NOT_CACHED: u64 = u64::MAX;

const ONE: Bits = 1;
const WIDTH: usize = std::mem::size_of::<Bits>() * 8;
/// 64-bit words per string in the arrays that cross the C interface (lowest word first)
const WORDS: usize = WIDTH / 64;

#[inline]
fn full_mask(n: usize) -> Bits {
    if n >= WIDTH { !0 } else { (ONE << n) - 1 }
}

#[inline]
fn fold(a: Bits) -> u64 {
    let x = a as u128;
    (x as u64) ^ ((x >> 64) as u64)
}

unsafe fn read_strings(p: *const u64, count: usize) -> Vec<Bits> {
    let words = std::slice::from_raw_parts(p, count * WORDS);
    words.chunks_exact(WORDS).map(|w| w.iter().rev().fold(0u128, |x, &part| (x << 64) | part as u128) as Bits).collect()
}

unsafe fn write_string(p: *mut u64, k: usize, value: Bits) {
    let x = value as u128;
    for w in 0..WORDS {
        *p.add(k * WORDS + w) = (x >> (64 * w)) as u64;
    }
}

#[inline]
fn phase(bits: Bits, i: usize, a: usize) -> f64 {
    let (lo, hi) = if i < a { (i, a) } else { (a, i) };
    let mask = ((ONE << hi) - 1) & !((ONE << (lo + 1)) - 1);
    if (bits & mask).count_ones() & 1 == 1 { -1.0 } else { 1.0 }
}

#[inline]
fn bits_of(mut x: Bits, out: &mut [usize; WIDTH]) -> usize {
    let mut k = 0;
    while x != 0 {
        out[k] = x.trailing_zeros() as usize;
        k += 1;
        x &= x - 1;
    }
    k
}

impl Ci {
    #[inline]
    fn g(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        let n = self.n;
        unsafe { *self.g.get_unchecked(((p * n + q) * n + r) * n + s) }
    }

    /// Sort the double-excitation couplings by size, once. A search that only
    /// wants couplings above a threshold then reads the front of a list and
    /// stops, instead of trying every pair of target orbitals.
    /// `opposite[(i*n+j)*n*n ..]`: electrons leave i (one spin) and j (other spin) for p and q.
    /// `same[pair(i,j)*npair ..]` with i < j: both of one spin, to p < q.
    fn sorted_couplings(&mut self) {
        let n = self.n;
        let this = &*self;
        // each list is sorted where it lies, in one array allocated at its final size
        let mut opposite = vec![(0u8, 0u8, 0f64); n * n * n * n];
        opposite.par_chunks_mut((n * n).max(1)).enumerate().for_each(|(ij, v)| {
            let (i, j) = (ij / n, ij % n);
            for (pq, e) in v.iter_mut().enumerate() {
                *e = ((pq / n) as u8, (pq % n) as u8, this.g(i, pq / n, j, pq % n));
            }
            v.sort_unstable_by(|x, y| y.2.abs().partial_cmp(&x.2.abs()).unwrap());
        });
        let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (i + 1..n).map(move |j| (i, j))).collect();
        let mut same = vec![(0u8, 0u8, 0f64); pairs.len() * pairs.len()];
        same.par_chunks_mut(pairs.len().max(1)).enumerate().for_each(|(ij, v)| {
            let (i, j) = pairs[ij];
            for (e, &(p, q)) in v.iter_mut().zip(&pairs) {
                *e = (p as u8, q as u8, this.g(i, p, j, q) - this.g(i, q, j, p));
            }
            v.sort_unstable_by(|x, y| y.2.abs().partial_cmp(&x.2.abs()).unwrap());
        });
        self.opposite = opposite;
        self.same = same;
    }

    fn diag_of(&self, a: Bits, b: Bits) -> f64 {
        let (mut oa, mut ob) = ([0usize; WIDTH], [0usize; WIDTH]);
        let (ka, kb) = (bits_of(a, &mut oa), bits_of(b, &mut ob));
        let n = self.n;
        let mut e = 0.0;
        for &p in &oa[..ka] {
            e += self.h[p * n + p];
            for &q in &oa[..ka] {
                e += 0.5 * (self.g(p, p, q, q) - self.g(p, q, q, p));
            }
            for &q in &ob[..kb] {
                e += self.g(p, p, q, q);
            }
        }
        for &p in &ob[..kb] {
            e += self.h[p * n + p];
            for &q in &ob[..kb] {
                e += 0.5 * (self.g(p, p, q, q) - self.g(p, q, q, p));
            }
        }
        e
    }

    /// <D| H |D with i -> a in `same`>, without the sign.
    #[inline]
    fn single(&self, same: Bits, other: Bits, i: usize, a: usize) -> f64 {
        let mut v = self.h[i * self.n + a];
        let mut x = same;
        while x != 0 {
            let j = x.trailing_zeros() as usize;
            v += self.g(i, a, j, j) - self.g(i, j, j, a);
            x &= x - 1;
        }
        let mut x = other;
        while x != 0 {
            let j = x.trailing_zeros() as usize;
            v += self.g(i, a, j, j);
            x &= x - 1;
        }
        v
    }

    /// Matrix element between two determinants that differ only in one spin
    /// (strings x and x2; `other` is the common string of the other spin).
    #[inline]
    fn elem_same(&self, x: Bits, x2: Bits, other: Bits) -> f64 {
        let d = x ^ x2;
        let (holes, parts) = (x & d, x2 & d);
        match holes.count_ones() {
            1 => {
                let (i, a) = (holes.trailing_zeros() as usize, parts.trailing_zeros() as usize);
                phase(x, i, a) * self.single(x, other, i, a)
            }
            2 => {
                let (i, j) = (holes.trailing_zeros() as usize, WIDTH - 1 - holes.leading_zeros() as usize);
                let (a, b) = (parts.trailing_zeros() as usize, WIDTH - 1 - parts.leading_zeros() as usize);
                let p1 = phase(x, i, a);
                let x1 = x ^ (ONE << i) ^ (ONE << a);
                p1 * phase(x1, j, b) * (self.g(i, a, j, b) - self.g(i, b, j, a))
            }
            _ => 0.0,
        }
    }

    #[inline]
    fn find(list: &[(Bits, u32)], key: Bits) -> Option<u32> {
        list.binary_search_by_key(&key, |e| e.0).ok().map(|k| list[k].1)
    }

    /// Visit the listed partners that differ from (x, other) in string x only.
    /// Returns how many entries were looked at.
    fn same_spin<F: FnMut(u32, f64)>(&self, x: Bits, other: Bits, partners: &[(Bits, u32)], visit: &mut F) -> u64 {
        let (mut occ, mut vir) = ([0usize; WIDTH], [0usize; WIDTH]);
        let ko = bits_of(x, &mut occ);
        let kv = bits_of(!x & (full_mask(self.n)), &mut vir);
        let generated = ko * kv + ko * (ko.saturating_sub(1)) / 2 * kv * (kv.saturating_sub(1)) / 2;
        if partners.len() <= 8 * generated {
            for &(x2, j) in partners {
                let d = (x ^ x2).count_ones();
                if d == 2 || d == 4 {
                    visit(j, self.elem_same(x, x2, other));
                }
            }
            partners.len() as u64
        } else {
            for ii in 0..ko {
                for aa in 0..kv {
                    let x1 = x ^ (ONE << occ[ii]) ^ (ONE << vir[aa]);
                    if let Some(j) = Self::find(partners, x1) {
                        visit(j, self.elem_same(x, x1, other));
                    }
                    for jj in ii + 1..ko {
                        for bb in aa + 1..kv {
                            let x2 = x1 ^ (ONE << occ[jj]) ^ (ONE << vir[bb]);
                            if let Some(j) = Self::find(partners, x2) {
                                visit(j, self.elem_same(x, x2, other));
                            }
                        }
                    }
                }
            }
            4 * generated as u64 // a lookup costs several steps
        }
    }

    /// Visit every listed partner of determinant `idx` with its matrix
    /// element (the diagonal excluded). Returns the work done, in entries
    /// looked at.
    fn partners<F: FnMut(u32, f64)>(&self, idx: usize, visit: &mut F) -> u64 {
        let (a, b) = (self.da[idx], self.db[idx]);
        let mut work = self.same_spin(a, b, &self.cols[self.bid[idx] as usize], visit);
        work += self.same_spin(b, a, &self.rows[self.aid[idx] as usize], visit);
        // one electron of each spin moves
        let (mut occ, mut vir) = ([0usize; WIDTH], [0usize; WIDTH]);
        let ko = bits_of(b, &mut occ);
        let kv = bits_of(!b & (full_mask(self.n)), &mut vir);
        for &(aid2, i, p, sign_a) in &self.alpha_singles[self.aid[idx] as usize] {
            let row = &self.rows[aid2 as usize];
            let (i, p) = (i as usize, p as usize);
            if row.len() <= 4 * ko * kv {
                for &(b2, j) in row {
                    let d = b ^ b2;
                    if d.count_ones() == 2 {
                        let (q, r) = ((b & d).trailing_zeros() as usize, (b2 & d).trailing_zeros() as usize);
                        visit(j, sign_a * phase(b, q, r) * self.g(i, p, q, r));
                    }
                }
                work += row.len() as u64;
            } else {
                for &q in &occ[..ko] {
                    for &r in &vir[..kv] {
                        if let Some(j) = Self::find(row, b ^ (ONE << q) ^ (ONE << r)) {
                            visit(j, sign_a * phase(b, q, r) * self.g(i, p, q, r));
                        }
                    }
                }
                work += 4 * (ko * kv) as u64;
            }
        }
        work
    }

    fn sigma_one(&self, idx: usize, c: &[f64]) -> f64 {
        let mut acc = self.diag[idx] * c[idx];
        let start = self.cache_start[idx];
        if start != NOT_CACHED {
            for k in start as usize..start as usize + self.cache_len[idx] as usize {
                acc += self.cache_v[k] * c[self.cache_j[k] as usize];
            }
        } else {
            self.partners(idx, &mut |j, v| acc += v * c[j as usize]);
        }
        acc
    }

    fn contains(&self, amap: &HashMap<Bits, u32>, a: Bits, b: Bits) -> bool {
        match amap.get(&a) {
            Some(&id) => Self::find(&self.rows[id as usize], b).is_some(),
            None => false,
        }
    }
}

fn build(n: usize, h: &[f64], g: &[f64], da: &[Bits], db: &[Bits]) -> (Ci, HashMap<Bits, u32>) {
    let mut amap: HashMap<Bits, u32> = HashMap::new();
    let mut bmap: HashMap<Bits, u32> = HashMap::new();
    let (mut aid, mut bid) = (Vec::with_capacity(da.len()), Vec::with_capacity(da.len()));
    let (mut rows, mut cols): (Vec<Vec<(Bits, u32)>>, Vec<Vec<(Bits, u32)>>) = (Vec::new(), Vec::new());
    let mut alpha_strings = Vec::new();
    for k in 0..da.len() {
        let ia = *amap.entry(da[k]).or_insert_with(|| {
            rows.push(Vec::new());
            alpha_strings.push(da[k]);
            (rows.len() - 1) as u32
        });
        let ib = *bmap.entry(db[k]).or_insert_with(|| {
            cols.push(Vec::new());
            (cols.len() - 1) as u32
        });
        rows[ia as usize].push((db[k], k as u32));
        cols[ib as usize].push((da[k], k as u32));
        aid.push(ia);
        bid.push(ib);
    }
    rows.par_iter_mut().for_each(|r| r.sort_unstable());
    cols.par_iter_mut().for_each(|r| r.sort_unstable());
    let full = full_mask(n);
    let alpha_singles: Vec<Vec<(u32, u8, u8, f64)>> = alpha_strings
        .par_iter()
        .map(|&a| {
            let (mut occ, mut vir) = ([0usize; WIDTH], [0usize; WIDTH]);
            let ko = bits_of(a, &mut occ);
            let kv = bits_of(!a & full, &mut vir);
            let mut out = Vec::new();
            for &i in &occ[..ko] {
                for &p in &vir[..kv] {
                    if let Some(&id) = amap.get(&(a ^ (ONE << i) ^ (ONE << p))) {
                        out.push((id, i as u8, p as u8, phase(a, i, p)));
                    }
                }
            }
            out
        })
        .collect();
    let g_max = g.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let mut ci = Ci {
        n, h: h.to_vec(), g: g.to_vec(), da: da.to_vec(), db: db.to_vec(), aid, bid,
        diag: Vec::new(), rows, cols, alpha_singles, g_max,
        opposite: Vec::new(), same: Vec::new(), plain_search: std::env::var("CI_PLAIN_SEARCH").is_ok(),
        cache_start: vec![NOT_CACHED; da.len()], cache_len: vec![0; da.len()], cache_j: Vec::new(), cache_v: Vec::new(),
    };
    if !ci.plain_search {
        ci.sorted_couplings();
    }
    ci.diag = (0..da.len()).into_par_iter().map(|k| ci.diag_of(ci.da[k], ci.db[k])).collect();
    (ci, amap)
}

pub struct Handle {
    ci: Ci,
    amap: HashMap<Bits, u32>,
}

/// # Safety
/// Pointers must reference arrays of the stated sizes.
pub unsafe fn ci_new(n: usize, h: *const f64, g: *const f64, nd: usize, da: *const u64, db: *const u64) -> *mut Handle {
    let h = std::slice::from_raw_parts(h, n * n);
    let g = std::slice::from_raw_parts(g, n * n * n * n);
    let (da, db) = (read_strings(da, nd), read_strings(db, nd));
    let (ci, amap) = build(n, h, g, &da, &db);
    Box::into_raw(Box::new(Handle { ci, amap }))
}

/// # Safety
/// `handle` must come from `ci_new` and not have been freed.
pub unsafe fn ci_free(handle: *mut Handle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// # Safety
/// `out` must hold one value per determinant.
pub unsafe fn ci_diag(handle: *const Handle, out: *mut f64) {
    let ci = &(*handle).ci;
    std::slice::from_raw_parts_mut(out, ci.da.len()).copy_from_slice(&ci.diag);
}

/// out = H c, restricted to the listed determinants.
/// # Safety
/// `c` and `out` must hold one value per determinant.
pub unsafe fn ci_sigma(handle: *const Handle, c: *const f64, out: *mut f64) {
    let ci = &(*handle).ci;
    let c = std::slice::from_raw_parts(c, ci.da.len());
    let out = std::slice::from_raw_parts_mut(out, ci.da.len());
    out.par_iter_mut().enumerate().for_each(|(k, o)| *o = ci.sigma_one(k, c));
}

/// Decide from measurement where remembering pays. One pass visits every
/// determinant and records the work it took to find its partners and how
/// many were found. The partner lists are then stored for the determinants
/// with the most work per stored entry, until `budget_bytes` is used; later
/// applications of H read those lists instead of searching again.
/// Returns the number of determinants cached; `saved` gets the share of the
/// measured work that the cache removes.
/// # Safety
/// `handle` must come from `ci_new`.
pub unsafe fn ci_plan(handle: *mut Handle, budget_bytes: usize, saved: *mut f64) -> usize {
    let ci = &mut (*handle).ci;
    let nd = ci.da.len();
    ci.cache_start = vec![NOT_CACHED; nd];
    ci.cache_len = vec![0; nd];
    ci.cache_j = Vec::new();
    ci.cache_v = Vec::new();
    let measured: Vec<(u64, u32)> = {
        let ci = &*ci;
        (0..nd)
            .into_par_iter()
            .map(|k| {
                let mut hits = 0u32;
                let work = ci.partners(k, &mut |_, _| hits += 1);
                (work, hits)
            })
            .collect()
    };
    let mut order: Vec<usize> = (0..nd).collect();
    order.sort_unstable_by(|&x, &y| {
        let (rx, ry) = (measured[x].0 as f64 / (measured[x].1 as f64 + 1.0), measured[y].0 as f64 / (measured[y].1 as f64 + 1.0));
        ry.partial_cmp(&rx).unwrap()
    });
    let (mut bytes, mut chosen, mut work_saved, mut work_all) = (0usize, vec![false; nd], 0u64, 0u64);
    for &(w, _) in &measured {
        work_all += w;
    }
    for &k in &order {
        let need = 12 * measured[k].1 as usize;
        if bytes + need > budget_bytes {
            break;
        }
        // reading a stored entry is work too
        if measured[k].0 <= measured[k].1 as u64 {
            break;
        }
        bytes += need;
        chosen[k] = true;
        work_saved += measured[k].0 - measured[k].1 as u64;
    }
    // the stored lists go straight into two flat arrays of known size
    let total: usize = (0..nd).filter(|&k| chosen[k]).map(|k| measured[k].1 as usize).sum();
    let (mut flat_j, mut flat_v) = (vec![0u32; total], vec![0f64; total]);
    let mut count = 0;
    {
        let (mut rest_j, mut rest_v, mut at) = (&mut flat_j[..], &mut flat_v[..], 0usize);
        let mut pieces: Vec<(usize, &mut [u32], &mut [f64])> = Vec::new();
        for k in 0..nd {
            if !chosen[k] {
                continue;
            }
            let len = measured[k].1 as usize;
            let (js, tail_j) = rest_j.split_at_mut(len);
            let (vs, tail_v) = rest_v.split_at_mut(len);
            rest_j = tail_j;
            rest_v = tail_v;
            ci.cache_start[k] = at as u64;
            ci.cache_len[k] = len as u32;
            at += len;
            count += 1;
            pieces.push((k, js, vs));
        }
        let ci = &*ci;
        pieces.into_par_iter().for_each(|(k, js, vs)| {
            let mut at = 0;
            ci.partners(k, &mut |j, v| {
                js[at] = j;
                vs[at] = v;
                at += 1;
            });
        });
    }
    ci.cache_j = flat_j;
    ci.cache_v = flat_v;
    *saved = if work_all > 0 { work_saved as f64 / work_all as f64 } else { 0.0 };
    count
}

/// Sort by determinant and add up the entries of each determinant, in place.
fn compact(v: &mut Vec<(Bits, Bits, f64)>, parallel: bool) {
    if parallel {
        v.par_sort_unstable_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)));
    } else {
        v.sort_unstable_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)));
    }
    v.dedup_by(|later, kept| {
        let same = later.0 == kept.0 && later.1 == kept.1;
        if same {
            kept.2 += later.2;
        }
        same
    });
}

/// Compact a growing list when it has doubled since the last time, so that
/// the many contributions to one determinant do not pile up in memory.
fn tidy(mut state: (Vec<(Bits, Bits, f64)>, usize)) -> (Vec<(Bits, Bits, f64)>, usize) {
    if state.0.len() > 2 * state.1 + (1 << 20) {
        compact(&mut state.0, false);
        state.1 = state.0.len();
    }
    state
}

#[inline]
fn bucket(a: Bits, parts: u64) -> u64 {
    (fold(a).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) % parts
}

/// Every determinant outside the list that the vector couples to with
/// |H_ji c_i| > eps, with the summed coupling. With `parts` > 1 only the
/// determinants whose alpha string falls in bucket `part` are returned, so
/// that a large search can be done in pieces that fit in memory.
fn gather(hd: &Handle, c: &[f64], eps: f64, parts: u64, part: u64) -> Vec<(Bits, Bits, f64)> {
    let ci = &hd.ci;
    let n = ci.n;
    let full = full_mask(n);
    let scan = |mut acc: Vec<(Bits, Bits, f64)>, k: usize| -> Vec<(Bits, Bits, f64)> {
            let ck = c[k];
            if ck.abs() * 4.0 * ci.g_max.max(1.0) < eps {
                return acc;
            }
            let (a, b) = (ci.da[k], ci.db[k]);
            let (mut oa, mut va, mut ob, mut vb) = ([0usize; WIDTH], [0usize; WIDTH], [0usize; WIDTH], [0usize; WIDTH]);
            let (koa, kva) = (bits_of(a, &mut oa), bits_of(!a & full, &mut va));
            let (kob, kvb) = (bits_of(b, &mut ob), bits_of(!b & full, &mut vb));
            let mut push = |a2: Bits, b2: Bits, v: f64| {
                if v.abs() > eps && (parts == 1 || bucket(a2, parts) == part) && !ci.contains(&hd.amap, a2, b2) {
                    acc.push((a2, b2, v));
                }
            };
            // one spin changes: singles and doubles
            for spin in 0..2 {
                if spin == 1 && parts > 1 && bucket(a, parts) != part {
                    continue; // these targets keep the alpha string
                }
                let (x, other, occ, ko, vir, kv) =
                    if spin == 0 { (a, b, &oa, koa, &va, kva) } else { (b, a, &ob, kob, &vb, kvb) };
                for ii in 0..ko {
                    for aa in 0..kv {
                        let (i, p) = (occ[ii], vir[aa]);
                        let x1 = x ^ (ONE << i) ^ (ONE << p);
                        let s1 = phase(x, i, p);
                        let v = s1 * ci.single(x, other, i, p) * ck;
                        if spin == 0 { push(x1, other, v) } else { push(other, x1, v) }
                        if !ci.plain_search {
                            continue;
                        }
                        for jj in ii + 1..ko {
                            for bb in aa + 1..kv {
                                let (j, q) = (occ[jj], vir[bb]);
                                let v = s1 * phase(x1, j, q) * (ci.g(i, p, j, q) - ci.g(i, q, j, p)) * ck;
                                if v.abs() > eps {
                                    let x2 = x1 ^ (ONE << j) ^ (ONE << q);
                                    if spin == 0 { push(x2, other, v) } else { push(other, x2, v) }
                                }
                            }
                        }
                    }
                }
                if !ci.plain_search {
                    let npair = n * (n - 1) / 2;
                    for ii in 0..ko {
                        for jj in ii + 1..ko {
                            let (i, j) = (occ[ii], occ[jj]);
                            let at = (i * (2 * n - i - 1) / 2 + j - i - 1) * npair;
                            for &(p, q, w) in &ci.same[at..at + npair] {
                                if (w * ck).abs() <= eps {
                                    break;
                                }
                                let (p, q) = (p as usize, q as usize);
                                if x >> p & 1 == 1 || x >> q & 1 == 1 {
                                    continue;
                                }
                                let x1 = x ^ (ONE << i) ^ (ONE << p);
                                let v = phase(x, i, p) * phase(x1, j, q) * w * ck;
                                let x2 = x1 ^ (ONE << j) ^ (ONE << q);
                                if spin == 0 { push(x2, other, v) } else { push(other, x2, v) }
                            }
                        }
                    }
                }
            }
            // one electron of each spin moves
            if !ci.plain_search {
                for &i in &oa[..koa] {
                    for &j in &ob[..kob] {
                        let at = (i * n + j) * n * n;
                        for &(p, q, w) in &ci.opposite[at..at + n * n] {
                            if (w * ck).abs() <= eps {
                                break;
                            }
                            let (p, q) = (p as usize, q as usize);
                            if a >> p & 1 == 1 || b >> q & 1 == 1 {
                                continue;
                            }
                            let a2 = a ^ (ONE << i) ^ (ONE << p);
                            if parts > 1 && bucket(a2, parts) != part {
                                continue;
                            }
                            push(a2, b ^ (ONE << j) ^ (ONE << q), phase(a, i, p) * ck * w * phase(b, j, q));
                        }
                    }
                }
                return acc;
            }
            for &i in &oa[..koa] {
                for &p in &va[..kva] {
                    let sa = phase(a, i, p) * ck;
                    let a2 = a ^ (ONE << i) ^ (ONE << p);
                    if parts > 1 && bucket(a2, parts) != part {
                        continue;
                    }
                    for &j in &ob[..kob] {
                        for &q in &vb[..kvb] {
                            let v = sa * ci.g(i, p, j, q);
                            if v.abs() > eps {
                                push(a2, b ^ (ONE << j) ^ (ONE << q), v * phase(b, j, q));
                            }
                        }
                    }
                }
            }
            acc
    };
    let (mut found, _) = (0..ci.da.len())
        .into_par_iter()
        .fold(|| (Vec::new(), 0usize), |(acc, mark), k| tidy((scan(acc, k), mark)))
        .reduce(|| (Vec::new(), 0usize), |mut x, mut y| {
            if x.0.len() < y.0.len() {
                std::mem::swap(&mut x, &mut y);
            }
            x.0.append(&mut y.0);
            x.0.shrink_to_fit();
            tidy((x.0, x.1.max(y.1)))
        });
    compact(&mut found, true);
    found
}

/// Determinants outside the list, ranked by the energy each would bring:
/// (sum_i H_ji c_i)^2 / (E - H_jj). Only couplings |H_ji c_i| > eps are
/// followed. Writes at most `max_out` of the best, and the sum over all
/// found (second-order estimate of what is missing) to `total`. Returns the
/// number of distinct determinants found.
/// # Safety
/// Output arrays must hold `max_out` values.
pub unsafe fn ci_candidates(
    handle: *const Handle, c: *const f64, energy: f64, eps: f64, max_out: usize,
    out_a: *mut u64, out_b: *mut u64, out_gain: *mut f64, total: *mut f64, written: *mut usize,
) -> usize {
    let hd = &*handle;
    let ci = &hd.ci;
    let c = std::slice::from_raw_parts(c, ci.da.len());
    let mut gains = gather(hd, c, eps, 1, 0);
    gains.par_iter_mut().for_each(|e| e.2 = e.2 * e.2 / (energy - ci.diag_of(e.0, e.1)).min(-1e-6));
    *total = gains.par_iter().map(|e| e.2).sum();
    let count = gains.len();
    let keep = max_out.min(count);
    if keep > 0 && keep < count {
        gains.select_nth_unstable_by(keep - 1, |x, y| x.2.partial_cmp(&y.2).unwrap());
    }
    gains.truncate(keep);
    gains.par_sort_unstable_by(|x, y| x.2.partial_cmp(&y.2).unwrap());
    for (k, e) in gains.iter().enumerate() {
        *out_gain.add(k) = e.2;
        write_string(out_a, k, e.0);
        write_string(out_b, k, e.1);
    }
    *written = keep;
    count
}

/// Second-order estimate of the energy of everything outside the list,
/// restricted to bucket `part` of `parts` (sum over all parts for the whole).
/// Returns the number of determinants found; the sum goes to `total`.
/// # Safety
/// `c` must hold one value per determinant.
pub unsafe fn ci_pt2(handle: *const Handle, c: *const f64, energy: f64, eps: f64, parts: u64, part: u64, total: *mut f64) -> usize {
    let hd = &*handle;
    let ci = &hd.ci;
    let c = std::slice::from_raw_parts(c, ci.da.len());
    let merged = gather(hd, c, eps, parts.max(1), part);
    *total = merged.par_iter().map(|&(a, b, v)| v * v / (energy - ci.diag_of(a, b)).min(-1e-6)).sum();
    merged.len()
}
