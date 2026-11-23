//! Selected configuration interaction on a free list of determinants.
//!
//! A determinant is a pair of bit strings (alpha, beta): bit p set means
//! orbital p is occupied by an electron of that spin. The Hamiltonian is
//! never stored. `ci_sigma` applies it to a vector by walking, for every
//! determinant, only the partners that are actually in the list; and
//! `ci_candidates` proposes the determinants outside the list that the
//! current wavefunction couples to most strongly.
//!
//! Integrals: h[p*n+q] one-electron, g[((p*n+q)*n+r)*n+s] = (pq|rs).
//!
//! The kernel is compiled twice from `core.rs`: with 64-bit strings (up to 64
//! orbitals; functions `ci_new`, ...) and with 128-bit strings (up to 128
//! orbitals; `ci_new_wide`, ...). Across the C interface a string is one or
//! two 64-bit words, lowest first.

pub mod walk;

mod narrow {
    type Bits = u64;
    include!("core.rs");
}

mod wide {
    type Bits = u128;
    include!("core.rs");
}

macro_rules! export {
    ($m:ident, $new:ident, $free:ident, $diag:ident, $sigma:ident, $plan:ident, $candidates:ident, $pt2:ident, $rdm1:ident) => {
        #[no_mangle]
        pub unsafe extern "C" fn $new(n: usize, h: *const f64, g: *const f64, nd: usize, da: *const u64, db: *const u64) -> *mut $m::Handle {
            $m::ci_new(n, h, g, nd, da, db)
        }
        #[no_mangle]
        pub unsafe extern "C" fn $free(handle: *mut $m::Handle) {
            $m::ci_free(handle)
        }
        #[no_mangle]
        pub unsafe extern "C" fn $diag(handle: *const $m::Handle, out: *mut f64) {
            $m::ci_diag(handle, out)
        }
        #[no_mangle]
        pub unsafe extern "C" fn $sigma(handle: *const $m::Handle, c: *const f64, out: *mut f64) {
            $m::ci_sigma(handle, c, out)
        }
        #[no_mangle]
        pub unsafe extern "C" fn $plan(handle: *mut $m::Handle, budget_bytes: usize, saved: *mut f64) -> usize {
            $m::ci_plan(handle, budget_bytes, saved)
        }
        #[no_mangle]
        pub unsafe extern "C" fn $candidates(
            handle: *const $m::Handle, c: *const f64, energy: f64, eps: f64, max_out: usize,
            out_a: *mut u64, out_b: *mut u64, out_gain: *mut f64, total: *mut f64, written: *mut usize,
        ) -> usize {
            $m::ci_candidates(handle, c, energy, eps, max_out, out_a, out_b, out_gain, total, written)
        }
        #[no_mangle]
        pub unsafe extern "C" fn $pt2(handle: *const $m::Handle, c: *const f64, energy: f64, eps: f64, parts: u64, part: u64, total: *mut f64) -> usize {
            $m::ci_pt2(handle, c, energy, eps, parts, part, total)
        }
        #[no_mangle]
        pub unsafe extern "C" fn $rdm1(handle: *const $m::Handle, c: *const f64, out: *mut f64) {
            $m::ci_rdm1(handle, c, out)
        }
    };
}

export!(narrow, ci_new, ci_free, ci_diag, ci_sigma, ci_plan, ci_candidates, ci_pt2, ci_rdm1);
export!(wide, ci_new_wide, ci_free_wide, ci_diag_wide, ci_sigma_wide, ci_plan_wide, ci_candidates_wide, ci_pt2_wide, ci_rdm1_wide);
