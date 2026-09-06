//! Per-swap counters for the client-side compute-unit model. **Not used on chain.**
//!
//! Everything here is behind the `cu-counters` feature and compiles to nothing without it, so
//! the deployed program is byte-identical either way. It exists because Solarbitrage prices a
//! Raydium CLMM hop before sending, and the dominant per-sub-step cost turned out to be the
//! branch path the runtime's `u128` division takes on this swap's own operands — a quantity a
//! caller cannot reconstruct without replaying the whole swap.
//!
//! Traced on the deployed `CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK` (2026-09-05): a
//! dynamic-fee sub-step performs six `u128` divisions, and their paths account for **103%** of
//! the measured per-sub-step cost difference between two pools whose every other counter is
//! identical (+1,001 predicted against +971 measured). Two of the six divisors are pool state —
//! `liquidity` and `sqrt_price_x64` — which is why the cost is per-pool and why no constant can
//! hold it: a `liquidity` under 2^32 takes a 168-instruction fast path where a wider one takes
//! 505, and a smaller `sqrt_price_x64` widens the quotient into a 637-instruction arm.
//!
//! [`udiv_path`] is a port of `compiler_builtins`' trifecta `u128_div_rem`, and **the index
//! layout is deliberately identical to the one the Orca fork uses**
//! (`orca_whirlpools_core::counters`), so one cost table shape serves both DEXes. Validated
//! against 531 traced CLMM divisions: 10 distinct paths, every one constant to within 2 CU.

#[cfg(feature = "cu-counters")]
use std::cell::{Cell, RefCell};

/// The number of distinct `__udivti3` branch paths [`udiv_path`] distinguishes.
pub const UDIV_PATHS: usize = 28;

/// What one swap spent, as counts the caller can price.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwapCounters {
    /// Histogram over [`udiv_path`] of every `u128` division the swap performed at an
    /// instrumented site. The cost table lives with the consumer, because the per-path
    /// instruction counts are a property of the deployed binary and not of this source.
    pub udiv_paths: [u32; UDIV_PATHS],
    /// Divisions offered to [`note_div`], whether or not they classified — a tripwire against
    /// an instrumented site being dropped by a future edit.
    pub divisions: u32,
}

impl SwapCounters {
    pub const ZERO: Self = Self { udiv_paths: [0; UDIV_PATHS], divisions: 0 };

    /// This counter set less `base`, saturating — the shape a caller wants when it brackets one
    /// `swap` inside a longer-lived process.
    pub fn since(&self, base: &Self) -> Self {
        let mut out = Self::ZERO;
        for i in 0..UDIV_PATHS {
            out.udiv_paths[i] = self.udiv_paths[i].saturating_sub(base.udiv_paths[i]);
        }
        out.divisions = self.divisions.saturating_sub(base.divisions);
        out
    }
}

#[cfg(feature = "cu-counters")]
thread_local! {
    static COUNTERS: RefCell<SwapCounters> = const { RefCell::new(SwapCounters::ZERO) };
    static ENABLED: Cell<bool> = const { Cell::new(false) };
}

/// Turn counting on or off for this thread; returns the previous state.
///
/// Off by default, and off is free: [`note_div`] reads one thread-local `Cell` and returns.
/// A caller that quotes 20-40 times per hop inside `optimize` wants it off for all but the one
/// call whose cost it is about to charge.
#[cfg(feature = "cu-counters")]
pub fn set_enabled(on: bool) -> bool {
    ENABLED.with(|e| e.replace(on))
}

#[cfg(not(feature = "cu-counters"))]
pub fn set_enabled(_on: bool) -> bool { false }

#[cfg(feature = "cu-counters")]
pub fn enabled() -> bool { ENABLED.with(|e| e.get()) }

#[cfg(not(feature = "cu-counters"))]
pub fn enabled() -> bool { false }

/// This thread's counters as they stand.
#[cfg(feature = "cu-counters")]
pub fn snapshot() -> SwapCounters { COUNTERS.with(|c| *c.borrow()) }

#[cfg(not(feature = "cu-counters"))]
pub fn snapshot() -> SwapCounters { SwapCounters::ZERO }

/// Record one `u128 / u128` the chain performs, classified by the branch path its division
/// takes. Call it with the operands the **deployed** program divides, which are not always the
/// ones this source divides — where the vendored code widens to `U256` and the chain does not,
/// the `u128` pair is what to pass.
#[inline]
#[allow(unused_variables)]
pub fn note_div(numerator: u128, denominator: u128) {
    #[cfg(feature = "cu-counters")]
    {
        if !enabled() { return; }
        let p = udiv_path(numerator, denominator);
        COUNTERS.with(|c| {
            let mut c = c.borrow_mut();
            c.udiv_paths[p] = c.udiv_paths[p].saturating_add(1);
            c.divisions = c.divisions.saturating_add(1);
        });
    }
}

/// Which branch path Rust's `compiler_builtins` `u128` division takes on these operands.
///
/// The chain's `__udivti3` is `compiler_builtins::int::specialized_div_rem::u128_div_rem`, the
/// **trifecta** algorithm (`impl_trifecta!` with `n = 64`, `n_h = 32`). Index layout, kept
/// identical to the Orca fork's so the two cost tables have the same shape:
///
/// | index | path |
/// |---|---|
/// | 0 / 1 | quotient 0 / quotient 1 (`div_lz <= duo_lz`) |
/// | 2 | half division (`duo < 2^64`) |
/// | 3 | short division (`div < 2^32`) |
/// | 4 / 5 | two-possibility, uncorrected / corrected (`quo - 1`) |
/// | 6 / 7 | the same with `duo` a full 128 bits |
/// | 8 + 10·(steps − 1) + 2·exit + zero_shl | long division |
///
/// A faithful port, kept in the algorithm's own variable names so it can be read against
/// `trifecta.rs`; only the quotient bookkeeping is dropped, because the path is the answer.
#[cfg(feature = "cu-counters")]
pub fn udiv_path(duo: u128, div: u128) -> usize {
    const N: u32 = 64;
    const N_H: u32 = 32;
    #[inline(always)]
    fn twopos_corrected(duo: u128, div: u128, duo_lz: u32) -> usize {
        let shift = N - duo_lz;
        let duo_sig_n = (duo >> shift) as u64;
        let div_sig_n = (div >> shift) as u64;
        let quo = duo_sig_n / div_sig_n;
        let div_lo = div as u64;
        let div_hi = (div >> N) as u64;
        let tmp_a = (quo as u128).wrapping_mul(div_lo as u128);
        let (tmp_lo, carry) = (tmp_a as u64, (tmp_a >> N) as u64);
        let tmp_b = (quo as u128)
            .wrapping_mul(div_hi as u128)
            .wrapping_add(carry as u128);
        let (tmp_hi, overflow) = (tmp_b as u64, (tmp_b >> N) as u64);
        let tmp = (tmp_lo as u128) | ((tmp_hi as u128) << N);
        usize::from(overflow != 0 || duo < tmp)
    }
    #[inline(always)]
    fn long(steps: u32, exit: usize, zero_shl: bool) -> usize {
        8 + 10 * (steps.clamp(1, 2) as usize - 1) + 2 * exit + usize::from(zero_shl)
    }
    if div == 0 { return 0; }
    let div_lz = div.leading_zeros();
    let mut duo_lz = duo.leading_zeros();
    if div_lz <= duo_lz { return usize::from(duo >= div); }
    if duo_lz >= N { return 2; }
    if div_lz >= N + N_H { return 3; }
    let lz_diff = div_lz - duo_lz;
    if lz_diff < N_H {
        return 4 + twopos_corrected(duo, div, duo_lz) + if duo_lz == 0 { 2 } else { 0 };
    }
    let mut duo = duo;
    let div_extra = (N + N_H) - div_lz;
    let div_sig_n_h = (div >> div_extra) as u32;
    let div_sig_n_h_add1 = (div_sig_n_h as u64) + 1;
    let mut steps = 0u32;
    let mut zero_shl = false;
    loop {
        let duo_extra = N - duo_lz;
        let duo_sig_n = (duo >> duo_extra) as u64;
        if div_extra <= duo_extra {
            let quo_part = (duo_sig_n / div_sig_n_h_add1) as u128;
            let extra_shl = duo_extra - div_extra;
            zero_shl |= extra_shl == 0;
            duo = duo.wrapping_sub(div.wrapping_mul(quo_part) << extra_shl);
            steps += 1;
        } else {
            return long(steps, 3 + twopos_corrected(duo, div, duo_lz), zero_shl);
        }
        duo_lz = duo.leading_zeros();
        if div_lz <= duo_lz { return long(steps, usize::from(div <= duo), zero_shl); }
        if N <= duo_lz { return long(steps, 2, zero_shl); }
    }
}

#[cfg(all(test, feature = "cu-counters"))]
mod tests {
    use super::*;
    use crate::libraries::big_num::U256;

    /// The limb replay must reproduce `uint`'s own quotient, or the windows it hands
    /// [`note_div`] describe divisions the chain never performs. Both shapes are covered:
    /// `div_mod_small` (divisor under 2^64) and `div_mod_knuth`, including the D3 refinement
    /// and the D6 add-back, which do not divide but do move the running remainder.
    #[test]
    fn the_limb_replay_reproduces_uints_own_quotient() {
        let cases: [(U256, U256); 8] = [
            (U256::from(1u128 << 100), U256::from(3u128)),
            (U256::from(u128::MAX), U256::from(1u128 << 64)),
            // the swap's widest division: `L * dsqrt << 64` over `sqrt_a * sqrt_b`
            (
                U256::from(369_220_455_397u128) * U256::from(44_000_000_000_000u128) << 64usize,
                U256::from(87_895_387_825_834_134u128) * U256::from(87_895_387_825_834_200u128),
            ),
            (U256::from(u128::MAX) << 64usize, U256::from(u128::MAX)),
            (U256::from(u128::MAX) << 100usize, U256::from(7u128 << 90)),
            (U256::from(5u128), U256::from(7u128)),
            (U256::from(1u128 << 127), U256::from(1u128 << 63)),
            (U256::MAX, U256::from(1u128 << 65) + U256::from(1u8)),
        ];
        let prev = set_enabled(true);
        for (num, den) in cases {
            let q = note_wide_div(&num.0, &den.0);
            let mut got = U256::zero();
            for (i, limb) in q.iter().take(4).enumerate() {
                got = got + (U256::from(*limb) << (64 * i));
            }
            assert_eq!(got, num / den, "num {num} den {den}");
        }
        set_enabled(prev);
    }

    /// The count is the point: `uint` reaches the runtime's division routine once per quotient
    /// limb, and it is that multiplicity the model was blind to. Traced on `ABk1rvmb`, the
    /// swap's `sqrt_a * sqrt_b` division is **two** `__udivti3` calls.
    #[test]
    fn the_widest_swap_division_is_two_calls_against_the_normalised_top_limb() {
        let num = U256::from(369_220_455_397u128) * U256::from(44_000_000_000_000u128) << 64usize;
        let den = U256::from(87_895_387_825_834_134u128) * U256::from(87_895_387_825_834_200u128);
        let prev = set_enabled(true);
        let before = snapshot();
        note_div_u256(num, den);
        let after = snapshot().since(&before);
        set_enabled(prev);
        assert_eq!(after.divisions, 2, "one q_hat estimate per quotient limb");
    }

    /// Every branch of the trifecta port, on operands built to select it. Verbatim from the
    /// Orca fork's own test, so the two ports cannot drift apart silently. The indices are the
    /// layout documented on [`udiv_path`]; a slip here would mis-price a class.
    #[test]
    fn udiv_path_selects_each_trifecta_branch() {
        let norm = 1u128 << 63 | 12345; // a normalised 64-bit divisor, as Knuth D hands it over
        assert_eq!(udiv_path(norm - 1, norm), 0, "quotient 0");
        assert_eq!(udiv_path(norm + 1, norm), 1, "quotient 1");
        assert_eq!(udiv_path(1u128 << 40, 1u128 << 20), 2, "half: duo < 2^64, div_lz > duo_lz");
        assert_eq!(udiv_path(1u128 << 100, 1u128 << 20), 3, "short: div < 2^32");
        // two possibility: duo of 65..95 bits against a 64-bit divisor
        assert_eq!(udiv_path(1u128 << 80, norm) & !1, 4);
        // the same with a full 128-bit duo against a divisor within 32 bits of it
        assert_eq!(udiv_path(u128::MAX - 7, 1u128 << 100) & !1, 6);
        // long division: duo of 96+ bits against the 64-bit divisor; one step then an exit
        let p = udiv_path(1u128 << 100, norm);
        assert!((8..18).contains(&p), "one long step, got {p}");
        let p = udiv_path(u128::MAX, norm);
        assert!((18..28).contains(&p), "two long steps, got {p}");
    }

    /// The four operand shapes traced on chain, with the path each was observed to take.
    /// These are the classes the CLMM cost table prices; if one moves, the table is stale.
    #[test]
    fn traced_clmm_operand_shapes_keep_their_paths() {
        // `(amount << 64) / liquidity`, liquidity 32 bits -> the short arm, 168 CU.
        assert_eq!(udiv_path(1u128 << 74, 2_871_814_867u128), 3);
        // the same with a 40-bit liquidity -> long division, 505 CU.
        assert_eq!(udiv_path(1u128 << 79, 824_085_722_713u128), 12);
        // `mul_div result / sqrt_ratio_a` with a 64-bit price -> two-possibility, 374 CU.
        assert_eq!(udiv_path(1u128 << 80, 15_344_570_447_257_825_405u128) & !1, 4);
        // the fee-rate denominators are always small numerators -> the half arm, 177 CU.
        assert_eq!(udiv_path(1_000_000u128 * 900_000, 1_000_000u128), 2);
    }
}

/// Note every `__udivti3` one wide division performs, by replaying `uint`'s own algorithm.
///
/// **A `U256 / U256` is not one `__udivti3`.** `uint::construct_uint!` divides limb by limb, and
/// each limb step is a `div_mod_word(hi, lo, y)` — a real `u128 / u128`, so a single source-level
/// `/` reaches the runtime's division routine once per *quotient limb*. Counting it as one
/// division is what left the deployed program's widest per-sub-step division uncounted: traced
/// on two dynamic-fee pools it is **two** calls costing 813 and 552 CU respectively, a 261 CU
/// per-sub-step spread that is pure per-pool operand width.
///
/// The replay follows `uint-0.9.5`'s `div_mod`, which has three shapes:
///
/// - `numerator < denominator` — early return, **no division at all**.
/// - `denominator < 2^64` — `div_mod_small`: one `div_mod_word` per limb, top down. The **top**
///   limb's step is not counted: its `hi` is the literal `0`, so `(0 << 64) | lo` is a
///   zero-extended `u64` and LLVM narrows the division to 64 bits. Confirmed on the trace — the
///   LP-fee-growth division (`remaining << 64 / liquidity`, a two-limb `U128`) reaches
///   `__udivti3` exactly **once** per sub-step, not twice.
/// - otherwise — `div_mod_knuth`: one `q_hat` estimate per quotient limb, each dividing a
///   128-bit window of the *normalized* running remainder by the divisor's normalized top limb.
///   That top limb is what a trace sees as the divisor, and it is why the observed divisor of
///   this swap's widest division is `(sqrt_a * sqrt_b) >> 49` rather than either sqrt price
///   (verified to four significant figures on `ABk1rvmb`).
///
/// The Knuth windows are computed in closed form rather than by simulating D4/D6: before digit
/// `j` the algorithm holds `P_{j+1} mod V` where `P_k = floor(U / 2^(64k))` and `U`, `V` are the
/// shift-normalized operands, so the dividend is `(P_{j+1} mod V) >> 64(n-2)`. No correction
/// step divides, so nothing else has to be modelled.
#[inline]
#[allow(unused_variables)]
pub fn note_div_u256(numerator: crate::libraries::big_num::U256, denominator: crate::libraries::big_num::U256) {
    #[cfg(feature = "cu-counters")]
    {
        if !enabled() {
            return;
        }
        let _ = note_wide_div(&numerator.0, &denominator.0);
    }
}

/// [`note_div_u256`] for a two-limb `U128`.
#[inline]
#[allow(unused_variables)]
pub fn note_div_u128(numerator: crate::libraries::big_num::U128, denominator: crate::libraries::big_num::U128) {
    #[cfg(feature = "cu-counters")]
    {
        if !enabled() {
            return;
        }
        let _ = note_wide_div(&numerator.0, &denominator.0);
    }
}

/// The limb-by-limb replay behind [`note_div_u256`], over `uint`'s own little-endian limbs.
///
/// Faithful to `uint-0.9.5`: `div_mod_small` for a divisor under 2^64, otherwise Algorithm D
/// with the same normalization, the same two-step `q_hat` correction and the same D6 add-back.
/// The corrections do not divide, but they change the running remainder, so a replay that
/// skipped them would feed the *next* digit a wrong window.
#[cfg(feature = "cu-counters")]
fn note_wide_div(num: &[u64], den: &[u64]) -> [u64; 10] {
    const MAX: usize = 10;
    let mut quotient = [0u64; MAX];
    let w = num.len();
    debug_assert!(w <= 8 && w == den.len());
    let bits = |v: &[u64]| -> usize {
        for i in (0..v.len()).rev() {
            if v[i] != 0 {
                return i * 64 + (64 - v[i].leading_zeros() as usize);
            }
        }
        0
    };
    let my_bits = bits(num);
    let your_bits = bits(den);
    // Dividing by zero, or by something larger than us: `div_mod` returns before dividing.
    if your_bits == 0 || my_bits < your_bits {
        return quotient;
    }

    if your_bits <= 64 {
        // `div_mod_small`: one `div_mod_word` per limb, top down. The **top** limb's step is not
        // counted -- its `hi` is the literal `0`, so `(0 << 64) | lo` is a zero-extended `u64`
        // and the division narrows to 64 bits. Confirmed on the trace: the LP-fee-growth
        // division (a two-limb `U128`) reaches `__udivti3` exactly once per sub-step, not twice.
        let y = u128::from(den[0]);
        let mut rem = u128::from(num[w - 1]) % y;
        for i in (0..w - 1).rev() {
            let x = (rem << 64) | u128::from(num[i]);
            note_div(x, y);
            quotient[i] = (x / y) as u64;
            rem = x % y;
        }
        quotient[w - 1] = num[w - 1] / (y as u64);
        return quotient;
    }

    // `div_mod_knuth`.
    let words = |b: usize| 1 + (b - 1) / 64;
    let n = words(your_bits);
    let m = words(my_bits) - n;
    let shift = den[n - 1].leading_zeros();

    // D1: normalize. `v` cannot grow a limb -- `shift` is exactly the room above its top limb --
    // but `u` can, which is why `full_shl` returns one limb more than the type holds.
    let mut v = [0u64; MAX];
    let mut u = [0u64; MAX];
    if shift == 0 {
        v[..w].copy_from_slice(den);
        u[..w].copy_from_slice(num);
    } else {
        for i in 0..w {
            v[i] = (den[i] << shift) | if i == 0 { 0 } else { den[i - 1] >> (64 - shift) };
            u[i] = (num[i] << shift) | if i == 0 { 0 } else { num[i - 1] >> (64 - shift) };
        }
        u[w] = num[w - 1] >> (64 - shift);
    }
    let v_n_1 = v[n - 1];
    let v_n_2 = v[n - 2];

    // D2..D7: one quotient digit per iteration, each estimating `q_hat` with a single
    // `div_mod_word` -- the only division in the whole algorithm.
    for j in (0..=m).rev() {
        let u_jn = u[j + n];
        let mut q_hat = if u_jn < v_n_1 {
            let x = (u128::from(u_jn) << 64) | u128::from(u[j + n - 1]);
            note_div(x, u128::from(v_n_1));
            let mut q = (x / u128::from(v_n_1)) as u64;
            let mut r = (x % u128::from(v_n_1)) as u64;
            // D3's refinement: at most two iterations, and it does not divide.
            loop {
                let prod = u128::from(q) * u128::from(v_n_2);
                let (hi, lo) = ((prod >> 64) as u64, prod as u64);
                if (hi, lo) <= (r, u[j + n - 2]) {
                    break;
                }
                q -= 1;
                let (nr, ov) = r.overflowing_add(v_n_1);
                r = nr;
                if ov {
                    break;
                }
            }
            q
        } else {
            u64::MAX
        };

        // D4: u[j..j+n+1] -= q_hat * v[..n]
        let mut borrow = 0u64;
        let mut carry = 0u64;
        for i in 0..n {
            let p = u128::from(q_hat) * u128::from(v[i]) + u128::from(carry);
            carry = (p >> 64) as u64;
            let (t, b1) = u[j + i].overflowing_sub(p as u64);
            let (t, b2) = t.overflowing_sub(borrow);
            u[j + i] = t;
            borrow = u64::from(b1 || b2);
        }
        let (t, b1) = u[j + n].overflowing_sub(carry);
        let (t, b2) = t.overflowing_sub(borrow);
        u[j + n] = t;

        // D6: the estimate was one too high (~2^-63), so add `v` back.
        if b1 || b2 {
            q_hat -= 1;
            let mut c = 0u64;
            for i in 0..n {
                let sum = u128::from(u[j + i]) + u128::from(v[i]) + u128::from(c);
                u[j + i] = sum as u64;
                c = (sum >> 64) as u64;
            }
            u[j + n] = u[j + n].wrapping_add(c);
        }
        quotient[j] = q_hat;
    }
    quotient
}

/// [`note_div`] for `U128::MAX / ratio` -- the positive-tick inversion in
/// `tick_math::get_sqrt_price_at_tick`.
///
/// **The operands are not the ones the source writes.** `U128` is `uint`'s two-limb type, so a
/// 128-by-64 division is done in two 64-bit steps and only the *second* reaches `__udivti3`:
/// the first computes `u64::MAX / ratio` natively, and the second divides
/// `(u64::MAX % ratio) << 64 | u64::MAX` by the same ratio. Read off the deployed bytecode at
/// pc 0x20228..0x20233 and verified against 16 of 16 traced calls on
/// `HwU4MRZ4mCZpH2SA`.
///
/// This is why the term is invisible to a caller: it fires once per positive-tick
/// `get_sqrt_price_at_tick`, so a pool whose ticks are all negative never pays it, and one whose
/// ticks are positive pays 635..751 CU per sub-step.
#[inline]
#[allow(unused_variables)]
pub fn note_u128_max_div(ratio: u128) {
    #[cfg(feature = "cu-counters")]
    {
        if !enabled() || ratio == 0 {
            return;
        }
        // A ratio of 64 bits or more takes a differently-shaped division; leave it uncounted
        // rather than counted wrongly.
        if ratio > u64::MAX as u128 {
            return;
        }
        let rem = (u64::MAX as u128) % ratio;
        note_div((rem << 64) | (u64::MAX as u128), ratio);
    }
}
