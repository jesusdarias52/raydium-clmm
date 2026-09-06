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

/// [`note_div`] for a division this source performs on [`U256`] values.
///
/// The vendored library widens to `U256` at several sites where the deployed program divides
/// natively in 128 bits; the *operands* are the same numbers, so the path is the same. A pair
/// that genuinely does not fit in `u128` is a 256-bit division through a different routine and
/// is deliberately **not** counted here — it is not in the consumer's cost table.
#[inline]
#[allow(unused_variables)]
pub fn note_div_u256(numerator: crate::libraries::big_num::U256, denominator: crate::libraries::big_num::U256) {
    #[cfg(feature = "cu-counters")]
    {
        use crate::libraries::big_num::U256;
        if numerator > U256::from(u128::MAX) || denominator > U256::from(u128::MAX) {
            return;
        }
        note_div(numerator.as_u128(), denominator.as_u128());
    }
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
