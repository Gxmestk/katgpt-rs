// SPDX-License-Identifier: Apache-2.0
// Distilled from Percepta's `transformer-vm` (Apache-2.0 © Percepta).

//! Circle-point opcode dispatch for the WASM interpreter computation graph.
//!
//! Each opcode maps to a unique (x, y) point on the circle x² + y² = 32045.
//! The transformer detects the current opcode via dot product:
//!
//! ```text
//! op_dot(op) = px·x + py·y − R²·1 + 1
//! ```
//!
//! which equals 1 when the opcode matches and ≤ −1 otherwise.
//!
//! The dispatch mechanism uses geometric hashing: 16 base points on the circle
//! × 4 sign combinations = 64 circle points, of which the first 36 are assigned
//! to the 36 interpreter opcodes.

use crate::graph::types::{Expression, GraphBuilder};

// ── Constants ──────────────────────────────────────────────────

/// Squared radius of the opcode dispatch circle.
///
/// All circle points satisfy x² + y² = 32045.
pub const POINTS_R2: i32 = 32045;

/// Address stride between local variable slots per call depth level.
///
/// Each local occupies 4 bytes; stride of 256 allows up to 64 locals per frame.
pub const LOCAL_STRIDE: i32 = 256;

/// Number of supported opcodes.
pub const OPCODE_COUNT: usize = 36;

/// Circle points for opcode dispatch.
///
/// 16 base points × 4 sign combinations = 64 points total.
/// The first 36 correspond to the 36 opcodes in [`Opcode`].
pub const CIRCLE_POINTS: [(i32, i32); 64] = [
    // Base point (179, 2)
    (179, 2),
    (179, -2),
    (-179, 2),
    (-179, -2),
    // Base point (2, 179)
    (2, 179),
    (2, -179),
    (-2, 179),
    (-2, -179),
    // Base point (178, 19)
    (178, 19),
    (178, -19),
    (-178, 19),
    (-178, -19),
    // Base point (19, 178)
    (19, 178),
    (19, -178),
    (-19, 178),
    (-19, -178),
    // Base point (173, 46)
    (173, 46),
    (173, -46),
    (-173, 46),
    (-173, -46),
    // Base point (46, 173)
    (46, 173),
    (46, -173),
    (-46, 173),
    (-46, -173),
    // Base point (166, 67)
    (166, 67),
    (166, -67),
    (-166, 67),
    (-166, -67),
    // Base point (67, 166)
    (67, 166),
    (67, -166),
    (-67, 166),
    (-67, -166),
    // Base point (163, 74)
    (163, 74),
    (163, -74),
    (-163, 74),
    (-163, -74),
    // Base point (74, 163)
    (74, 163),
    (74, -163),
    (-74, 163),
    (-74, -163),
    // Base point (157, 86)
    (157, 86),
    (157, -86),
    (-157, 86),
    (-157, -86),
    // Base point (86, 157)
    (86, 157),
    (86, -157),
    (-86, 157),
    (-86, -157),
    // Base point (142, 109)
    (142, 109),
    (142, -109),
    (-142, 109),
    (-142, -109),
    // Base point (109, 142)
    (109, 142),
    (109, -142),
    (-109, 142),
    (-109, -142),
    // Base point (131, 122)
    (131, 122),
    (131, -122),
    (-131, 122),
    (-131, -122),
    // Base point (122, 131)
    (122, 131),
    (122, -131),
    (-122, 131),
    (-122, -131),
];

// ── Opcode enum ────────────────────────────────────────────────

/// All opcodes supported by the WASM interpreter computation graph.
///
/// Each variant maps to a unique circle point for geometric dispatch.
/// The discriminant gives the index into [`CIRCLE_POINTS`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Opcode {
    Halt = 0,
    Return = 1,
    Call = 2,
    Br = 3,
    BrIf = 4,
    Drop = 5,
    Select = 6,
    LocalGet = 7,
    LocalSet = 8,
    LocalTee = 9,
    GlobalGet = 10,
    GlobalSet = 11,
    I32Load = 12,
    I32Load8S = 13,
    I32Load8U = 14,
    I32Load16S = 15,
    I32Load16U = 16,
    I32Store = 17,
    I32Store8 = 18,
    I32Store16 = 19,
    I32Const = 20,
    I32Eqz = 21,
    I32Eq = 22,
    I32Ne = 23,
    I32LtS = 24,
    I32LtU = 25,
    I32GtS = 26,
    I32GtU = 27,
    I32LeS = 28,
    I32LeU = 29,
    I32GeS = 30,
    I32GeU = 31,
    I32Add = 32,
    I32Sub = 33,
    Output = 34,
    InputBase = 35,
}

impl Opcode {
    /// All opcodes in circle-point order.
    pub const ALL: [Self; OPCODE_COUNT] = [
        Self::Halt,
        Self::Return,
        Self::Call,
        Self::Br,
        Self::BrIf,
        Self::Drop,
        Self::Select,
        Self::LocalGet,
        Self::LocalSet,
        Self::LocalTee,
        Self::GlobalGet,
        Self::GlobalSet,
        Self::I32Load,
        Self::I32Load8S,
        Self::I32Load8U,
        Self::I32Load16S,
        Self::I32Load16U,
        Self::I32Store,
        Self::I32Store8,
        Self::I32Store16,
        Self::I32Const,
        Self::I32Eqz,
        Self::I32Eq,
        Self::I32Ne,
        Self::I32LtS,
        Self::I32LtU,
        Self::I32GtS,
        Self::I32GtU,
        Self::I32LeS,
        Self::I32LeU,
        Self::I32GeS,
        Self::I32GeU,
        Self::I32Add,
        Self::I32Sub,
        Self::Output,
        Self::InputBase,
    ];

    /// WASM bytecode for this opcode.
    #[rustfmt::skip]
    pub const fn wasm_byte(self) -> u8 {
        match self {
            Self::Halt        => 0x00,
            Self::Return      => 0x0F,
            Self::Call        => 0x10,
            Self::Br          => 0x0C,
            Self::BrIf        => 0x0D,
            Self::Drop        => 0x1A,
            Self::Select      => 0x1B,
            Self::LocalGet    => 0x20,
            Self::LocalSet    => 0x21,
            Self::LocalTee    => 0x22,
            Self::GlobalGet   => 0x23,
            Self::GlobalSet   => 0x24,
            Self::I32Load     => 0x28,
            Self::I32Load8S   => 0x2C,
            Self::I32Load8U   => 0x2D,
            Self::I32Load16S  => 0x2E,
            Self::I32Load16U  => 0x2F,
            Self::I32Store    => 0x36,
            Self::I32Store8   => 0x3A,
            Self::I32Store16  => 0x3B,
            Self::I32Const    => 0x41,
            Self::I32Eqz      => 0x45,
            Self::I32Eq       => 0x46,
            Self::I32Ne       => 0x47,
            Self::I32LtS      => 0x48,
            Self::I32LtU      => 0x49,
            Self::I32GtS      => 0x4A,
            Self::I32GtU      => 0x4B,
            Self::I32LeS      => 0x4C,
            Self::I32LeU      => 0x4D,
            Self::I32GeS      => 0x4E,
            Self::I32GeU      => 0x4F,
            Self::I32Add      => 0x6A,
            Self::I32Sub      => 0x6B,
            Self::Output      => 0xFF,
            Self::InputBase   => 0xFE,
        }
    }

    /// Circle point (x, y) for geometric hashing dispatch.
    ///
    /// The point satisfies x² + y² = [`POINTS_R2`].
    pub const fn circle_point(self) -> (i32, i32) {
        CIRCLE_POINTS[self as usize]
    }

    /// Stack depth change when this opcode executes.
    #[rustfmt::skip]
    pub const fn stack_delta(self) -> i32 {
        match self {
            Self::Halt        =>  0,
            Self::Return      =>  0,
            Self::Call        =>  0,
            Self::Br          =>  0,
            Self::BrIf        => -1,
            Self::Drop        => -1,
            Self::Select      => -2,
            Self::LocalGet    =>  1,
            Self::LocalSet    => -1,
            Self::LocalTee    =>  0,
            Self::GlobalGet   =>  1,
            Self::GlobalSet   => -1,
            Self::I32Load     =>  0,
            Self::I32Load8S   =>  0,
            Self::I32Load8U   =>  0,
            Self::I32Load16S  =>  0,
            Self::I32Load16U  =>  0,
            Self::I32Store    => -2,
            Self::I32Store8   => -2,
            Self::I32Store16  => -2,
            Self::I32Const    =>  1,
            Self::I32Eqz      =>  0,
            Self::I32Eq       => -1,
            Self::I32Ne       => -1,
            Self::I32LtS      => -1,
            Self::I32LtU      => -1,
            Self::I32GtS      => -1,
            Self::I32GtU      => -1,
            Self::I32LeS      => -1,
            Self::I32LeU      => -1,
            Self::I32GeS      => -1,
            Self::I32GeU      => -1,
            Self::I32Add      => -1,
            Self::I32Sub      => -1,
            Self::Output      => -1,
            Self::InputBase   =>  0,
        }
    }

    /// Whether this opcode stores a result to the stack (STS_OPS).
    ///
    /// These opcodes produce a value that gets pushed onto the WASM stack.
    pub const fn is_sts(self) -> bool {
        matches!(
            self,
            Opcode::I32Const
                | Opcode::I32Add
                | Opcode::I32Sub
                | Opcode::I32Eqz
                | Opcode::Select
                | Opcode::I32GtS
                | Opcode::I32GtU
                | Opcode::I32LeS
                | Opcode::I32LeU
                | Opcode::I32GeS
                | Opcode::I32GeU
                | Opcode::I32LtS
                | Opcode::I32LtU
                | Opcode::I32Eq
                | Opcode::I32Ne
                | Opcode::LocalGet
                | Opcode::GlobalGet
                | Opcode::I32Load8S
                | Opcode::I32Load8U
                | Opcode::I32Load16S
                | Opcode::I32Load16U
                | Opcode::I32Load
        )
    }

    /// Whether this opcode writes to memory or local variables.
    ///
    /// Used for Futamura specialization (WRITE_OPS in the Python source).
    pub const fn is_write(self) -> bool {
        matches!(
            self,
            Opcode::LocalSet
                | Opcode::LocalTee
                | Opcode::I32Store8
                | Opcode::I32Store16
                | Opcode::I32Store
        )
    }

    /// Python-style opcode name (e.g., `"i32.add"`, `"local.get"`).
    #[rustfmt::skip]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Halt        => "halt",
            Self::Return      => "return",
            Self::Call        => "call",
            Self::Br          => "br",
            Self::BrIf        => "br_if",
            Self::Drop        => "drop",
            Self::Select      => "select",
            Self::LocalGet    => "local.get",
            Self::LocalSet    => "local.set",
            Self::LocalTee    => "local.tee",
            Self::GlobalGet   => "global.get",
            Self::GlobalSet   => "global.set",
            Self::I32Load     => "i32.load",
            Self::I32Load8S   => "i32.load8_s",
            Self::I32Load8U   => "i32.load8_u",
            Self::I32Load16S  => "i32.load16_s",
            Self::I32Load16U  => "i32.load16_u",
            Self::I32Store    => "i32.store",
            Self::I32Store8   => "i32.store8",
            Self::I32Store16  => "i32.store16",
            Self::I32Const    => "i32.const",
            Self::I32Eqz      => "i32.eqz",
            Self::I32Eq       => "i32.eq",
            Self::I32Ne       => "i32.ne",
            Self::I32LtS      => "i32.lt_s",
            Self::I32LtU      => "i32.lt_u",
            Self::I32GtS      => "i32.gt_s",
            Self::I32GtU      => "i32.gt_u",
            Self::I32LeS      => "i32.le_s",
            Self::I32LeU      => "i32.le_u",
            Self::I32GeS      => "i32.ge_s",
            Self::I32GeU      => "i32.ge_u",
            Self::I32Add      => "i32.add",
            Self::I32Sub      => "i32.sub",
            Self::Output      => "output",
            Self::InputBase   => "input_base",
        }
    }

    /// Parse an opcode from its Python-style name.
    #[rustfmt::skip]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "halt"        => Some(Self::Halt),
            "return"      => Some(Self::Return),
            "call"        => Some(Self::Call),
            "br"          => Some(Self::Br),
            "br_if"       => Some(Self::BrIf),
            "drop"        => Some(Self::Drop),
            "select"      => Some(Self::Select),
            "local.get"   => Some(Self::LocalGet),
            "local.set"   => Some(Self::LocalSet),
            "local.tee"   => Some(Self::LocalTee),
            "global.get"  => Some(Self::GlobalGet),
            "global.set"  => Some(Self::GlobalSet),
            "i32.load"    => Some(Self::I32Load),
            "i32.load8_s" => Some(Self::I32Load8S),
            "i32.load8_u" => Some(Self::I32Load8U),
            "i32.load16_s"=> Some(Self::I32Load16S),
            "i32.load16_u"=> Some(Self::I32Load16U),
            "i32.store"   => Some(Self::I32Store),
            "i32.store8"  => Some(Self::I32Store8),
            "i32.store16" => Some(Self::I32Store16),
            "i32.const"   => Some(Self::I32Const),
            "i32.eqz"     => Some(Self::I32Eqz),
            "i32.eq"      => Some(Self::I32Eq),
            "i32.ne"      => Some(Self::I32Ne),
            "i32.lt_s"    => Some(Self::I32LtS),
            "i32.lt_u"    => Some(Self::I32LtU),
            "i32.gt_s"    => Some(Self::I32GtS),
            "i32.gt_u"    => Some(Self::I32GtU),
            "i32.le_s"    => Some(Self::I32LeS),
            "i32.le_u"    => Some(Self::I32LeU),
            "i32.ge_s"    => Some(Self::I32GeS),
            "i32.ge_u"    => Some(Self::I32GeU),
            "i32.add"     => Some(Self::I32Add),
            "i32.sub"     => Some(Self::I32Sub),
            "output"      => Some(Self::Output),
            "input_base"  => Some(Self::InputBase),
            _ => None,
        }
    }

    /// Look up opcode by WASM bytecode value.
    #[rustfmt::skip]
    pub fn from_wasm_byte(byte: u8) -> Option<Self> {
        match byte {
            0x00 => Some(Self::Halt),
            0x0F => Some(Self::Return),
            0x10 => Some(Self::Call),
            0x0C => Some(Self::Br),
            0x0D => Some(Self::BrIf),
            0x1A => Some(Self::Drop),
            0x1B => Some(Self::Select),
            0x20 => Some(Self::LocalGet),
            0x21 => Some(Self::LocalSet),
            0x22 => Some(Self::LocalTee),
            0x23 => Some(Self::GlobalGet),
            0x24 => Some(Self::GlobalSet),
            0x28 => Some(Self::I32Load),
            0x2C => Some(Self::I32Load8S),
            0x2D => Some(Self::I32Load8U),
            0x2E => Some(Self::I32Load16S),
            0x2F => Some(Self::I32Load16U),
            0x36 => Some(Self::I32Store),
            0x3A => Some(Self::I32Store8),
            0x3B => Some(Self::I32Store16),
            0x41 => Some(Self::I32Const),
            0x45 => Some(Self::I32Eqz),
            0x46 => Some(Self::I32Eq),
            0x47 => Some(Self::I32Ne),
            0x48 => Some(Self::I32LtS),
            0x49 => Some(Self::I32LtU),
            0x4A => Some(Self::I32GtS),
            0x4B => Some(Self::I32GtU),
            0x4C => Some(Self::I32LeS),
            0x4D => Some(Self::I32LeU),
            0x4E => Some(Self::I32GeS),
            0x4F => Some(Self::I32GeU),
            0x6A => Some(Self::I32Add),
            0x6B => Some(Self::I32Sub),
            0xFF => Some(Self::Output),
            0xFE => Some(Self::InputBase),
            _ => None,
        }
    }
}

// ── Program instruction (for specialized / Futamura mode) ──────

/// A single instruction in a WASM program for specialized builds.
///
/// Used when baking a specific program into the computation graph
/// via the First Futamura Projection.
#[derive(Clone, Debug)]
pub struct ProgramInstruction {
    /// Which opcode this instruction executes.
    pub opcode: Opcode,
    /// Little-endian immediate bytes (padded to 4 for i32.const, etc.).
    pub bytes: [u8; 4],
}

impl ProgramInstruction {
    /// Create a new instruction with the given opcode and zero bytes.
    pub fn new(opcode: Opcode) -> Self {
        Self {
            opcode,
            bytes: [0; 4],
        }
    }

    /// Create an instruction with an i32 immediate value.
    pub fn with_i32(opcode: Opcode, value: i32) -> Self {
        Self {
            opcode,
            bytes: value.to_le_bytes(),
        }
    }

    /// Compute the unsigned 32-bit immediate from bytes.
    pub fn immediate_u32(&self) -> u32 {
        u32::from_le_bytes(self.bytes)
    }

    /// Compute the signed 32-bit immediate from bytes.
    pub fn immediate_i32(&self) -> i32 {
        i32::from_le_bytes(self.bytes)
    }
}

// ── OpcodeDispatch — cached dispatch helper ────────────────────

/// Cached opcode dispatch helper for the WASM interpreter build.
///
/// Holds the fetched opcode coordinate expressions and provides
/// [`op_dot()`](OpcodeDispatch::op_dot) and [`is_op()`](OpcodeDispatch::is_op)
/// methods with internal caching.
///
/// # Universal vs Specialized mode
///
/// In **universal mode** (`specialized = false`), `is_op` uses `reglu(one, op_dot(op))`
/// — a soft gate suitable for attention-based instruction fetch.
///
/// In **specialized mode** (`specialized = true`, Futamura projection),
/// `is_op` uses `stepglu(one, op_dot(op))` — a hard step function since
/// the opcode is known exactly from the program counter.
pub struct OpcodeDispatch {
    /// Fetched opcode x-coordinate from instruction fetch.
    fetched_x: Expression,
    /// Fetched opcode y-coordinate from instruction fetch.
    fetched_y: Expression,
    /// The `one` scalar expression.
    one_expr: Expression,
    /// Whether this is a specialized (Futamura) build.
    specialized: bool,
    /// Cache: opcode → dot product expression.
    ///
    /// Index is `Opcode` discriminant (0..OPCODE_COUNT). Array-backed for O(1)
    /// lookup instead of `HashMap` hashing on a `#[repr(u8)]` enum.
    op_dot_cache: [Option<Expression>; OPCODE_COUNT],
    /// Cache: opcode → is_op gate expression. Same indexing as `op_dot_cache`.
    is_op_cache: [Option<Expression>; OPCODE_COUNT],
}

impl OpcodeDispatch {
    /// Create a new dispatch context.
    ///
    /// * `fetched_x` — expression for the fetched opcode x-coordinate
    /// * `fetched_y` — expression for the fetched opcode y-coordinate
    /// * `one_expr` — the scalar `one` expression
    /// * `specialized` — `true` for Futamura (hard step), `false` for universal (soft reglu)
    pub fn new(
        fetched_x: Expression,
        fetched_y: Expression,
        one_expr: Expression,
        specialized: bool,
    ) -> Self {
        Self {
            fetched_x,
            fetched_y,
            one_expr,
            specialized,
            op_dot_cache: std::array::from_fn(|_| None),
            is_op_cache: std::array::from_fn(|_| None),
        }
    }

    /// Compute the dot-product gate expression for an opcode.
    ///
    /// Returns `px·x + py·y − R²·1 + 1`:
    /// - Equals 1 when `op` matches the fetched opcode (dot product with circle point).
    /// - ≤ −1 otherwise (geometric hashing guarantees).
    ///
    /// Results are cached per opcode.
    pub fn op_dot(&mut self, op: Opcode) -> Expression {
        let idx = op as usize;
        if let Some(cached) = &self.op_dot_cache[idx] {
            return cached.clone();
        }

        let (px, py) = op.circle_point();
        let result = self.fetched_x.clone() * (px as f64) + self.fetched_y.clone() * (py as f64)
            - self.one_expr.clone() * (POINTS_R2 as f64)
            + self.one_expr.clone();

        self.op_dot_cache[idx] = Some(result.clone());
        result
    }

    /// Compute the is-op gate expression for an opcode.
    ///
    /// * Universal mode: `reglu(one, op_dot(op))` — soft gate via ReLU.
    /// * Specialized mode: `stepglu(one, op_dot(op))` — hard gate via step function.
    ///
    /// Results are cached per opcode.
    pub fn is_op(&mut self, builder: &mut GraphBuilder, op: Opcode) -> Expression {
        let idx = op as usize;
        if let Some(cached) = &self.is_op_cache[idx] {
            return cached.clone();
        }

        let dot = self.op_dot(op);
        let result = if self.specialized {
            builder.stepglu(self.one_expr.clone(), dot)
        } else {
            builder.reglu(self.one_expr.clone(), dot)
        };

        self.is_op_cache[idx] = Some(result.clone());
        result
    }

    /// Returns whether this dispatch context is in specialized (Futamura) mode.
    #[inline]
    pub fn is_specialized(&self) -> bool {
        self.specialized
    }
}

// ── Unique (stack_delta, is_sts) pairs for token generation ────

/// Compute the set of unique `(stack_delta, is_sts)` pairs across all opcodes.
///
/// Used to generate the commit token vocabulary.
pub fn unique_stack_pairs() -> Vec<(i32, u8)> {
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();
    for op in Opcode::ALL {
        let sd = op.stack_delta();
        let sts: u8 = if op.is_sts() { 1 } else { 0 };
        if seen.insert((sd, sts)) {
            result.push((sd, sts));
        }
    }
    result
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circle_points_on_circle() {
        for &(x, y) in &CIRCLE_POINTS {
            let r2 = (x as i64) * (x as i64) + (y as i64) * (y as i64);
            assert_eq!(
                r2, POINTS_R2 as i64,
                "point ({x}, {y}) has r²={r2}, expected {POINTS_R2}"
            );
        }
    }

    #[test]
    fn test_opcode_count() {
        assert_eq!(Opcode::ALL.len(), OPCODE_COUNT);
        assert_eq!(Opcode::ALL.len(), 36);
    }

    #[test]
    fn test_opcode_discriminants_sequential() {
        for (i, op) in Opcode::ALL.iter().enumerate() {
            assert_eq!(*op as usize, i, "opcode {i} has wrong discriminant");
        }
    }

    #[test]
    fn test_opcode_name_roundtrip() {
        for op in Opcode::ALL {
            let name = op.as_str();
            let parsed = Opcode::from_name(name);
            assert_eq!(parsed, Some(op), "roundtrip failed for {name}");
        }
    }

    #[test]
    fn test_opcode_wasm_bytes_unique() {
        let mut seen = std::collections::HashSet::new();
        for op in Opcode::ALL {
            let byte = op.wasm_byte();
            assert!(
                seen.insert(byte),
                "duplicate wasm byte 0x{byte:02X} for {}",
                op.as_str()
            );
        }
    }

    #[test]
    fn test_opcode_from_wasm_byte_roundtrip() {
        for op in Opcode::ALL {
            let byte = op.wasm_byte();
            let parsed = Opcode::from_wasm_byte(byte);
            assert_eq!(parsed, Some(op));
        }
    }

    #[test]
    fn test_opcode_circle_point_matches_index() {
        for (i, op) in Opcode::ALL.iter().enumerate() {
            assert_eq!(*op as usize, i);
            let point = op.circle_point();
            assert_eq!(point, CIRCLE_POINTS[i]);
        }
    }

    #[test]
    fn test_sts_ops_count() {
        // 22 opcodes are STS (store-to-stack), matching the Python STS_OPS set
        let sts_count = Opcode::ALL.iter().filter(|op| op.is_sts()).count();
        assert_eq!(sts_count, 22);
    }

    #[test]
    fn test_write_ops_count() {
        // 5 opcodes are WRITE, matching the Python WRITE_OPS set
        let write_count = Opcode::ALL.iter().filter(|op| op.is_write()).count();
        assert_eq!(write_count, 5);
    }

    #[test]
    fn test_unique_stack_pairs() {
        let pairs = unique_stack_pairs();
        // The Python code generates tokens for each unique (sd, sts) pair × 2 bt values
        assert!(!pairs.is_empty());
        // Verify all pairs come from actual opcodes
        for (sd, sts) in &pairs {
            assert!(
                Opcode::ALL
                    .iter()
                    .any(|op| op.stack_delta() == *sd && op.is_sts() == (*sts == 1)),
                "pair ({sd}, {sts}) not found in any opcode"
            );
        }
    }

    #[test]
    fn test_program_instruction_immediate() {
        let instr = ProgramInstruction::with_i32(Opcode::I32Const, 42);
        assert_eq!(instr.immediate_i32(), 42);
        assert_eq!(instr.immediate_u32(), 42u32);

        let instr_neg = ProgramInstruction::with_i32(Opcode::I32Const, -1);
        assert_eq!(instr_neg.immediate_i32(), -1);
        assert_eq!(instr_neg.immediate_u32(), 0xFFFFFFFF);
    }

    #[test]
    fn test_dispatch_op_dot_creates_expression() {
        let mut builder = GraphBuilder::new();
        let one_expr = Expression::from_dim(builder.one);
        let fetched_x = builder.generic("fx");
        let fetched_y = builder.generic("fy");

        let mut dispatch = OpcodeDispatch::new(fetched_x, fetched_y, one_expr, false);

        let dot = dispatch.op_dot(Opcode::I32Add);
        assert!(
            !dot.is_zero(),
            "op_dot should produce a non-zero expression"
        );

        // Second call should return cached result (same expression)
        let dot2 = dispatch.op_dot(Opcode::I32Add);
        assert_eq!(dot, dot2, "cached op_dot should be identical");
    }

    #[test]
    fn test_dispatch_is_op_universal_uses_reglu() {
        let mut builder = GraphBuilder::new();
        let one_expr = Expression::from_dim(builder.one);
        let fetched_x = builder.generic("fx");
        let fetched_y = builder.generic("fy");

        let dim_count_before = builder.dim_count();
        let mut dispatch = OpcodeDispatch::new(fetched_x, fetched_y, one_expr, false);

        let gate = dispatch.is_op(&mut builder, Opcode::I32Add);
        assert!(!gate.is_zero());

        // Universal mode: is_op creates one ReGLU dimension
        assert!(builder.dim_count() > dim_count_before);
    }

    #[test]
    fn test_dispatch_is_op_specialized_uses_stepglu() {
        let mut builder = GraphBuilder::new();
        let one_expr = Expression::from_dim(builder.one);
        let fetched_x = builder.generic("fx");
        let fetched_y = builder.generic("fy");

        let dim_count_before = builder.dim_count();
        let mut dispatch = OpcodeDispatch::new(fetched_x, fetched_y, one_expr, true);

        let gate = dispatch.is_op(&mut builder, Opcode::I32Add);
        assert!(!gate.is_zero());

        // Specialized mode: is_op creates stepglu (2 ReGLU + 1 persist)
        assert!(builder.dim_count() > dim_count_before);
    }
}
