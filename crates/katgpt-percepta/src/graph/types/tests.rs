//! Tests for graph::types (extracted from mod.rs by Issue 176).

use super::ValidationError;
use super::*;

// ── Expression Tests ────────────────────────────────────────

#[test]
fn test_expression_zero() {
    let expr = Expression::zero();
    assert!(expr.is_zero());
    assert!(expr.terms.is_empty());
    assert_eq!(expr.len(), 0);
}

#[test]
fn test_expression_from_dim() {
    let expr = Expression::from_dim(42);
    assert!(!expr.is_zero());
    assert_eq!(expr.get(42), 1.0);
    assert_eq!(expr.get(99), 0.0);
    assert_eq!(expr.len(), 1);
}

#[test]
fn test_expression_from_scalar() {
    let expr = Expression::from_scalar(3.5, 1);
    assert_eq!(expr.get(1), 3.5);
    assert_eq!(expr.len(), 1);

    let zero = Expression::from_scalar(0.0, 1);
    assert!(zero.is_zero());
}

#[test]
fn test_expression_from_terms_removes_zeros() {
    let expr = Expression::from_terms(HashMap::from([(1, 2.0), (2, 0.0), (3, -1.0)]));
    assert_eq!(expr.len(), 2);
    assert!(!expr.terms.contains_key(&2));
}

#[test]
fn test_expression_add() {
    let a = Expression::from_terms(HashMap::from([(1, 2.0)]));
    let b = Expression::from_terms(HashMap::from([(1, 3.0), (2, 1.0)]));
    let result = a + b;
    assert_eq!(result.get(1), 5.0);
    assert_eq!(result.get(2), 1.0);
}

#[test]
fn test_expression_add_cancels() {
    let a = Expression::from_terms(HashMap::from([(1, 2.0)]));
    let b = Expression::from_terms(HashMap::from([(1, -2.0)]));
    let result = a + b;
    assert!(result.is_zero());
}

#[test]
fn test_expression_sub() {
    let a = Expression::from_terms(HashMap::from([(1, 5.0), (2, 3.0)]));
    let b = Expression::from_terms(HashMap::from([(1, 2.0)]));
    let result = a - b;
    assert_eq!(result.get(1), 3.0);
    assert_eq!(result.get(2), 3.0);
}

#[test]
fn test_expression_mul_scalar() {
    let expr = Expression::from_terms(HashMap::from([(1, 2.0), (3, -1.0)]));
    let result = expr * 3.0;
    assert_eq!(result.get(1), 6.0);
    assert_eq!(result.get(3), -3.0);
}

#[test]
fn test_expression_mul_scalar_commutative() {
    let expr = Expression::from_terms(HashMap::from([(1, 2.0)]));
    let result = 3.0 * expr;
    assert_eq!(result.get(1), 6.0);
}

#[test]
fn test_expression_mul_zero() {
    let expr = Expression::from_terms(HashMap::from([(1, 2.0), (3, -1.0)]));
    let result = expr * 0.0;
    assert!(result.is_zero());
}

#[test]
fn test_expression_neg() {
    let expr = Expression::from_terms(HashMap::from([(1, 2.0), (3, -1.0)]));
    let result = -expr;
    assert_eq!(result.get(1), -2.0);
    assert_eq!(result.get(3), 1.0);
}

#[test]
fn test_expression_evaluate() {
    let expr = Expression::from_terms(HashMap::from([(1, 2.0), (3, -1.0)]));
    let values = HashMap::from([(1, 5.0), (3, 10.0)]);
    // 2.0 * 5.0 + (-1.0) * 10.0 = 10.0 - 10.0 = 0.0
    assert_eq!(expr.evaluate(&values), 0.0);

    let values2 = HashMap::from([(1, 5.0)]);
    // 2.0 * 5.0 + (-1.0) * 0.0 = 10.0
    assert_eq!(expr.evaluate(&values2), 10.0);
}

#[test]
fn test_expression_set_removes_zero() {
    let mut expr = Expression::from_terms(HashMap::from([(1, 2.0), (2, 3.0)]));
    expr.set(1, 0.0);
    assert!(!expr.terms.contains_key(&1));
    assert_eq!(expr.get(2), 3.0);
}

#[test]
fn test_expression_set_nonzero() {
    let mut expr = Expression::from_terms(HashMap::from([(1, 2.0)]));
    expr.set(1, 5.0);
    assert_eq!(expr.get(1), 5.0);
}

#[test]
fn test_expression_equality() {
    let a = Expression::from_terms(HashMap::from([(1, 2.0), (2, 3.0)]));
    let b = Expression::from_terms(HashMap::from([(2, 3.0), (1, 2.0)]));
    assert_eq!(a, b);

    let c = Expression::from_terms(HashMap::from([(1, 2.0)]));
    assert_ne!(a, c);
}

#[test]
fn test_expression_display() {
    let expr = Expression::from_terms(HashMap::from([(1, 1.0)]));
    let displayed = format!("{expr}");
    assert!(displayed.contains("dim_1"));

    let zero = Expression::zero();
    assert_eq!(format!("{zero}"), "0");
}

// ── Dimension Tests ─────────────────────────────────────────

#[test]
fn test_dimension_display() {
    let dim = Dimension {
        id: 0,
        name: "one".to_string(),
        kind: DimensionKind::Input,
    };
    assert_eq!(format!("{dim}"), "input:one[0]");

    let dim_reglu = Dimension {
        id: 5,
        name: "reglu_5".to_string(),
        kind: DimensionKind::ReGLU {
            a_expr: Expression::zero(),
            b_expr: Expression::zero(),
        },
    };
    assert_eq!(format!("{dim_reglu}"), "reglu:reglu_5[5]");
}

#[test]
fn test_dimension_new_generic() {
    let dim = Dimension::new_generic(99, "test".to_string());
    assert_eq!(dim.id, 99);
    assert_eq!(dim.name, "test");
    assert!(matches!(dim.kind, DimensionKind::Generic));
}

#[test]
fn test_dimension_kind_display_names() {
    let cases: Vec<(DimensionKind, &str)> = vec![
        (DimensionKind::Input, "input"),
        (
            DimensionKind::ReGLU {
                a_expr: Expression::zero(),
                b_expr: Expression::zero(),
            },
            "reglu",
        ),
        (
            DimensionKind::Persist {
                expr: Expression::zero(),
            },
            "persist",
        ),
        (
            DimensionKind::LookUp {
                lookup_id: 0,
                value_index: 0,
            },
            "lookup",
        ),
        (
            DimensionKind::CumSum {
                value_expr: Expression::zero(),
            },
            "cumsum",
        ),
        (DimensionKind::Generic, "generic"),
    ];

    for (kind, expected_prefix) in cases {
        let dim = Dimension {
            id: 0,
            name: "test".to_string(),
            kind,
        };
        let displayed = format!("{dim}");
        assert!(
            displayed.starts_with(expected_prefix),
            "Expected '{displayed}' to start with '{expected_prefix}'"
        );
    }
}

// ── GraphBuilder Tests ──────────────────────────────────────

#[test]
fn test_builder_new_has_input_dims() {
    let builder = GraphBuilder::new();
    assert_eq!(builder.dim_count(), 4);

    let one = builder.get_dim(builder.one).unwrap();
    assert_eq!(one.name, "one");
    assert!(matches!(one.kind, DimensionKind::Input));

    let pos = builder.get_dim(builder.position).unwrap();
    assert_eq!(pos.name, "position");
    assert!(matches!(pos.kind, DimensionKind::Input));
}

#[test]
fn test_builder_input_dim_ids() {
    let builder = GraphBuilder::new();
    assert_eq!(builder.one, 0);
    assert_eq!(builder.position, 1);
    assert_eq!(builder.inv_log_pos, 2);
    assert_eq!(builder.position_sq, 3);
}

#[test]
fn test_builder_reglu() {
    let mut builder = GraphBuilder::new();

    let result = builder.reglu(3.0_f64, 2.0_f64);
    assert_eq!(builder.dim_count(), 5); // 4 inputs + 1 reglu
    assert_eq!(result.len(), 1);

    let dim_id = *result.terms.keys().next().unwrap();
    let dim = builder.get_dim(dim_id).unwrap();
    assert!(matches!(dim.kind, DimensionKind::ReGLU { .. }));
}

#[test]
fn test_builder_reglu_caching() {
    let mut builder = GraphBuilder::new();

    let r1 = builder.reglu(3.0_f64, 2.0_f64);
    let r2 = builder.reglu(3.0_f64, 2.0_f64);
    assert_eq!(r1, r2);
    assert_eq!(builder.dim_count(), 5); // 4 inputs + 1 reglu (cached)
}

#[test]
fn test_builder_reglu_different_inputs() {
    let mut builder = GraphBuilder::new();

    let r1 = builder.reglu(3.0_f64, 2.0_f64);
    let r2 = builder.reglu(3.0_f64, 5.0_f64);
    assert_ne!(r1, r2);
    assert_eq!(builder.dim_count(), 6); // 4 inputs + 2 reglu
}

#[test]
fn test_builder_stepglu() {
    let mut builder = GraphBuilder::new();

    let result = builder.stepglu(5.0_f64, 0.0_f64);
    // stepglu creates: 2 ReGLU + 1 Persist = 3 new dims
    assert_eq!(builder.dim_count(), 7); // 4 inputs + 3 new
    assert_eq!(result.len(), 1);

    let dim_id = *result.terms.keys().next().unwrap();
    let dim = builder.get_dim(dim_id).unwrap();
    assert!(matches!(dim.kind, DimensionKind::Persist { .. }));
}

#[test]
fn test_builder_stepglu_caching() {
    let mut builder = GraphBuilder::new();

    let s1 = builder.stepglu(5.0_f64, 0.0_f64);
    let s2 = builder.stepglu(5.0_f64, 0.0_f64);
    assert_eq!(s1, s2);
    assert_eq!(builder.dim_count(), 7); // 4 inputs + 3 new (cached)
}

#[test]
fn test_builder_persist() {
    let mut builder = GraphBuilder::new();
    let one = builder.one;

    let expr = Expression::from_terms(HashMap::from([(one, 1.0)]));
    let result = builder.persist(expr);
    assert_eq!(builder.dim_count(), 5); // 4 inputs + 1 persist

    let dim_id = *result.terms.keys().next().unwrap();
    let dim = builder.get_dim(dim_id).unwrap();
    assert!(matches!(dim.kind, DimensionKind::Persist { .. }));
}

#[test]
fn test_builder_generic() {
    let mut builder = GraphBuilder::new();

    let result = builder.generic("my_intermediate");
    let dim_id = *result.terms.keys().next().unwrap();
    let dim = builder.get_dim(dim_id).unwrap();
    assert_eq!(dim.name, "my_intermediate");
    assert!(matches!(dim.kind, DimensionKind::Generic));
}

#[test]
fn test_builder_fetch() {
    let mut builder = GraphBuilder::new();

    let value = Expression::from_dim(builder.position);
    let result = builder.fetch(value, None, None, None, TieBreak::Latest);

    // 4 inputs + 3 (multiply for zero key in to_2d_key) + 1 lookup dim
    assert_eq!(builder.dim_count(), 8);
    assert_eq!(builder.lookup_count(), 1);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_builder_fetch_vec() {
    let mut builder = GraphBuilder::new();

    let v1 = Expression::from_dim(builder.position);
    let v2 = Expression::from_dim(builder.one);
    let results = builder.fetch_vec(vec![v1, v2], None, None, None, TieBreak::Latest);

    assert_eq!(results.len(), 2);
    // 4 inputs + 3 (multiply for zero key) + 2 lookup dims
    assert_eq!(builder.dim_count(), 9);
    assert_eq!(builder.lookup_count(), 1);
}

#[test]
fn test_builder_fetch_with_query_key() {
    let mut builder = GraphBuilder::new();
    let one = builder.one;
    let pos = builder.position;

    let value = Expression::from_dim(pos);
    let query = Expression::from_dim(one);
    let key = Expression::from_dim(pos);

    let result = builder.fetch(value, Some(query), Some(key), None, TieBreak::Latest);

    assert_eq!(result.len(), 1);
    assert_eq!(builder.lookup_count(), 1);
}

#[test]
fn test_builder_fetch_sum() {
    let mut builder = GraphBuilder::new();

    let values = vec![
        Expression::from_dim(builder.position),
        Expression::from_dim(builder.one),
    ];
    let results = builder.fetch_sum(values);

    assert_eq!(results.len(), 2);
    // 4 inputs + 3 (multiply for zero key) + 2 lookup dims + 2 ReGLU
    assert_eq!(builder.dim_count(), 11);
    assert_eq!(builder.lookup_count(), 1);
}

#[test]
fn test_builder_fetch_sum_single() {
    let mut builder = GraphBuilder::new();

    let value = Expression::from_dim(builder.position);
    let result = builder.fetch_sum_single(value);

    assert_eq!(result.len(), 1);
    // 4 inputs + 3 (multiply for zero key) + 1 lookup dim + 1 ReGLU
    assert_eq!(builder.dim_count(), 9);
}

// ── ProgramGraph Tests ──────────────────────────────────────

// ── C5: Graph validation tests ─────────────────────────────
#[test]
fn test_graph_validate_simple() {
    let mut b = GraphBuilder::new();
    let one = b.one;
    let pos = b.position;
    let x = b.reglu(pos.into_expr(one), 2.0_f64.into_expr(one));
    let y = b.persist(x.clone());
    let graph = b.build(vec![], vec![y]);
    assert!(graph.validate().is_ok());
}

#[test]
fn test_graph_validate_missing_dim() {
    let mut b = GraphBuilder::new();
    let one = b.one;
    let pos = b.position;
    let x = b.reglu(pos.into_expr(one), 2.0_f64.into_expr(one));
    let y = b.persist(x.clone());
    let mut graph = b.build(vec![], vec![y]);

    // Corrupt: remove a dependency dim (pos is referenced by reglu's b_expr)
    // check_dim_consistency runs first, so it catches MissingDim, not OutputMissingDim
    graph.all_dims.remove(&pos);
    let err = graph.validate();
    assert!(
        matches!(err, Err(ValidationError::MissingDim { missing, .. }) if missing == pos),
        "expected MissingDim for pos, got {err:?}"
    );
}

#[test]
fn test_graph_validate_cycle() {
    // Build a valid graph first, then verify validate passes
    let mut b = GraphBuilder::new();
    let one = b.one;
    let pos = b.position;
    let x = b.reglu(pos.into_expr(one), 2.0_f64.into_expr(one));
    let _y = b.persist(x.clone());
    let graph = b.build(vec![], vec![_y]);
    assert!(
        graph.validate().is_ok(),
        "valid graph should pass validation"
    );
}

#[test]
fn test_graph_validate_diamond_dependency() {
    let mut b = GraphBuilder::new();
    let pos = b.position;
    let one = b.one;
    // Diamond: pos -> reglu_a, pos -> reglu_b, both -> persist
    let a = b.reglu(pos.into_expr(one), one.into_expr(one));
    let b_expr = b.reglu(pos.into_expr(one), 3.0_f64.into_expr(one));
    let combined = a.clone() + b_expr.clone();
    let _out = b.persist(combined);
    let graph = b.build(vec![], vec![_out]);
    assert!(graph.validate().is_ok());
}

// ── ProgramGraph build tests ───────────────────────────────

#[test]
fn test_program_graph_build() {
    let mut builder = GraphBuilder::new();
    let one = builder.one;

    let x = builder.reglu(1.0_f64, 2.0_f64);
    let y = builder.persist(x.clone());

    let input_tokens = vec![Expression::from_dim(one)];
    let output_tokens = vec![y.clone()];

    let graph = builder.build(input_tokens, output_tokens);

    assert_eq!(graph.all_dims.len(), 6); // 4 inputs + 1 reglu + 1 persist
    assert_eq!(graph.input_tokens.len(), 1);
    assert_eq!(graph.output_tokens.len(), 1);
    assert_eq!(graph.one, 0);
    assert_eq!(graph.position, 1);
}

#[test]
fn test_program_graph_captures_all_dims() {
    let mut builder = GraphBuilder::new();

    let r = builder.reglu(1.0_f64, 1.0_f64);
    let p = builder.persist(r.clone());
    let s = builder.stepglu(1.0_f64, 0.0_f64);

    let graph = builder.build(vec![], vec![p, s]);

    // 4 inputs + 1 reglu(r) + 1 persist(p) + 2 reglu(s) + 1 persist(s) = 9
    assert_eq!(graph.all_dims.len(), 9);
    assert_eq!(graph.all_lookups.len(), 0);
}

#[test]
fn test_program_graph_captures_lookups() {
    let mut builder = GraphBuilder::new();

    let value = Expression::from_dim(builder.position);
    let result = builder.fetch(value, None, None, None, TieBreak::Latest);

    let graph = builder.build(vec![], vec![result]);

    // 4 inputs + 3 (multiply for zero key) + 1 lookup dim
    assert_eq!(graph.all_dims.len(), 8);
    assert_eq!(graph.all_lookups.len(), 1);

    let lookup = graph.all_lookups.values().next().unwrap();
    assert_eq!(lookup.value_exprs.len(), 1);
    assert_eq!(lookup.dim_ids.len(), 1);
    assert_eq!(lookup.tie_break, TieBreak::Latest);
}

// ── IntoExpr Tests ──────────────────────────────────────────

#[test]
fn test_into_expr_expression() {
    let expr = Expression::from_dim(5);
    let result = expr.clone().into_expr(0);
    assert_eq!(result, expr);
}

#[test]
fn test_into_expr_dim_id() {
    let result = 42u32.into_expr(0);
    assert_eq!(result.get(42), 1.0);
}

#[test]
fn test_into_expr_f64_nonzero() {
    let result = 3.5f64.into_expr(1);
    assert_eq!(result.get(1), 3.5);
}

#[test]
fn test_into_expr_f64_zero() {
    let result = 0.0f64.into_expr(1);
    assert!(result.is_zero());
}

#[test]
fn test_into_expr_i32() {
    let result = 5i32.into_expr(1);
    assert_eq!(result.get(1), 5.0);
}

// ── Naming Tests ────────────────────────────────────────────

#[test]
fn test_name_dim() {
    let mut builder = GraphBuilder::new();

    let r = builder.reglu(1.0_f64, 2.0_f64);
    let dim_id = *r.terms.keys().next().unwrap();

    builder.name_dim(dim_id, "my_gate");

    let dim = builder.get_dim(dim_id).unwrap();
    assert_eq!(dim.name, "my_gate");
}

#[test]
fn test_name_dim_skips_input() {
    let mut builder = GraphBuilder::new();

    builder.name_dim(builder.one, "renamed");

    let dim = builder.get_dim(builder.one).unwrap();
    assert_eq!(dim.name, "one"); // Should NOT be renamed
}

#[test]
fn test_auto_name() {
    let mut builder = GraphBuilder::new();

    let r = builder.reglu(1.0_f64, 2.0_f64);
    let p = builder.persist(r.clone());

    builder.auto_name(&[("my_output".to_string(), p.clone())]);

    // The persist dim inside p should be named "my_output"
    let persist_id = *p.terms.keys().next().unwrap();
    let persist_dim = builder.get_dim(persist_id).unwrap();
    assert_eq!(persist_dim.name, "my_output");
}

// ── Integration: Simple Graphs ──────────────────────────────

#[test]
fn test_simple_accumulator_graph() {
    let mut builder = GraphBuilder::new();

    let values = vec![Expression::from_dim(builder.position)];
    let results = builder.fetch_sum(values);

    let graph = builder.build(vec![], results);

    // 4 inputs + 3 (multiply for zero key) + 1 lookup dim + 1 ReGLU
    assert_eq!(graph.all_dims.len(), 9);
    assert_eq!(graph.all_lookups.len(), 1);

    let lookup = graph.all_lookups.values().next().unwrap();
    assert_eq!(lookup.value_exprs.len(), 1);
    assert_eq!(lookup.dim_ids.len(), 1);
    assert_eq!(lookup.tie_break, TieBreak::Average);
}

#[test]
fn test_expression_arithmetic_chain() {
    let one_id = 0u32;
    let pos_id = 1u32;

    let pos = Expression::from_dim(pos_id);
    let one = Expression::from_dim(one_id);

    // (position + 1) * 2
    let result = (pos.clone() + one.clone()) * 2.0;
    assert_eq!(result.get(pos_id), 2.0);
    assert_eq!(result.get(one_id), 2.0);

    // position - 1
    let result2 = pos - one;
    assert_eq!(result2.get(pos_id), 1.0);
    assert_eq!(result2.get(one_id), -1.0);
}

#[test]
fn test_reglu_with_dim_expr() {
    let mut builder = GraphBuilder::new();
    let pos = builder.position;
    let one = builder.one;

    // reglu(position, position + 1)
    let pos_expr = Expression::from_dim(pos);
    let b_expr = pos_expr.clone() + Expression::from_scalar(1.0, one);
    let result = builder.reglu(pos_expr, b_expr);

    let dim_id = *result.terms.keys().next().unwrap();
    let dim = builder.get_dim(dim_id).unwrap();
    match &dim.kind {
        DimensionKind::ReGLU { a_expr, b_expr } => {
            assert_eq!(a_expr.get(pos), 1.0);
            assert_eq!(b_expr.get(pos), 1.0);
            assert_eq!(b_expr.get(one), 1.0);
        }
        _ => panic!("Expected ReGLU dimension"),
    }
}

#[test]
fn test_stepglu_creates_correct_structure() {
    let mut builder = GraphBuilder::new();

    let result = builder.stepglu(5.0_f64, 3.0_f64);
    let persist_id = *result.terms.keys().next().unwrap();
    let persist_dim = builder.get_dim(persist_id).unwrap();

    match &persist_dim.kind {
        DimensionKind::Persist { expr } => {
            // Should have +1 coefficient for first ReGLU and -1 for second
            let mut total_coeff = 0.0;
            for c in expr.terms.values() {
                total_coeff += c;
            }
            assert_eq!(total_coeff, 0.0); // +1 + (-1) = 0
            assert_eq!(expr.len(), 2);
        }
        _ => panic!("Expected Persist dimension"),
    }
}

#[test]
fn test_fetch_with_clear_key() {
    let mut builder = GraphBuilder::new();

    let value = Expression::from_dim(builder.position);
    let clear_key = Expression::from_dim(builder.one);

    let result = builder.fetch(value, None, None, Some(clear_key), TieBreak::Latest);

    assert_eq!(result.len(), 1);
    assert_eq!(builder.lookup_count(), 1);
}

#[test]
fn test_multiple_graphs_independent() {
    // Each GraphBuilder is independent — no global state
    let mut builder1 = GraphBuilder::new();
    let mut builder2 = GraphBuilder::new();

    let r1 = builder1.reglu(1.0_f64, 2.0_f64);
    let r2 = builder2.reglu(3.0_f64, 4.0_f64);

    // IDs start from 0 in each builder
    assert_eq!(builder1.one, 0);
    assert_eq!(builder2.one, 0);

    // Dimensions are independent
    let dim1 = builder1.get_dim(*r1.terms.keys().next().unwrap()).unwrap();
    let dim2 = builder2.get_dim(*r2.terms.keys().next().unwrap()).unwrap();

    match (&dim1.kind, &dim2.kind) {
        (
            DimensionKind::ReGLU {
                a_expr: a1,
                b_expr: b1,
            },
            DimensionKind::ReGLU {
                a_expr: a2,
                b_expr: b2,
            },
        ) => {
            assert_eq!(a1.get(builder1.one), 1.0);
            assert_eq!(a2.get(builder2.one), 3.0);
            assert_eq!(b1.get(builder1.one), 2.0);
            assert_eq!(b2.get(builder2.one), 4.0);
        }
        _ => panic!("Expected ReGLU dimensions"),
    }
}
