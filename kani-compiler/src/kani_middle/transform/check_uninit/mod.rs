// Copyright Kani Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! Implement a transformation pass that instruments the code to detect possible UB due to
//! the accesses to uninitialized memory.

use crate::args::ExtraChecks;
use crate::kani_middle::find_fn_def;
use crate::kani_middle::transform::body::{
    CheckType, InsertPosition, MutableBody, SourceInstruction,
};
use crate::kani_middle::transform::{TransformPass, TransformationType};
use crate::kani_queries::QueryDb;
use rustc_middle::ty::TyCtxt;
use rustc_smir::rustc_internal;
use stable_mir::mir::mono::Instance;
use stable_mir::mir::{
    AggregateKind, BasicBlock, Body, ConstOperand, Mutability, Operand, Place, Rvalue, Statement,
    StatementKind, Terminator, TerminatorKind, UnwindAction,
};
use stable_mir::ty::{
    FnDef, GenericArgKind, GenericArgs, MirConst, RigidTy, Ty, TyConst, TyKind, UintTy,
};
use stable_mir::CrateDef;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use tracing::{debug, trace};

mod ty_layout;
mod uninit_visitor;

pub use ty_layout::{PointeeInfo, PointeeLayout};
use uninit_visitor::{CheckUninitVisitor, InitRelevantInstruction, MemoryInitOp};

const SKIPPED_DIAGNOSTIC_ITEMS: &[&str] = &[
    "KaniIsUnitPtrInitialized",
    "KaniSetUnitPtrInitialized",
    "KaniIsPtrInitialized",
    "KaniSetPtrInitialized",
    "KaniIsSlicePtrInitialized",
    "KaniSetSlicePtrInitialized",
    "KaniIsStrPtrInitialized",
    "KaniSetStrPtrInitialized",
    "KaniMemInitInnerGet",
    "KaniMemInitInnerSet",
    "KaniInitializeMemoryInitializationState",
];

/// Instrument the code with checks for uninitialized memory.
#[derive(Debug)]
pub struct UninitPass {
    pub check_type: CheckType,
    /// Used to cache FnDef lookups of injected memory initialization functions.
    pub mem_init_fn_cache: HashMap<&'static str, FnDef>,
}

impl TransformPass for UninitPass {
    fn transformation_type() -> TransformationType
    where
        Self: Sized,
    {
        TransformationType::Instrumentation
    }

    fn is_enabled(&self, query_db: &QueryDb) -> bool
    where
        Self: Sized,
    {
        let args = query_db.args();
        args.ub_check.contains(&ExtraChecks::Uninit)
    }

    fn transform(&mut self, tcx: TyCtxt, body: Body, instance: Instance) -> (bool, Body) {
        trace!(function=?instance.name(), "transform");

        // Need to break infinite recursion when memory initialization checks are inserted, so the
        // internal functions responsible for memory initialization are skipped.
        if tcx
            .get_diagnostic_name(rustc_internal::internal(tcx, instance.def.def_id()))
            .map(|diagnostic_name| {
                SKIPPED_DIAGNOSTIC_ITEMS.contains(&diagnostic_name.to_ident_string().as_str())
            })
            .unwrap_or(false)
        {
            return (false, body);
        }

        // if format!("{:?}", body.span).contains("src/rust/library/core/src/ptr") {
        //     return (false, body);
        // }

        let mut new_body = MutableBody::from(body);
        let orig_len = new_body.blocks().len();

        // Inject a call to set-up memory initialization state if the function is a harness.
        if is_harness(instance, tcx) {
            inject_memory_init_setup(&mut new_body, tcx, &mut self.mem_init_fn_cache);
        }

        // Set of basic block indices for which analyzing first statement should be skipped.
        //
        // This is necessary because some checks are inserted before the source instruction, which, in
        // turn, gets moved to the next basic block. Hence, we would not need to look at the
        // instruction again as a part of new basic block. However, if the check is inserted after the
        // source instruction, we still need to look at the first statement of the new basic block, so
        // we need to keep track of which basic blocks were created as a part of injecting checks after
        // the source instruction.
        let mut skip_first = HashSet::new();

        let mut autogenerated_bbs = HashSet::new();

        // Do not cache body.blocks().len() since it will change as we add new checks.
        let mut bb_idx = 0;
        while bb_idx < new_body.blocks().len() {
            if !autogenerated_bbs.contains(&bb_idx) {
                if let Some(candidate) =
                    CheckUninitVisitor::find_next(&new_body, bb_idx, skip_first.contains(&bb_idx))
                {
                    let blocks_len_before = new_body.blocks().len();
                    self.build_check_for_instruction(
                        tcx,
                        &mut new_body,
                        candidate,
                        &mut skip_first,
                        &mut autogenerated_bbs,
                    );
                    let blocks_len_after = new_body.blocks().len();
                    // println!("bb_len: {} -> {}", blocks_len_before, blocks_len_after);
                    // for i in blocks_len_before..blocks_len_after {
                    //     println!(
                    //         "bb{}, skip_first = {}, autogenerated = {}",
                    //         i,
                    //         skip_first.contains(&i),
                    //         autogenerated_bbs.contains(&i)
                    //     );
                    //     println!("{:#?}", new_body.blocks()[i]);
                    // }
                }
            }
            bb_idx += 1;
        }
        (orig_len != new_body.blocks().len(), new_body.into())
    }
}

impl UninitPass {
    /// Inject memory initialization checks for each operation in an instruction.
    fn build_check_for_instruction(
        &mut self,
        tcx: TyCtxt,
        body: &mut MutableBody,
        instruction: InitRelevantInstruction,
        skip_first: &mut HashSet<usize>,
        autogenerated_bbs: &mut HashSet<usize>,
    ) {
        debug!(?instruction, "build_check");
        let mut source = instruction.source;
        for operation in instruction.before_instruction {
            self.build_check_for_operation(
                tcx,
                body,
                &mut source,
                operation,
                skip_first,
                autogenerated_bbs,
            );
        }
        for operation in instruction.after_instruction {
            self.build_check_for_operation(
                tcx,
                body,
                &mut source,
                operation,
                skip_first,
                autogenerated_bbs,
            );
        }
    }

    /// Inject memory initialization check for an operation.
    fn build_check_for_operation(
        &mut self,
        tcx: TyCtxt,
        body: &mut MutableBody,
        source: &mut SourceInstruction,
        operation: MemoryInitOp,
        skip_first: &mut HashSet<usize>,
        autogenerated_bbs: &mut HashSet<usize>,
    ) {
        if let MemoryInitOp::Unsupported { reason } = &operation {
            self.unsupported_check(
                body,
                source,
                operation.position(),
                reason,
                skip_first,
                autogenerated_bbs,
            );
            return;
        };

        let pointee_ty_info = {
            let ptr_operand_ty = operation.operand_ty(body);
            let pointee_ty = match ptr_operand_ty.kind().rigid().unwrap() {
                RigidTy::RawPtr(pointee_ty, _) | RigidTy::Ref(_, pointee_ty, _) => {
                    pointee_ty.clone()
                }
                _ => {
                    unreachable!(
                        "Should only build checks for raw pointers, `{ptr_operand_ty}` encountered."
                    )
                }
            };
            match PointeeInfo::from_ty(pointee_ty) {
                Ok(type_info) => type_info,
                Err(_) => {
                    let reason = format!(
                        "Kani currently doesn't support checking memory initialization for pointers to `{pointee_ty}.",
                    );
                    self.unsupported_check(
                        body,
                        source,
                        operation.position(),
                        &reason,
                        skip_first,
                        autogenerated_bbs,
                    );
                    return;
                }
            }
        };

        let new_bb = match &operation {
            MemoryInitOp::Check { .. } | MemoryInitOp::CheckRef { .. } => {
                self.build_get_and_check(tcx, body, source, operation.clone(), pointee_ty_info)
            }
            MemoryInitOp::Set { .. } | MemoryInitOp::SetRef { .. } => {
                self.build_set(tcx, body, source, operation.clone(), pointee_ty_info)
            }
            MemoryInitOp::Unsupported { .. } => {
                unreachable!()
            }
        };
        collect_skipped(operation.position(), source, body, skip_first);
        body.add_bb(new_bb, source, operation.position());
        autogenerated_bbs.insert(body.blocks().len() - 1);
    }

    /// Inject a load from memory initialization state and an assertion that all non-padding bytes
    /// are initialized.
    fn build_get_and_check(
        &mut self,
        tcx: TyCtxt,
        body: &mut MutableBody,
        source: &mut SourceInstruction,
        operation: MemoryInitOp,
        pointee_info: PointeeInfo,
    ) -> BasicBlock {
        let ret_place = Place {
            local: body.new_local(Ty::new_tuple(&[]), source.span(body.blocks()), Mutability::Not),
            projection: vec![],
        };

        let mut new_bb = BasicBlock {
            statements: vec![],
            terminator: Terminator {
                kind: TerminatorKind::Return,
                span: source.span(body.blocks()),
            },
        };

        let ptr_operand = operation.mk_operand(body, &mut new_bb, source);
        match pointee_info.layout() {
            PointeeLayout::Sized { layout } => {
                let is_ptr_initialized_instance = resolve_mem_init_fn(
                    get_mem_init_fn_def(tcx, "KaniIsPtrInitialized", &mut self.mem_init_fn_cache),
                    layout.len(),
                    *pointee_info.ty(),
                );
                let layout_operand = mk_layout_operand(body, &mut new_bb, source, &layout);

                new_bb.terminator = Terminator {
                    kind: TerminatorKind::Call {
                        func: Operand::Copy(Place::from(body.new_local(
                            is_ptr_initialized_instance.ty(),
                            source.span(body.blocks()),
                            Mutability::Not,
                        ))),
                        args: vec![ptr_operand.clone(), layout_operand, operation.expect_count()],
                        destination: ret_place.clone(),
                        target: None, // this will be overriden in add_bb
                        unwind: UnwindAction::Terminate,
                    },
                    span: source.span(body.blocks()),
                };
            }
            PointeeLayout::Slice { element_layout } => {
                // Since `str`` is a separate type, need to differentiate between [T] and str.
                let (slicee_ty, diagnostic) = match pointee_info.ty().kind() {
                    TyKind::RigidTy(RigidTy::Slice(slicee_ty)) => {
                        (slicee_ty, "KaniIsSlicePtrInitialized")
                    }
                    TyKind::RigidTy(RigidTy::Str) => {
                        (Ty::unsigned_ty(UintTy::U8), "KaniIsStrPtrInitialized")
                    }
                    _ => unreachable!(),
                };
                let is_ptr_initialized_instance = resolve_mem_init_fn(
                    get_mem_init_fn_def(tcx, diagnostic, &mut self.mem_init_fn_cache),
                    element_layout.len(),
                    slicee_ty,
                );
                let layout_operand = mk_layout_operand(body, &mut new_bb, source, &element_layout);

                new_bb.terminator = Terminator {
                    kind: TerminatorKind::Call {
                        func: Operand::Copy(Place::from(body.new_local(
                            is_ptr_initialized_instance.ty(),
                            source.span(body.blocks()),
                            Mutability::Not,
                        ))),
                        args: vec![ptr_operand.clone(), layout_operand],
                        destination: ret_place.clone(),
                        target: None, // this will be overriden in add_bb
                        unwind: UnwindAction::Terminate,
                    },
                    span: source.span(body.blocks()),
                };
            }
            PointeeLayout::TraitObject => {
                unreachable!(
                    "Kani does not support reasoning about memory initialization of pointers to trait objects."
                );
            }
        };
        new_bb
    }

    /// Inject a store into memory initialization state to initialize or deinitialize all
    /// non-padding bytes.
    fn build_set(
        &mut self,
        tcx: TyCtxt,
        body: &mut MutableBody,
        source: &mut SourceInstruction,
        operation: MemoryInitOp,
        pointee_info: PointeeInfo,
    ) -> BasicBlock {
        let ret_place = Place {
            local: body.new_local(Ty::new_tuple(&[]), source.span(body.blocks()), Mutability::Not),
            projection: vec![],
        };

        let mut new_bb = BasicBlock {
            statements: vec![],
            terminator: Terminator {
                kind: TerminatorKind::Return,
                span: source.span(body.blocks()),
            },
        };

        let ptr_operand = operation.mk_operand(body, &mut new_bb, source);
        let value = operation.expect_value();

        match pointee_info.layout() {
            PointeeLayout::Sized { layout } => {
                let set_ptr_initialized_instance = resolve_mem_init_fn(
                    get_mem_init_fn_def(tcx, "KaniSetPtrInitialized", &mut self.mem_init_fn_cache),
                    layout.len(),
                    *pointee_info.ty(),
                );
                let layout_operand = mk_layout_operand(body, &mut new_bb, source, &layout);

                new_bb.terminator = Terminator {
                    kind: TerminatorKind::Call {
                        func: Operand::Copy(Place::from(body.new_local(
                            set_ptr_initialized_instance.ty(),
                            source.span(body.blocks()),
                            Mutability::Not,
                        ))),
                        args: vec![
                            ptr_operand,
                            layout_operand,
                            operation.expect_count(),
                            Operand::Constant(ConstOperand {
                                span: source.span(body.blocks()),
                                user_ty: None,
                                const_: MirConst::from_bool(value),
                            }),
                        ],
                        destination: ret_place.clone(),
                        target: None, // this will be overriden in add_bb
                        unwind: UnwindAction::Terminate,
                    },
                    span: source.span(body.blocks()),
                };
            }
            PointeeLayout::Slice { element_layout } => {
                // Since `str`` is a separate type, need to differentiate between [T] and str.
                let (slicee_ty, diagnostic) = match pointee_info.ty().kind() {
                    TyKind::RigidTy(RigidTy::Slice(slicee_ty)) => {
                        (slicee_ty, "KaniSetSlicePtrInitialized")
                    }
                    TyKind::RigidTy(RigidTy::Str) => {
                        (Ty::unsigned_ty(UintTy::U8), "KaniSetStrPtrInitialized")
                    }
                    _ => unreachable!(),
                };
                let set_ptr_initialized_instance = resolve_mem_init_fn(
                    get_mem_init_fn_def(tcx, diagnostic, &mut self.mem_init_fn_cache),
                    element_layout.len(),
                    slicee_ty,
                );
                let layout_operand = mk_layout_operand(body, &mut new_bb, source, &element_layout);

                new_bb.terminator = Terminator {
                    kind: TerminatorKind::Call {
                        func: Operand::Copy(Place::from(body.new_local(
                            set_ptr_initialized_instance.ty(),
                            source.span(body.blocks()),
                            Mutability::Not,
                        ))),
                        args: vec![
                            ptr_operand,
                            layout_operand,
                            Operand::Constant(ConstOperand {
                                span: source.span(body.blocks()),
                                user_ty: None,
                                const_: MirConst::from_bool(value),
                            }),
                        ],
                        destination: ret_place.clone(),
                        target: None, // this will be overriden in add_bb
                        unwind: UnwindAction::Terminate,
                    },
                    span: source.span(body.blocks()),
                };
            }
            PointeeLayout::TraitObject => {
                unreachable!("Cannot change the initialization state of a trait object directly.");
            }
        };
        new_bb
    }

    fn unsupported_check(
        &self,
        body: &mut MutableBody,
        source: &mut SourceInstruction,
        position: InsertPosition,
        reason: &str,
        skip_first: &mut HashSet<usize>,
        autogenerated_bbs: &mut HashSet<usize>,
    ) {
        let span = source.span(body.blocks());
        let rvalue = Rvalue::Use(Operand::Constant(ConstOperand {
            const_: MirConst::from_bool(false),
            span,
            user_ty: None,
        }));
        let ret_ty = rvalue.ty(body.locals()).unwrap();
        let result = body.new_local(ret_ty, span, Mutability::Not);
        let stmt = Statement { kind: StatementKind::Assign(Place::from(result), rvalue), span };

        let CheckType::Assert(assert_fn) = self.check_type else { unimplemented!() };
        let assert_op =
            Operand::Copy(Place::from(body.new_local(assert_fn.ty(), span, Mutability::Not)));
        let msg_op = body.new_str_operand(reason, span);
        let kind = TerminatorKind::Call {
            func: assert_op,
            args: vec![Operand::Move(Place::from(result)), msg_op],
            destination: Place {
                local: body.new_local(Ty::new_tuple(&[]), span, Mutability::Not),
                projection: vec![],
            },
            target: None, // this will be overwritten by add_bb
            unwind: UnwindAction::Terminate,
        };
        let terminator = Terminator { kind, span };
        let new_bb = BasicBlock { statements: vec![stmt], terminator };

        collect_skipped(position, source, body, skip_first);
        body.add_bb(new_bb, source, position);
        autogenerated_bbs.insert(body.blocks().len() - 1);
    }
}

/// Create an operand from a bit array that represents a byte mask for a type layout where padding
/// bytes are marked as `false` and data bytes are marked as `true`.
///
/// For example, the layout for:
/// ```
/// [repr(C)]
/// struct {
///     a: u16,
///     b: u8
/// }
/// ```
/// will have the following byte mask `[true, true, true, false]`.
pub fn mk_layout_operand(
    body: &mut MutableBody,
    bb: &mut BasicBlock,
    source: &mut SourceInstruction,
    layout_byte_mask: &[bool],
) -> Operand {
    let span = source.span(body.blocks());
    let rvalue = Rvalue::Aggregate(
        AggregateKind::Array(Ty::bool_ty()),
        layout_byte_mask
            .iter()
            .map(|byte| {
                Operand::Constant(ConstOperand {
                    span: source.span(body.blocks()),
                    user_ty: None,
                    const_: MirConst::from_bool(*byte),
                })
            })
            .collect(),
    );
    let ret_ty = rvalue.ty(body.locals()).unwrap();
    let result = body.new_local(ret_ty, span, Mutability::Not);
    let stmt = Statement { kind: StatementKind::Assign(Place::from(result), rvalue), span };
    bb.statements.push(stmt);

    Operand::Move(Place { local: result, projection: vec![] })
}

/// If injecting a new call to the function before the current statement, need to skip the original
/// statement when analyzing it as a part of the new basic block.
fn collect_skipped(
    position: InsertPosition,
    source: &SourceInstruction,
    body: &MutableBody,
    skip_first: &mut HashSet<usize>,
) {
    if position == InsertPosition::Before
        || (position == InsertPosition::After && source.is_terminator())
    {
        let new_bb_idx = body.blocks().len();
        skip_first.insert(new_bb_idx);
    }
}

/// Retrieve a function definition by diagnostic string, caching the result.
pub fn get_mem_init_fn_def(
    tcx: TyCtxt,
    diagnostic: &'static str,
    cache: &mut HashMap<&'static str, FnDef>,
) -> FnDef {
    let entry = cache.entry(diagnostic).or_insert_with(|| find_fn_def(tcx, diagnostic).unwrap());
    *entry
}

/// Resolves a given memory initialization function with passed type parameters.
pub fn resolve_mem_init_fn(fn_def: FnDef, layout_size: usize, associated_type: Ty) -> Instance {
    Instance::resolve(
        fn_def,
        &GenericArgs(vec![
            GenericArgKind::Const(TyConst::try_from_target_usize(layout_size as u64).unwrap()),
            GenericArgKind::Type(associated_type),
        ]),
    )
    .unwrap()
}

/// Checks if the instance is a harness -- an entry point of Kani analysis.
fn is_harness(instance: Instance, tcx: TyCtxt) -> bool {
    let harness_identifiers = [
        vec![
            rustc_span::symbol::Symbol::intern("kanitool"),
            rustc_span::symbol::Symbol::intern("proof_for_contract"),
        ],
        vec![
            rustc_span::symbol::Symbol::intern("kanitool"),
            rustc_span::symbol::Symbol::intern("proof"),
        ],
    ];
    harness_identifiers.iter().any(|attr_path| {
        tcx.has_attrs_with_path(rustc_internal::internal(tcx, instance.def.def_id()), attr_path)
    })
}

/// Inject an initial call to set-up memory initialization tracking.
fn inject_memory_init_setup(
    new_body: &mut MutableBody,
    tcx: TyCtxt,
    mem_init_fn_cache: &mut HashMap<&'static str, FnDef>,
) {
    // First statement or terminator in the harness.
    let mut source = if !new_body.blocks()[0].statements.is_empty() {
        SourceInstruction::Statement { idx: 0, bb: 0 }
    } else {
        SourceInstruction::Terminator { bb: 0 }
    };

    // Dummy return place.
    let ret_place = Place {
        local: new_body.new_local(
            Ty::new_tuple(&[]),
            source.span(new_body.blocks()),
            Mutability::Not,
        ),
        projection: vec![],
    };

    // Resolve the instance and inject a call to set-up the memory initialization state.
    let memory_initialization_init = Instance::resolve(
        get_mem_init_fn_def(tcx, "KaniInitializeMemoryInitializationState", mem_init_fn_cache),
        &GenericArgs(vec![]),
    )
    .unwrap();

    new_body.add_call(
        &memory_initialization_init,
        &mut source,
        InsertPosition::Before,
        vec![],
        ret_place,
    );
}
