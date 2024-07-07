// Copyright Kani Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! Visitor that collects all instructions relevant to uninitialized memory access.

use crate::kani_middle::transform::body::{InsertPosition, MutableBody, SourceInstruction};
use stable_mir::mir::alloc::GlobalAlloc;
use stable_mir::mir::mono::{Instance, InstanceKind};
use stable_mir::mir::visit::{Location, PlaceContext};
use stable_mir::mir::{
    BasicBlock, BasicBlockIdx, CastKind, ConstOperand, LocalDecl, MirVisitor, Mutability,
    NonDivergingIntrinsic, Operand, Place, PointerCoercion, ProjectionElem, Rvalue, Statement,
    StatementKind, Terminator, TerminatorKind,
};
use stable_mir::ty::{ConstantKind, MirConst, RigidTy, Span, Ty, TyKind, UintTy};
use strum_macros::AsRefStr;

/// Memory initialization operations: set or get memory initialization state for a given pointer.
#[derive(AsRefStr, Clone, Debug)]
pub enum MemoryInitOp {
    /// Check memory initialization of data bytes in a memory region starting from the pointer
    /// `operand` and of length `count * sizeof(operand)` bytes.
    Check { operand: Operand, count: Operand },
    /// Set memory initialization state of data bytes in a memory region starting from the pointer
    /// `operand` and of length `count * sizeof(operand)` bytes.
    Set { operand: Operand, count: Operand, value: bool, position: InsertPosition },
    /// Check memory initialization of data bytes in a memory region starting from the reference to
    /// `operand` and of length `count * sizeof(operand)` bytes.
    CheckRef { operand: Operand, count: Operand },
    /// Set memory initialization of data bytes in a memory region starting from the reference to
    /// `operand` and of length `count * sizeof(operand)` bytes.
    SetRef { operand: Operand, count: Operand, value: bool, position: InsertPosition },
    /// Unsupported memory initialization operation.
    Unsupported { reason: String },
}

impl MemoryInitOp {
    /// Produce an operand for the relevant memory initialization related operation. This is mostly
    /// required so that the analysis can create a new local to take a reference in
    /// `MemoryInitOp::SetRef`.
    pub fn mk_operand(
        &self,
        body: &mut MutableBody,
        bb: &mut BasicBlock,
        source: &mut SourceInstruction,
    ) -> Operand {
        match self {
            MemoryInitOp::Check { operand, .. } | MemoryInitOp::Set { operand, .. } => {
                let span = source.span(body.blocks());

                let appropriate_unit_ptr_ty =
                    match operand.ty(body.locals()).unwrap().kind().rigid().unwrap() {
                        RigidTy::Ref(_, pointee_ty, _) | RigidTy::RawPtr(pointee_ty, _) => {
                            match pointee_ty.kind().rigid().unwrap() {
                                RigidTy::Slice(..) | RigidTy::Str => {
                                    Ty::from_rigid_kind(RigidTy::RawPtr(
                                        Ty::from_rigid_kind(RigidTy::Slice(Ty::from_rigid_kind(
                                            RigidTy::Tuple(vec![]),
                                        ))),
                                        Mutability::Not,
                                    ))
                                }
                                _ => Ty::from_rigid_kind(RigidTy::RawPtr(
                                    Ty::from_rigid_kind(RigidTy::Tuple(vec![])),
                                    Mutability::Not,
                                )),
                            }
                        }
                        _ => unreachable!(),
                    };

                let unit_local = {
                    let rvalue =
                        Rvalue::Cast(CastKind::Transmute, operand.clone(), appropriate_unit_ptr_ty);
                    let ret_ty = rvalue.ty(body.locals()).unwrap();
                    let result = body.new_local(ret_ty, span, Mutability::Not);
                    let stmt = Statement {
                        kind: StatementKind::Assign(Place::from(result), rvalue),
                        span,
                    };
                    bb.statements.push(stmt);
                    result
                };

                Operand::Copy(Place { local: unit_local, projection: vec![] })
            }
            MemoryInitOp::SetRef { operand, .. } | MemoryInitOp::CheckRef { operand, .. } => {
                let span = source.span(body.blocks());

                let (ref_local, ret_ty) = {
                    let place = match operand {
                        Operand::Copy(place) | Operand::Move(place) => place,
                        Operand::Constant(_) => unreachable!(),
                    };
                    let rvalue = Rvalue::AddressOf(Mutability::Not, place.clone());
                    let ret_ty = rvalue.ty(body.locals()).unwrap();
                    let result = body.new_local(ret_ty, span, Mutability::Not);
                    let stmt = Statement {
                        kind: StatementKind::Assign(Place::from(result), rvalue),
                        span,
                    };
                    bb.statements.push(stmt);
                    (result, ret_ty)
                };

                let appropriate_unit_ptr_ty = match ret_ty.kind().rigid().unwrap() {
                    RigidTy::Ref(_, pointee_ty, _) | RigidTy::RawPtr(pointee_ty, _) => {
                        match pointee_ty.kind().rigid().unwrap() {
                            RigidTy::Slice(..) | RigidTy::Str => {
                                Ty::from_rigid_kind(RigidTy::RawPtr(
                                    Ty::from_rigid_kind(RigidTy::Slice(Ty::from_rigid_kind(
                                        RigidTy::Tuple(vec![]),
                                    ))),
                                    Mutability::Not,
                                ))
                            }
                            _ => Ty::from_rigid_kind(RigidTy::RawPtr(
                                Ty::from_rigid_kind(RigidTy::Tuple(vec![])),
                                Mutability::Not,
                            )),
                        }
                    }
                    _ => unreachable!(),
                };

                let unit_local = {
                    let operand = Operand::Move(Place::from(ref_local));
                    let rvalue =
                        Rvalue::Cast(CastKind::Transmute, operand, appropriate_unit_ptr_ty);
                    let ret_ty = rvalue.ty(body.locals()).unwrap();
                    let result = body.new_local(ret_ty, span, Mutability::Not);
                    let stmt = Statement {
                        kind: StatementKind::Assign(Place::from(result), rvalue),
                        span,
                    };
                    bb.statements.push(stmt);
                    result
                };

                Operand::Copy(Place { local: unit_local, projection: vec![] })
            }
            MemoryInitOp::Unsupported { .. } => unreachable!(),
        }
    }

    pub fn operand_ty(&self, body: &MutableBody) -> Ty {
        match self {
            MemoryInitOp::Check { operand, .. } | MemoryInitOp::Set { operand, .. } => {
                operand.ty(body.locals()).unwrap()
            }
            MemoryInitOp::SetRef { operand, .. } | MemoryInitOp::CheckRef { operand, .. } => {
                let place = match operand {
                    Operand::Copy(place) | Operand::Move(place) => place,
                    Operand::Constant(_) => unreachable!(),
                };
                let rvalue = Rvalue::AddressOf(Mutability::Not, place.clone());
                rvalue.ty(body.locals()).unwrap()
            }
            MemoryInitOp::Unsupported { .. } => unreachable!(),
        }
    }

    pub fn expect_count(&self) -> Operand {
        match self {
            MemoryInitOp::Check { count, .. }
            | MemoryInitOp::Set { count, .. }
            | MemoryInitOp::CheckRef { count, .. }
            | MemoryInitOp::SetRef { count, .. } => count.clone(),
            MemoryInitOp::Unsupported { .. } => unreachable!(),
        }
    }

    pub fn expect_value(&self) -> bool {
        match self {
            MemoryInitOp::Set { value, .. } | MemoryInitOp::SetRef { value, .. } => *value,
            MemoryInitOp::Check { .. }
            | MemoryInitOp::CheckRef { .. }
            | MemoryInitOp::Unsupported { .. } => unreachable!(),
        }
    }

    pub fn position(&self) -> InsertPosition {
        match self {
            MemoryInitOp::Set { position, .. } | MemoryInitOp::SetRef { position, .. } => *position,
            MemoryInitOp::Check { .. }
            | MemoryInitOp::CheckRef { .. }
            | MemoryInitOp::Unsupported { .. } => InsertPosition::Before,
        }
    }
}

/// Represents an instruction in the source code together with all memory initialization checks/sets
/// that are connected to the memory used in this instruction and whether they should be inserted
/// before or after the instruction.
#[derive(Clone, Debug)]
pub struct InitRelevantInstruction {
    /// The instruction that affects the state of the memory.
    pub source: SourceInstruction,
    /// All memory-related operations that should happen after the instruction.
    pub before_instruction: Vec<MemoryInitOp>,
    /// All memory-related operations that should happen after the instruction.
    pub after_instruction: Vec<MemoryInitOp>,
}

impl InitRelevantInstruction {
    pub fn push_operation(&mut self, source_op: MemoryInitOp) {
        match source_op.position() {
            InsertPosition::Before => self.before_instruction.push(source_op),
            InsertPosition::After => self.after_instruction.push(source_op),
        }
    }
}

pub struct CheckUninitVisitor<'a> {
    locals: &'a [LocalDecl],
    /// Whether we should skip the next instruction, since it might've been instrumented already.
    /// When we instrument an instruction, we partition the basic block, and the instruction that
    /// may trigger UB becomes the first instruction of the basic block, which we need to skip
    /// later.
    skip_next: bool,
    /// The instruction being visited at a given point.
    current: SourceInstruction,
    /// The target instruction that should be verified.
    pub target: Option<InitRelevantInstruction>,
    /// The basic block being visited.
    bb: BasicBlockIdx,
}

impl<'a> CheckUninitVisitor<'a> {
    pub fn find_next(
        body: &'a MutableBody,
        bb: BasicBlockIdx,
        skip_first: bool,
    ) -> Option<InitRelevantInstruction> {
        let mut visitor = CheckUninitVisitor {
            locals: body.locals(),
            skip_next: skip_first,
            current: SourceInstruction::Statement { idx: 0, bb },
            target: None,
            bb,
        };
        visitor.visit_basic_block(&body.blocks()[bb]);
        visitor.target
    }

    fn push_target(&mut self, source_op: MemoryInitOp) {
        let target = self.target.get_or_insert_with(|| InitRelevantInstruction {
            source: self.current,
            after_instruction: vec![],
            before_instruction: vec![],
        });
        target.push_operation(source_op);
    }

    /// Checks memory initialization of inner dereference projections.
    fn check_inner_derefs(&mut self, place: &Place, ptx: PlaceContext, location: Location) {
        for (idx, elem) in place.projection.iter().enumerate() {
            let intermediate_place =
                Place { local: place.local, projection: place.projection[..idx].to_vec() };
            match elem {
                ProjectionElem::Deref => {
                    self.push_target(MemoryInitOp::Check {
                        operand: Operand::Copy(intermediate_place.clone()),
                        count: mk_const_operand(1, location.span()),
                    });
                }
                ProjectionElem::Field(idx, target_ty) => {
                    if target_ty.kind().is_union()
                        && (!ptx.is_mutating() || place.projection.len() > idx + 1)
                    {
                        self.push_target(MemoryInitOp::Unsupported {
                            reason: "Kani does not support reasoning about memory initialization of unions.".to_string(),
                        });
                    }
                }
                ProjectionElem::Index(_)
                | ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Subslice { .. } => {}
                ProjectionElem::Downcast(_) => {}
                ProjectionElem::OpaqueCast(_) => {}
                ProjectionElem::Subtype(_) => {}
            }
        }
    }
}

impl<'a> MirVisitor for CheckUninitVisitor<'a> {
    fn visit_statement(&mut self, stmt: &Statement, location: Location) {
        if self.skip_next {
            self.skip_next = false;
        } else if self.target.is_none() {
            match &stmt.kind {
                // We only care about intrinsics, because everything else is handled by checking
                // places inside `super_statement`.
                StatementKind::Intrinsic(non_diverging_intrinsic) => {
                    match non_diverging_intrinsic {
                        NonDivergingIntrinsic::CopyNonOverlapping(copy) => {
                            self.super_statement(stmt, location);
                            // Source is a *const T and it must be initialized.
                            self.push_target(MemoryInitOp::Check {
                                operand: copy.src.clone(),
                                count: copy.count.clone(),
                            });
                            // Destimation is a *mut T so it gets initialized.
                            self.push_target(MemoryInitOp::Set {
                                operand: copy.dst.clone(),
                                count: copy.count.clone(),
                                value: true,
                                position: InsertPosition::After,
                            });
                        }
                        NonDivergingIntrinsic::Assume(_) => {}
                    }
                }
                StatementKind::Assign(..) => {
                    self.super_statement(stmt, location);
                }
                _ => {}
            }
        }
        let SourceInstruction::Statement { idx, bb } = self.current else { unreachable!() };
        self.current = SourceInstruction::Statement { idx: idx + 1, bb };
    }

    fn visit_terminator(&mut self, term: &Terminator, location: Location) {
        if !(self.skip_next || self.target.is_some()) {
            self.current = SourceInstruction::Terminator { bb: self.bb };
            // Leave it as an exhaustive match to be notified when a new kind is added.
            match &term.kind {
                TerminatorKind::Call { func, args, destination, target: Some(_), .. } => {
                    self.super_terminator(term, location);
                    let instance = match try_resolve_instance(self.locals, func) {
                        Ok(instance) => instance,
                        Err(reason) => {
                            self.push_target(MemoryInitOp::Unsupported { reason });
                            return;
                        }
                    };
                    match instance.kind {
                        InstanceKind::Intrinsic => {
                            match instance.intrinsic_name().unwrap().as_str() {
                                intrinsic_name if can_skip_intrinsic(intrinsic_name) => {
                                    /* Intrinsics that can be safely skipped */
                                }
                                name if name.starts_with("atomic") => {
                                    let num_args = match name {
                                        // All `atomic_cxchg` intrinsics take `dst, old, src` as arguments.
                                        name if name.starts_with("atomic_cxchg") => 3,
                                        // All `atomic_load` intrinsics take `src` as an argument.
                                        name if name.starts_with("atomic_load") => 1,
                                        // All other `atomic` intrinsics take `dst, src` as arguments.
                                        _ => 2,
                                    };
                                    assert_eq!(
                                        args.len(),
                                        num_args,
                                        "Unexpected number of arguments for `{name}`"
                                    );
                                    assert!(matches!(
                                        args[0].ty(self.locals).unwrap().kind(),
                                        TyKind::RigidTy(RigidTy::RawPtr(..))
                                    ));
                                    self.push_target(MemoryInitOp::Check {
                                        operand: args[0].clone(),
                                        count: mk_const_operand(1, location.span()),
                                    });
                                }
                                "compare_bytes" => {
                                    assert_eq!(
                                        args.len(),
                                        3,
                                        "Unexpected number of arguments for `compare_bytes`"
                                    );
                                    assert!(matches!(
                                        args[0].ty(self.locals).unwrap().kind(),
                                        TyKind::RigidTy(RigidTy::RawPtr(_, Mutability::Not))
                                    ));
                                    assert!(matches!(
                                        args[1].ty(self.locals).unwrap().kind(),
                                        TyKind::RigidTy(RigidTy::RawPtr(_, Mutability::Not))
                                    ));
                                    self.push_target(MemoryInitOp::Check {
                                        operand: args[0].clone(),
                                        count: args[2].clone(),
                                    });
                                    self.push_target(MemoryInitOp::Check {
                                        operand: args[1].clone(),
                                        count: args[2].clone(),
                                    });
                                }
                                "copy"
                                | "volatile_copy_memory"
                                | "volatile_copy_nonoverlapping_memory" => {
                                    assert_eq!(
                                        args.len(),
                                        3,
                                        "Unexpected number of arguments for `copy`"
                                    );
                                    assert!(matches!(
                                        args[0].ty(self.locals).unwrap().kind(),
                                        TyKind::RigidTy(RigidTy::RawPtr(_, Mutability::Not))
                                    ));
                                    assert!(matches!(
                                        args[1].ty(self.locals).unwrap().kind(),
                                        TyKind::RigidTy(RigidTy::RawPtr(_, Mutability::Mut))
                                    ));
                                    self.push_target(MemoryInitOp::Check {
                                        operand: args[0].clone(),
                                        count: args[2].clone(),
                                    });
                                    self.push_target(MemoryInitOp::Set {
                                        operand: args[1].clone(),
                                        count: args[2].clone(),
                                        value: true,
                                        position: InsertPosition::After,
                                    });
                                }
                                "typed_swap" => {
                                    assert_eq!(
                                        args.len(),
                                        2,
                                        "Unexpected number of arguments for `typed_swap`"
                                    );
                                    assert!(matches!(
                                        args[0].ty(self.locals).unwrap().kind(),
                                        TyKind::RigidTy(RigidTy::RawPtr(_, Mutability::Mut))
                                    ));
                                    assert!(matches!(
                                        args[1].ty(self.locals).unwrap().kind(),
                                        TyKind::RigidTy(RigidTy::RawPtr(_, Mutability::Mut))
                                    ));
                                    self.push_target(MemoryInitOp::Check {
                                        operand: args[0].clone(),
                                        count: mk_const_operand(1, location.span()),
                                    });
                                    self.push_target(MemoryInitOp::Check {
                                        operand: args[1].clone(),
                                        count: mk_const_operand(1, location.span()),
                                    });
                                }
                                "volatile_load" | "unaligned_volatile_load" => {
                                    assert_eq!(
                                        args.len(),
                                        1,
                                        "Unexpected number of arguments for `volatile_load`"
                                    );
                                    assert!(matches!(
                                        args[0].ty(self.locals).unwrap().kind(),
                                        TyKind::RigidTy(RigidTy::RawPtr(_, Mutability::Not))
                                    ));
                                    self.push_target(MemoryInitOp::Check {
                                        operand: args[0].clone(),
                                        count: mk_const_operand(1, location.span()),
                                    });
                                }
                                "volatile_store" => {
                                    assert_eq!(
                                        args.len(),
                                        2,
                                        "Unexpected number of arguments for `volatile_store`"
                                    );
                                    assert!(matches!(
                                        args[0].ty(self.locals).unwrap().kind(),
                                        TyKind::RigidTy(RigidTy::RawPtr(_, Mutability::Mut))
                                    ));
                                    self.push_target(MemoryInitOp::Set {
                                        operand: args[0].clone(),
                                        count: mk_const_operand(1, location.span()),
                                        value: true,
                                        position: InsertPosition::After,
                                    });
                                }
                                "write_bytes" => {
                                    assert_eq!(
                                        args.len(),
                                        3,
                                        "Unexpected number of arguments for `write_bytes`"
                                    );
                                    assert!(matches!(
                                        args[0].ty(self.locals).unwrap().kind(),
                                        TyKind::RigidTy(RigidTy::RawPtr(_, Mutability::Mut))
                                    ));
                                    self.push_target(MemoryInitOp::Set {
                                        operand: args[0].clone(),
                                        count: args[2].clone(),
                                        value: true,
                                        position: InsertPosition::After,
                                    })
                                }
                                intrinsic => {
                                    self.push_target(MemoryInitOp::Unsupported {
                                    reason: format!("Kani does not support reasoning about memory initialization of intrinsic `{intrinsic}`."),
                                });
                                }
                            }
                        }
                        InstanceKind::Item => {
                            if instance.is_foreign_item() {
                                match instance.name().as_str() {
                                    "alloc::alloc::__rust_alloc"
                                    | "alloc::alloc::__rust_realloc" => {
                                        /* Memory is uninitialized, nothing to do here. */
                                    }
                                    "alloc::alloc::__rust_alloc_zeroed" => {
                                        /* Memory is initialized here, need to update shadow memory. */
                                        self.push_target(MemoryInitOp::Set {
                                            operand: Operand::Copy(destination.clone()),
                                            count: args[0].clone(),
                                            value: true,
                                            position: InsertPosition::After,
                                        });
                                    }
                                    "alloc::alloc::__rust_dealloc" => {
                                        /* Memory is uninitialized here, need to update shadow memory. */
                                        self.push_target(MemoryInitOp::Set {
                                            operand: args[0].clone(),
                                            count: args[1].clone(),
                                            value: false,
                                            position: InsertPosition::After,
                                        });
                                    }
                                    _ => {
                                        /* Check memory initialization of arguments */
                                        self.super_terminator(term, location);
                                    }
                                }
                            }
                        }
                        _ => {
                            /* Check memory initialization of arguments */
                            self.super_terminator(term, location);
                        }
                    }
                }
                TerminatorKind::Drop { place, .. } => {
                    /* Check memory initialization of arguments */
                    self.super_terminator(term, location);

                    // When drop is codegen'ed for types that could define their own dropping
                    // behavior, a reference is taken to the place which is later implicitly coerced
                    // to a pointer. Hence, we need to bless this pointer as initialized.
                    match place
                        .ty(&self.locals)
                        .unwrap()
                        .kind()
                        .rigid()
                        .expect("should be working with monomorphized code")
                    {
                        RigidTy::Adt(..) | RigidTy::Dynamic(_, _, _) => {
                            self.push_target(MemoryInitOp::SetRef {
                                operand: Operand::Copy(place.clone()),
                                count: mk_const_operand(1, location.span()),
                                value: true,
                                position: InsertPosition::Before,
                            });
                        }
                        _ => {}
                    }
                }
                TerminatorKind::SwitchInt { .. } => self.super_terminator(term, location),
                TerminatorKind::Call { target: None, .. }
                | TerminatorKind::Goto { .. }
                | TerminatorKind::Resume
                | TerminatorKind::Abort
                | TerminatorKind::Return
                | TerminatorKind::Unreachable
                | TerminatorKind::InlineAsm { .. }
                | TerminatorKind::Assert { .. } => {}
            }
        }
    }

    fn visit_place(&mut self, place: &Place, ptx: PlaceContext, location: Location) {
        if ptx.is_mutating() {
            match try_remove_topmost_deref(place) {
                Some(place_without_deref) => {
                    self.check_inner_derefs(&place_without_deref, ptx, location);
                }
                None => {}
            }
            self.push_target(MemoryInitOp::SetRef {
                operand: Operand::Copy(place.clone()),
                count: mk_const_operand(1, location.span()),
                value: true,
                position: InsertPosition::After,
            });
        } else {
            self.check_inner_derefs(place, ptx, location);
            self.push_target(MemoryInitOp::CheckRef {
                operand: Operand::Copy(place.clone()),
                count: mk_const_operand(1, location.span()),
            });
        };
        self.super_place(place, ptx, location)
    }

    fn visit_operand(&mut self, operand: &Operand, location: Location) {
        if let Operand::Constant(constant) = operand {
            if let ConstantKind::Allocated(allocation) = constant.const_.kind() {
                for (_, prov) in &allocation.provenance.ptrs {
                    if let GlobalAlloc::Static(_) = GlobalAlloc::from(prov.0) {
                        if constant.ty().kind().is_raw_ptr() || constant.ty().kind().is_ref() {
                            // If a static is a raw pointer, need to mark it as initialized.
                            self.push_target(MemoryInitOp::Set {
                                operand: Operand::Constant(constant.clone()),
                                count: mk_const_operand(1, location.span()),
                                value: true,
                                position: InsertPosition::Before,
                            });
                        } else {
                            self.push_target(MemoryInitOp::SetRef {
                                operand: Operand::Constant(constant.clone()),
                                count: mk_const_operand(1, location.span()),
                                value: true,
                                position: InsertPosition::Before,
                            });
                        }
                    };
                }
            }
        }
        self.super_operand(operand, location);
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue, location: Location) {
        self.super_rvalue(rvalue, location);
        if let Rvalue::Cast(CastKind::PointerCoercion(PointerCoercion::Unsize), _, ty) = rvalue {
            if let TyKind::RigidTy(RigidTy::RawPtr(pointee_ty, _)) = ty.kind() {
                if pointee_ty.kind().is_trait() {
                    self.push_target(MemoryInitOp::Unsupported {
                        reason: "Kani does not support reasoning about memory initialization of unsized pointers.".to_string(),
                    });
                }
            }
        };
    }
}

/// Determines if the intrinsic has no memory initialization related function and hence can be
/// safely skipped.
fn can_skip_intrinsic(intrinsic_name: &str) -> bool {
    match intrinsic_name {
        "add_with_overflow"
        | "arith_offset"
        | "assert_inhabited"
        | "assert_mem_uninitialized_valid"
        | "assert_zero_valid"
        | "assume"
        | "bitreverse"
        | "black_box"
        | "breakpoint"
        | "bswap"
        | "caller_location"
        | "ceilf32"
        | "ceilf64"
        | "copysignf32"
        | "copysignf64"
        | "cosf32"
        | "cosf64"
        | "ctlz"
        | "ctlz_nonzero"
        | "ctpop"
        | "cttz"
        | "cttz_nonzero"
        | "discriminant_value"
        | "exact_div"
        | "exp2f32"
        | "exp2f64"
        | "expf32"
        | "expf64"
        | "fabsf32"
        | "fabsf64"
        | "fadd_fast"
        | "fdiv_fast"
        | "floorf32"
        | "floorf64"
        | "fmaf32"
        | "fmaf64"
        | "fmul_fast"
        | "forget"
        | "fsub_fast"
        | "is_val_statically_known"
        | "likely"
        | "log10f32"
        | "log10f64"
        | "log2f32"
        | "log2f64"
        | "logf32"
        | "logf64"
        | "maxnumf32"
        | "maxnumf64"
        | "min_align_of"
        | "min_align_of_val"
        | "minnumf32"
        | "minnumf64"
        | "mul_with_overflow"
        | "nearbyintf32"
        | "nearbyintf64"
        | "needs_drop"
        | "powf32"
        | "powf64"
        | "powif32"
        | "powif64"
        | "pref_align_of"
        | "raw_eq"
        | "rintf32"
        | "rintf64"
        | "rotate_left"
        | "rotate_right"
        | "roundf32"
        | "roundf64"
        | "saturating_add"
        | "saturating_sub"
        | "sinf32"
        | "sinf64"
        | "sqrtf32"
        | "sqrtf64"
        | "sub_with_overflow"
        | "truncf32"
        | "truncf64"
        | "type_id"
        | "type_name"
        | "unchecked_div"
        | "unchecked_rem"
        | "unlikely"
        | "vtable_size"
        | "vtable_align"
        | "wrapping_add"
        | "wrapping_mul"
        | "wrapping_sub" => {
            /* Intrinsics that do not interact with memory initialization. */
            true
        }
        "ptr_guaranteed_cmp" | "ptr_offset_from" | "ptr_offset_from_unsigned" | "size_of_val" => {
            /* AFAICS from the documentation, none of those require the pointer arguments to be actually initialized. */
            true
        }
        name if name.starts_with("simd") => {
            /* SIMD operations */
            true
        }
        name if name.starts_with("atomic_fence")
            || name.starts_with("atomic_singlethreadfence") =>
        {
            /* Atomic fences */
            true
        }
        "copy_nonoverlapping" => unreachable!(
            "Expected `core::intrinsics::unreachable` to be handled by `StatementKind::CopyNonOverlapping`"
        ),
        "offset" => unreachable!(
            "Expected `core::intrinsics::unreachable` to be handled by `BinOp::OffSet`"
        ),
        "unreachable" => unreachable!(
            "Expected `std::intrinsics::unreachable` to be handled by `TerminatorKind::Unreachable`"
        ),
        "transmute" | "transmute_copy" | "unchecked_add" | "unchecked_mul" | "unchecked_shl"
        | "size_of" | "unchecked_shr" | "unchecked_sub" => {
            unreachable!("Expected intrinsic to be lowered before codegen")
        }
        "catch_unwind" => {
            unimplemented!("")
        }
        "retag_box_to_raw" => {
            unreachable!("This was removed in the latest Rust version.")
        }
        _ => {
            /* Everything else */
            false
        }
    }
}

/// Create a constant operand with a given value and span.
fn mk_const_operand(value: usize, span: Span) -> Operand {
    Operand::Constant(ConstOperand {
        span,
        user_ty: None,
        const_: MirConst::try_from_uint(value as u128, UintTy::Usize).unwrap(),
    })
}

/// Try removing a topmost deref projection from a place if it exists, returning a place without it.
fn try_remove_topmost_deref(place: &Place) -> Option<Place> {
    let mut new_place = place.clone();
    if let Some(ProjectionElem::Deref) = new_place.projection.pop() {
        Some(new_place)
    } else {
        None
    }
}

/// Try retrieving instance for the given function operand.
fn try_resolve_instance(locals: &[LocalDecl], func: &Operand) -> Result<Instance, String> {
    let ty = func.ty(locals).unwrap();
    match ty.kind() {
        TyKind::RigidTy(RigidTy::FnDef(def, args)) => Ok(Instance::resolve(def, &args).unwrap()),
        _ => Err(format!(
            "Kani does not support reasoning about memory initialization of arguments to `{ty:?}`."
        )),
    }
}
