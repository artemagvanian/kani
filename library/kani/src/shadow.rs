// Copyright Kani Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! This module contains an API for shadow memory.
//! Shadow memory is a mechanism by which we can store metadata on memory
//! locations, e.g. whether a memory location is initialized.
//!
//! The main data structure provided by this module is the `ShadowMem` struct,
//! which allows us to store metadata on a given memory location.
//!
//! # Example
//!
//! ```
//! use kani::shadow::ShadowMem;
//! use std::alloc::{alloc, Layout};
//!
//! let mut sm = ShadowMem::new(false);
//!
//! unsafe {
//!     let ptr = alloc(Layout::new::<u8>());
//!     // assert the memory location is not initialized
//!     assert!(!sm.get(ptr));
//!     // write to the memory location
//!     *ptr = 42;
//!     // update the shadow memory to indicate that this location is now initialized
//!     sm.set(ptr, true);
//! }
//! ```

const MAX_NUM_OBJECTS: usize = 1024;
const MAX_OBJECT_SIZE: usize = 64;

const MAX_NUM_OBJECTS_ASSERT_MSG: &str = "The number of objects exceeds the maximum number supported by Kani's shadow memory model (1024)";
const MAX_OBJECT_SIZE_ASSERT_MSG: &str =
    "The object size exceeds the maximum size supported by Kani's shadow memory model (64)";

/// A shadow memory data structure that contains a two-dimensional array of a
/// generic type `T`.
/// Each element of the outer array represents an object, and each element of
/// the inner array represents a byte in the object.
pub struct ShadowMem<T: Copy> {
    mem: [[T; MAX_OBJECT_SIZE]; MAX_NUM_OBJECTS],
}

impl<T: Copy> ShadowMem<T> {
    /// Create a new shadow memory instance initialized with the given value
    #[crate::unstable(
        feature = "ghost-state",
        issue = 3184,
        reason = "experimental ghost state/shadow memory API"
    )]
    pub const fn new(val: T) -> Self {
        Self { mem: [[val; MAX_OBJECT_SIZE]; MAX_NUM_OBJECTS] }
    }

    /// Get the shadow memory value of the given pointer
    #[crate::unstable(
        feature = "ghost-state",
        issue = 3184,
        reason = "experimental ghost state/shadow memory API"
    )]
    pub fn get<U>(&self, ptr: *const U) -> T {
        let obj = crate::mem::pointer_object(ptr);
        let offset = crate::mem::pointer_offset(ptr);
        crate::assert(obj < MAX_NUM_OBJECTS, MAX_NUM_OBJECTS_ASSERT_MSG);
        crate::assert(offset < MAX_OBJECT_SIZE, MAX_OBJECT_SIZE_ASSERT_MSG);
        self.mem[obj][offset]
    }

    /// Set the shadow memory value of the given pointer
    #[crate::unstable(
        feature = "ghost-state",
        issue = 3184,
        reason = "experimental ghost state/shadow memory API"
    )]
    pub fn set<U>(&mut self, ptr: *const U, val: T) {
        let obj = crate::mem::pointer_object(ptr);
        let offset = crate::mem::pointer_offset(ptr);
        crate::assert(obj < MAX_NUM_OBJECTS, MAX_NUM_OBJECTS_ASSERT_MSG);
        crate::assert(offset < MAX_OBJECT_SIZE, MAX_OBJECT_SIZE_ASSERT_MSG);
        self.mem[obj][offset] = val;
    }
}

#[allow(dead_code)]
mod meminit {
    use crate::any;

    pub struct MemInit {
        pub curr: usize,
        pub layout: u128,
    }

    impl MemInit {
        pub const fn new() -> Self {
            Self { curr: 0, layout: 0 }
        }

        pub fn get<const SIZE: usize>(&mut self, ptr: *const u8, layout: u128) -> bool {
            if SIZE == 0 {
                return true;
            }

            let obj = crate::mem::pointer_object(ptr);
            let offset = crate::mem::pointer_offset(ptr);
            crate::assert(
                offset + SIZE < 128,
                "Layout tag is a u128, so cannot represent larger layouts",
            );
            crate::assert(
                obj == crate::mem::pointer_object(unsafe { ptr.add(SIZE) }),
                "cannot set shadow memory for multiple objects at once",
            );

            if self.curr == obj {
                let bit_mask = ((1u128 << SIZE) - 1) << offset;
                ((self.layout | !bit_mask) | !(layout << offset)) == u128::MAX
            } else {
                true
            }
        }

        pub fn set<const SIZE: usize>(&mut self, ptr: *const u8, layout: u128, val: bool) {
            if SIZE == 0 {
                return;
            }

            let obj = crate::mem::pointer_object(ptr);
            let offset = crate::mem::pointer_offset(ptr);
            crate::assert(
                offset + SIZE < 128,
                "Layout tag is a u128, so cannot represent larger layouts",
            );
            crate::assert(
                obj == crate::mem::pointer_object(unsafe { ptr.add(SIZE) }),
                "cannot set shadow memory for multiple objects at once",
            );

            if self.curr == obj {
                let bit_mask = ((1u128 << SIZE) - 1) << offset;
                self.layout &= !bit_mask;
                if val {
                    self.layout |= layout << offset;
                }
            }
        }
    }

    /// Global shadow memory object for tracking memory initialization.
    #[rustc_diagnostic_item = "KaniMemInitSM"]
    static mut __KANI_MEM_INIT_SM: MemInit = MemInit::new();

    #[rustc_diagnostic_item = "KaniMemInitSMInit"]
    pub fn __kani_mem_init_sm_init() {
        unsafe {
            __KANI_MEM_INIT_SM.curr = any();
        }
    }

    /// Get initialization state of `len` items laid out according to the `layout` starting at address `ptr`.
    #[rustc_diagnostic_item = "KaniMemInitSMGetInner"]
    fn __kani_mem_init_sm_get_inner<const SIZE: usize>(
        ptr: *const (),
        layout: u128,
        len: usize,
    ) -> bool {
        let mut count: usize = 0;
        while count < len {
            if unsafe {
                !__KANI_MEM_INIT_SM.get::<SIZE>((ptr as *const u8).add(count * SIZE), layout)
            } {
                return false;
            }
            count += 1;
        }
        true
    }

    /// Set initialization state to `value` for `len` items laid out according to the `layout` starting at address `ptr`.
    #[rustc_diagnostic_item = "KaniMemInitSMSetInner"]
    fn __kani_mem_init_sm_set_inner<const SIZE: usize>(
        ptr: *const (),
        layout: u128,
        len: usize,
        value: bool,
    ) {
        let mut count: usize = 0;
        while count < len {
            unsafe {
                __KANI_MEM_INIT_SM.set::<SIZE>((ptr as *const u8).add(count * SIZE), layout, value);
            }
            count += 1;
        }
    }

    /// Get initialization state of `len` items laid out according to the `layout` starting at address `ptr`.
    #[rustc_diagnostic_item = "KaniMemInitSMGet"]
    fn __kani_mem_init_sm_get<const SIZE: usize, T>(
        ptr: *const T,
        layout: u128,
        len: usize,
    ) -> bool {
        let (ptr, _) = ptr.to_raw_parts();
        __kani_mem_init_sm_get_inner::<SIZE>(ptr, layout, len)
    }

    /// Set initialization state to `value` for `len` items laid out according to the `layout` starting at address `ptr`.
    #[rustc_diagnostic_item = "KaniMemInitSMSet"]
    fn __kani_mem_init_sm_set<const SIZE: usize, T>(
        ptr: *const T,
        layout: u128,
        len: usize,
        value: bool,
    ) {
        let (ptr, _) = ptr.to_raw_parts();
        __kani_mem_init_sm_set_inner::<SIZE>(ptr, layout, len, value);
    }

    /// Get initialization state of the slice, items of which are laid out according to the `layout` starting at address `ptr`.
    #[rustc_diagnostic_item = "KaniMemInitSMGetSlice"]
    fn __kani_mem_init_sm_get_slice<const SIZE: usize, T>(ptr: *const [T], layout: u128) -> bool {
        let (ptr, len) = ptr.to_raw_parts();
        __kani_mem_init_sm_get_inner::<SIZE>(ptr, layout, len)
    }

    /// Set initialization state of the slice, items of which are laid out according to the `layout` starting at address `ptr` to `value`.
    #[rustc_diagnostic_item = "KaniMemInitSMSetSlice"]
    fn __kani_mem_init_sm_set_slice<const SIZE: usize, T>(
        ptr: *const [T],
        layout: u128,
        value: bool,
    ) {
        let (ptr, len) = ptr.to_raw_parts();
        __kani_mem_init_sm_set_inner::<SIZE>(ptr, layout, len, value);
    }

    /// Get initialization state of the string slice, items of which are laid out according to the `layout` starting at address `ptr`.
    #[rustc_diagnostic_item = "KaniMemInitSMGetStr"]
    fn __kani_mem_init_sm_get_str<const SIZE: usize>(ptr: *const str, layout: u128) -> bool {
        let (ptr, len) = ptr.to_raw_parts();
        __kani_mem_init_sm_get_inner::<SIZE>(ptr, layout, len)
    }

    /// Set initialization state of the string slice, items of which are laid out according to the `layout` starting at address `ptr` to `value`.
    #[rustc_diagnostic_item = "KaniMemInitSMSetStr"]
    fn __kani_mem_init_sm_set_str<const SIZE: usize>(ptr: *const str, layout: u128, value: bool) {
        let (ptr, len) = ptr.to_raw_parts();
        __kani_mem_init_sm_set_inner::<SIZE>(ptr, layout, len, value);
    }
}
