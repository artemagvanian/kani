// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0 OR MIT
// rmc-flags: --use-abs --abs-type rmc
include!{"../../rmc-prelude.rs"}

fn main() {
    fn reserve_test() {
        let mut vec = rmc_vec![1];
        vec.reserve(10);
        assert!(vec.capacity() >= 11);
    }

    reserve_test();
}