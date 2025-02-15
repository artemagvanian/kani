searchState.loadedDescShard("kani", 0, "Enumeration with the cases currently covered by the …\nAllow users to auto generate <code>Arbitrary</code> implementations by …\nHolds information about a pointer that is generated …\nDangling pointers\nPointer to dead object\nIn bounds pointer (it may be unaligned)\nAllow users to auto generate <code>Invariant</code> implementations by …\nNull pointers\nThe pointer cannot be read / written to for the given type …\nPointer generator that can be used to generate arbitrary …\nThis creates an symbolic <em>valid</em> value of type <code>T</code>. You can …\nCreates a raw pointer with non-deterministic properties.\nCreates a in-bounds raw pointer with non-deterministic …\nThis creates a symbolic <em>valid</em> value of type <code>T</code>. The value …\nThis module introduces the <code>Arbitrary</code> trait as well as …\nThis macro implements <code>kani::Arbitrary</code> on a tuple whose …\nCreates an assertion of the specified condition and …\nCreates an assumption that will be valid after this …\nNOP <code>concrete_playback</code> for type checking during …\nKani implementation of function contracts.\nCreates a cover property with the specified condition and …\nAdd a postcondition to this function.\nThis module contains functions useful for float-related …\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nThis module contains functions to work with futures (and …\n<code>implies!(premise =&gt; conclusion)</code> means that if the <code>premise</code> …\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nThis module introduces the <code>Invariant</code> trait as well as its …\nWhether the pointer was generated with an initialized …\nKani intrinsics contains the public APIs used by users to …\nUsers should only need to invoke this.\nAdd a loop invariant to this loop.\nThis module contains functions useful for checking unsafe …\nDeclaration of an explicit write-set for the annotated …\nCreate a new PointerGenerator.\nCreate a pointer generator that fits at least <code>N</code> elements …\nMarks a Kani proof harness\nDesignates this function as a harness to check a function …\nThe pointer that was generated.\nSpecifies that a function contains recursion for contract …\nAdd a precondition to this function.\nThis module contains an API for shadow memory. Shadow …\nSpecifies that a proof harness is expected to panic.**\nSelect the SAT solver to use with CBMC for this harness\nThe expected allocation status.\nSpecify a function/method stub pair to use for proof …\n<code>stub_verified(TARGET)</code> is a harness attribute (to be used on\nSet Loop unwind limit for proof harnesses The attribute …\nAdd a postcondition to this function.\nDeclaration of an explicit write-set for the annotated …\nDesignates this function as a harness to check a function …\nAdd a precondition to this function.\n<code>stub_verified(TARGET)</code> is a harness attribute (to be used on\nReturns whether the given float <code>value</code> satisfies the range …\nResult of spawning a task.\nKeeps cycling through the tasks in a deterministic order\nIndicates to the scheduler whether it can <code>kani::assume</code> …\nTrait that determines the possible sequence of tasks …\nA very simple executor: it polls the future in a busy loop …\nPolls the given future and the tasks it may spawn until …\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nPicks the next task to be scheduled whenever the scheduler …\nSpawns a task on the current global executor (which is set …\nSuspends execution of the current future, to allow the …\nThis trait should be used to specify and check type safety …\nChecks that pointer <code>ptr</code> point to a valid value of type <code>T</code>.\nChecks that pointer <code>ptr</code> point to a valid value of type <code>T</code>.\nCheck if the pointer is valid for write access according …\nCheck if the pointer is valid for unaligned write access …\nCompute the size of the val pointed to if safe.\nCompute the size of the val pointed to if it is safe to do …\nCheck if two pointers points to the same allocated object, …\nA shadow memory data structure that contains a …\nReturns the argument unchanged.\nGet the shadow memory value of the given pointer\nCalls <code>U::from(self)</code>.\nCreate a new shadow memory instance initialized with the …\nSet the shadow memory value of the given pointer\nGiven an array <code>arr</code> of length <code>LENGTH</code>, this function returns …\nA mutable version of the previous function\nGenerates an arbitrary vector whose length is at most …\nGenerates an arbitrary vector that is exactly EXACT_LENGTH …")