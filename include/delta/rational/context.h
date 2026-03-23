// include/delta/rational/context.h
#pragma once

#include "delta/rational/rational_fwd.h"   // только fwd, без полного определения

namespace delta::internal {
    constexpr int MAX_LAZY_DEPTH = 1000;
    inline thread_local bool global_eager_mode = false;
}

namespace delta {
    inline bool eager_mode() {
        return internal::global_eager_mode;
    }
    inline void set_eager_mode(bool flag) {
        internal::global_eager_mode = flag;
    }

    class ScopedEagerEval {
        bool old_;
    public:
        ScopedEagerEval() : old_(internal::global_eager_mode) {
            internal::global_eager_mode = true;
        }
        ~ScopedEagerEval() {
            internal::global_eager_mode = old_;
        }
    };
}