// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// utils.h
#pragma once

#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/number.hpp>
#include <string>
#include <algorithm>

namespace delta::internal {

    // ============================================================================
    // Тип dumb_int с отключёнными expression templates (et_off)
    // Используется для числителя/знаменателя Value, а также во внешних интерфейсах.
    // ============================================================================
    using dumb_int = boost::multiprecision::number<
        boost::multiprecision::cpp_int_backend<>,
        boost::multiprecision::et_off
    >;

    // ----------------------------------------------------------------------------
    // Вспомогательные функции для работы со строками (только для отладки)
    // Могут быть удалены, если не используются.
    // ----------------------------------------------------------------------------
    inline std::string dumb_int_to_string(const dumb_int& n) {
        return n.str();
    }

} // namespace delta::internal