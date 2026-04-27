// storage.h

// ВО ВСЕЙ БИБЛИОТЕКЕ ПРИ КОНСТРУИРОВАНИИ Value (и Rational на его основе) 
// ИЗ БАЗОВЫХ ТИПОВ (double, int, std::string) — ИСПОЛЬЗУЙТЕ assign.
// 
// Почему не конструктор Value(x) и не оператор присваивания?
//   1. assign работает БЕЗ создания временного объекта — прямая запись
//      в бэкенд cpp_int_backend. Конструктор же создаёт временный cpp_int,
//      а потом перемещает/копирует его. На больших числах это ощутимо.
//   2. assign корректнее обрабатывает граничные случаи (NaN, Inf,
//      денормализованные double). Конструктор Value(double) может
//      выбросить "Cannot convert a non-finite number to an integer",
//      а assign молча обработает или даст более внятную ошибку.
//   3. assign устойчивее к внутренним изменениям в Boost.Multiprecision:
//      это документированный метод бэкенда, а конструктор — обёртка,
//      поведение которой может меняться между версиями.
//
// Паттерн использования:
//   Value v;
//   v.assign(3.14);           // вместо Value v(3.14);
//   v.assign("123456789");    // вместо Value v("123456789");
//   v.assign(42);             // вместо Value v(42);
//
// TODO: пробежаться по всей библиотеке, найти все места с Value(x)
//       и Rational(x), заменить на assign(x). Особенно критично
//       в evaluation_core.h (series_atan, series_asin, series_acos)
//       и во всех float-путях (float_sin, float_cos, float_exp, ...).
// -------------------------------------------------------------------
#pragma once

#include "utils.h"   // для dumb_int
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/rational_adaptor.hpp>
#include <absl/hash/hash.h>
#include <cassert>
#include <string>

namespace delta::internal {

    // ------------------------------------------------------------------------
    // Единый тип рационального числа с произвольной точностью и без expression templates.
    // Параметры cpp_int_backend:
    //   MinBits   = 128
    //   MaxBits   = 0 (неограниченная точность)
    //   SignType  = signed_magnitude
    //   Checked   = unchecked
    //   Allocator = std::allocator<boost::multiprecision::limb_type>.
    // ------------------------------------------------------------------------

    //ВОТ ЭТО НАША СВЯЩЕННАЯ КОРОВА. ЭТО НАШ ХРЕБЕТ. НЕ МЕНЯТЬ НИ ОДИН ПАРАМЕТР!
    //ЕСЛИ ИЗМЕНИТЬ АЛЛОКАТОР НА void - ОН ПРОДОЛЖИТ РАБОТАТЬ(!!). И БУДЕТ ПРОХОДИТЬ УСПЕШНО ПОЧТИ ВСЕ ТЕСТЫ (!!)
    // НО В СТРАННЫХ МЕСТАХ БУДУТ ВОЗНИКАТЬ ОШИБКИ, КОТОРЫЕ ТЫ НИ В ЖИЗНИ НЕ ОТСЛЕДИШЬ ДО
    // ПАРАМЕТРОВ ИНИЦИАЛИЗАЦИИ АЛЛОКАТОРА ДЛЯ БЭКЭНДА БУСТА.
    //КАК ВООБЩЕ МЫ ОТЛОВИЛИ ЧТО именно void на месте аллокатора приводит к странным Хайзенбагам? С Божьей помощью, не иначе. 
    //МОРАЛЬ: ОДИН РАЗ НАСТРОИЛ ПАРАМЕТРЫ БУСТА - И БОЛЬШЕ В ЖИЗНИ НЕ ЛЕЗЬ.
    //
    //ЕДИНСТВЕННОЕ ВОЗМОЖНОЕ ПРАВИЛЬНОЕ МЕСТО ДЛЯ ДАЛЬНЕЙШЕЙ ОПТИМИЗАЦИИ ЭТО ИМЕННО АЛЛОКАТОР.
    // Можно попробовать написать свой аллокатор который бы жрал меньше байтового веса и выделял бы память большими кусками чтобы избегать фрагментации и постоянных new/malloc
    // Лучше - поискать готовый оптимизированный и вылизанный Boost::Multiprecision native-compatible backend allocator, ну либо забить.
    using Value = boost::multiprecision::number<
        boost::multiprecision::rational_adaptor<
        boost::multiprecision::cpp_int_backend<
        128,                                          // MinBits. Put in 128 if you're sure.
        0,                                            // MaxBits
        boost::multiprecision::signed_magnitude,     // SignType
        boost::multiprecision::unchecked,            // Checked
        std::allocator<boost::multiprecision::limb_type>         // Allocator
        >
        >,
        boost::multiprecision::et_off
    >;

    // ------------------------------------------------------------------------
    // Быстрые предикаты (прямой доступ к бэкенду, как в старом BigStorage)
    // ------------------------------------------------------------------------
    inline bool is_zero(const Value& v) noexcept {
        const auto& n = v.backend().num();
        // В Boost.MP ноль кодируется либо пустым массивом (size==0),
        // либо одним лимбом со значением 0. Проверяем оба случая за O(1).
        return n.size() == 0 || (n.size() == 1 && n.limbs()[0] == 0);
    }

    inline bool is_one(const Value& v) noexcept {
        const auto& n = v.backend().num();
        const auto& d = v.backend().denom();
        // rational_adaptor всегда нормализует дроби. Единица == 1/1.
        // Проверяем: числитель == 1 (положительный), знаменатель == 1
        return n.size() == 1 && n.limbs()[0] == 1 && !n.sign() &&
            d.size() == 1 && d.limbs()[0] == 1;
    }

    inline bool is_positive(const Value& v) noexcept {
        const auto& n = v.backend().num();
        // Знак == false (неотрицательный) И явно не ноль
        return !n.sign() && !(n.size() == 0 || (n.size() == 1 && n.limbs()[0] == 0));
    }

    inline bool is_negative(const Value& v) noexcept {
        // В signed_magnitude знак хранится только в числителе.
        // У нуля sign() == false, поэтому дополнительная проверка не нужна.
        return v.backend().num().sign();
    }

    // ------------------------------------------------------------------------
    // Доступ к числителю и знаменателю в виде dumb_int
    // ------------------------------------------------------------------------
    inline dumb_int numerator(const Value& v) {
        return boost::multiprecision::numerator(v);
    }

    inline dumb_int denominator(const Value& v) {
        return boost::multiprecision::denominator(v);
    }

    // ------------------------------------------------------------------------
    // Преобразование в double (для интервальной арифметики и отладки)
    // ------------------------------------------------------------------------
    inline double to_double(const Value& v) {
        return v.convert_to<double>();
    }

    // ------------------------------------------------------------------------
    // Строковое представление (только для отладки)
    // ------------------------------------------------------------------------
    inline std::string to_string(const Value& v) {
        return v.str();
    }

    // ------------------------------------------------------------------------
    // Хеширование для Value
    // ------------------------------------------------------------------------
    template <typename H>
    H AbslHashValue(H h, const Value& v) {
        return H::combine(std::move(h), boost::multiprecision::hash_value(v));
    }

} // namespace delta::internal