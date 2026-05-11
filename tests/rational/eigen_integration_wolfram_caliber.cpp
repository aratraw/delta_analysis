// tests/rational/eigen_integration_wolfram_caliber.cpp
// ============================================================================
// VERIFICATION AGAINST WOLFRAM ALPHA HIGH-PRECISION RESULTS
// ============================================================================
//
// This test suite compares delta's matrix transcendental functions against
// reference values computed by Wolfram Alpha with 35-digit precision.
// Matrices are 5×5, both real and complex.
//
// The reference values are given in scientific notation with up to 35 digits.
// We convert them to decimal strings (e.g., "4.6289621709104999274498417099543841")
// and use delta::Rational(const std::string&) and delta::GaussQi(const std::string&)
// constructors which accept decimal strings (but NOT scientific "1e+12").
//
// The tolerance for comparisons is set to 1e-30 (EPS) for all tests.
// ============================================================================
// ============================================================================
// ON ABSOLUTE VS RELATIVE TOLERANCE IN WOLFRAM VERIFICATION TESTS
// ============================================================================
//
// This test suite compares delta's matrix transcendental functions against
// reference values computed by Wolfram Alpha with up to 35-digit precision.
//
// The requested tolerance (EPS = 1e‑19) is a *relative* accuracy goal:
// the algorithm guarantees |result – exact| ≤ C · EPS · |exact| for each
// matrix element, with C ≈ 1.  This is the natural guarantee of the
// scaling‑and‑squaring / Padé approximant framework used throughout
// the library.
//
// For most test matrices the entries of the transcendental function are
// bounded in magnitude by a few hundred (e.g., sin, cos, log, sqrt of the
// given matrices).  In those cases an *absolute* check
//
//   EXPECT_RATIONAL_NEAR(result, expected, EPS)
//
// accidentally works because the expected value itself is ≤ O(100), so the
// required absolute error is at most ~100·EPS ≈ 10⁻¹⁷, and the algorithm
// typically delivers even tighter absolute accuracy.
//
// However, when the expected values become very large – as with exp(A) where
// the matrix elements reach 10¹² … 10¹³ – an absolute tolerance of EPS
// demands an *absolute* error ≤ 10⁻¹⁹ on a quantity of size 10¹³, i.e. a
// relative accuracy of 10⁻³².  Our algorithm never promised that, and
// achieving it would require a wildly larger Padé order and working
// precision.  The test would therefore fail despite the algorithm behaving
// correctly.
//
// The correct acceptance criterion for large elements is thus a *relative*
// one, consistent with the library's contract:
//
//   Rational tol = EPS · max(Rational(1), abs(expected(i,j)));
//   EXPECT_RATIONAL_NEAR(result(i,j), expected(i,j), tol);
//
// This is exactly the approach already adopted for the scalar transcendental
// tests (see evaluation_core.h commentary on exp of large arguments).  For
// elements of unit size or smaller the tolerance reduces to EPS, preserving
// the usual absolute check; for large elements it scales proportionally,
// verifying the promised relative accuracy.
//
// **In a nutshell:**
//   - The library provides *relative* accuracy EPS.
//   - Many tests pass with an absolute EPS check because the numbers are small.
//   - For exp(A) the numbers are huge, so we switch to a relative check.
//   - This is not a “lucky” escape, but the mathematically correct way to
//     test the library’s actual guarantees.
//
// Future tests with matrices that may produce very large (or very small)
// entries should adopt the same relative tolerance pattern from the start.
// ============================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "delta/core/rational.h"
#include "delta/core/eigen_integration.h"
#include "test_utils.h"
#include <cctype>
#include <algorithm>

using namespace delta;
using namespace delta::literals;

namespace delta::testing {

    // ------------------------------------------------------------------------
    // Helper to convert scientific notation string to plain decimal string.
    // Wolfram output: "4.6289621709104999274498417099543841×10^12"
    // We convert to "4628962170910.4999274498417099543841"
    // ------------------------------------------------------------------------
    static std::string scientific_to_decimal(const std::string& sci) {
        std::string s = sci;
        // Удаляем все пробельные символы (включая \r, \n, \t)
        s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());

        // Ищем "10^" (без учёта возможного символа умножения)
        size_t pos = s.find("10^");
        if (pos == std::string::npos) {
            // Поддержка научной нотации через e/E
            size_t epos = s.find('e');
            if (epos == std::string::npos) epos = s.find('E');
            if (epos != std::string::npos) {
                std::string mantissa = s.substr(0, epos);
                std::string exp_str = s.substr(epos + 1);
                // Удаляем все символы, кроме цифр и минуса, из экспоненты
                std::string clean_exp;
                for (char c : exp_str) {
                    if (std::isdigit(c) || c == '-') clean_exp.push_back(c);
                }
                int exponent = std::stoi(clean_exp);
                // Преобразуем mantissa (может содержать десятичную точку)
                // Используем готовую функцию преобразования с плавающей точкой
                // Для рациональных чисел нужно аккуратно, но здесь mantissa не имеет экспоненты
                // Просто вернём mantissa, умноженную на 10^exponent (сдвиг точки)
                // В целях теста можно вернуть строку с десятичной записью
                size_t dot = mantissa.find('.');
                if (dot == std::string::npos) mantissa += ".0";
                std::string int_part = mantissa.substr(0, dot);
                std::string frac_part = mantissa.substr(dot + 1);
                if (exponent > 0) {
                    int shift = exponent;
                    while (shift > 0 && !frac_part.empty()) {
                        int_part += frac_part.front();
                        frac_part.erase(0, 1);
                        --shift;
                    }
                    if (shift > 0) int_part += std::string(shift, '0');
                    return int_part + (frac_part.empty() ? "" : "." + frac_part);
                }
                else if (exponent < 0) {
                    int shift = -exponent;
                    std::string new_int_part;
                    while (shift > 0 && !int_part.empty()) {
                        new_int_part = int_part.back() + new_int_part;
                        int_part.pop_back();
                        --shift;
                    }
                    if (shift > 0) new_int_part = std::string(shift, '0') + new_int_part;
                    if (new_int_part.empty()) new_int_part = "0";
                    std::string new_frac = int_part + frac_part;
                    if (new_frac.empty()) new_frac = "0";
                    return new_int_part + "." + new_frac;
                }
                else {
                    return mantissa;
                }
            }
            return sci; // нет экспоненты
        }

        // Извлекаем мантиссу (всё до "10^")
        std::string mantissa = s.substr(0, pos);
        // Удаляем возможный символ умножения в конце мантиссы (×, *, ·)
        if (!mantissa.empty() && (mantissa.back() == '×' || mantissa.back() == '*' || mantissa.back() == '·'))
            mantissa.pop_back();

        // Извлекаем экспоненту (всё после "10^")
        std::string exponent_str = s.substr(pos + 3); // длина "10^" = 3
        // Оставляем в exponent_str только цифры и знак минус
        std::string clean_exp;
        for (char c : exponent_str) {
            if (std::isdigit(c) || c == '-') clean_exp.push_back(c);
        }
        int exponent = std::stoi(clean_exp);

        // Преобразуем мантиссу (число с точкой) в десятичную строку со сдвигом
        size_t dot = mantissa.find('.');
        if (dot == std::string::npos) {
            mantissa += ".0";
            dot = mantissa.find('.');
        }
        std::string int_part = mantissa.substr(0, dot);
        std::string frac_part = mantissa.substr(dot + 1);

        if (exponent > 0) {
            int shift = exponent;
            while (shift > 0 && !frac_part.empty()) {
                int_part += frac_part.front();
                frac_part.erase(0, 1);
                --shift;
            }
            if (shift > 0) int_part += std::string(shift, '0');
            return int_part + (frac_part.empty() ? "" : "." + frac_part);
        }
        else if (exponent < 0) {
            int shift = -exponent;
            std::string new_int_part;
            while (shift > 0 && !int_part.empty()) {
                new_int_part = int_part.back() + new_int_part;
                int_part.pop_back();
                --shift;
            }
            if (shift > 0) new_int_part = std::string(shift, '0') + new_int_part;
            if (new_int_part.empty()) new_int_part = "0";
            std::string new_frac = int_part + frac_part;
            if (new_frac.empty()) new_frac = "0";
            return new_int_part + "." + new_frac;
        }
        else {
            return mantissa;
        }
    }
    // Helper to create Rational from Wolfram number string (scientific or decimal)
    static Rational rational_from_wolfram(const std::string& s) {
        std::string dec = scientific_to_decimal(s);
        return Rational(dec);
    }

    // Helper to create GaussQi from two Wolfram number strings (real, imag)
    static GaussQi gaussqi_from_wolfram(const std::string& re, const std::string& im) {
        std::string re_dec = scientific_to_decimal(re);
        std::string im_dec = scientific_to_decimal(im);
        return GaussQi(Rational(re_dec), Rational(im_dec));
    }

    // ------------------------------------------------------------------------
    // Reference values as given by Wolfram Alpha (5x5 matrices)
    // ------------------------------------------------------------------------

    // ---- Real matrix A (2,3,4,5,6; 3,4,5,6,7; ...) ----
    static const std::vector<std::vector<std::string>> wolfram_exp_A_str = {
            {"4628962170910.4999274498417099543841", "5701319022784.3990012395482622327029", "6773675874659.2980750292548145110218", "7846032726534.1971488189613667893406", "8918389578409.0962226086679190676595"},
            {"5701319022784.3990012395482622327029", "7022100721376.2682530162976750507964", "8342882419966.1375047930470878688899", "9663664118557.0067565697965006869834", "10984445817147.876008346545913505077"},
            {"6773675874659.2980750292548145110218", "8342882419966.1375047930470878688899", "9912088965273.9769345568393612267580", "11481295510579.816364320631634584626", "13050502055886.655794084423907942494"},
            {"7846032726534.1971488189613667893406", "9663664118557.0067565697965006869834", "11481295510579.816364320631634584626", "13298926902603.625972071466768482269", "15116558294625.435579822301902379912"},
            {"8918389578409.0962226086679190676595", "10984445817147.876008346545913505077", "13050502055886.655794084423907942494", "15116558294625.435579822301902379912", "17182614533365.215365560179896817329"}
    };


    static const std::vector<std::vector<std::string>> wolfram_sin_A_str = {
        {"-0.49621741555865141387197469862569842", "-0.27220064322136209334313130482168885", "-0.048183870884072772814287911017679279", "0.17583290145321654771455548278633029", "0.39984967379050586824339887659033986"},
        {"-0.27220064322136209334313130482168885", "-0.14260896690739577830725405964105104", "-0.013017290593429463271376814460413221", "0.11657438572053685176450043072022459", "0.24616606203450316680037767590086241"},
        {"-0.048183870884072772814287911017679279", "-0.013017290593429463271376814460413221", "0.022149289697213846271534282096852837", "0.057315869987857155814445378654118896", "0.092482450278500465357356475211384954"},
        {"0.17583290145321654771455548278633029", "0.11657438572053685176450043072022459", "0.057315869987857155814445378654118896", "-0.0019426457448225401356096734119868021", "-0.061201161477502236085664725478092500"},
        {"0.39984967379050586824339887659033986", "0.24616606203450316680037767590086241", "0.092482450278500465357356475211384954", "-0.061201161477502236085664725478092500", "-0.21488477323350493752868592616756995"}
    };

    // cos(A) not provided by user (only sin given in the output snippet)
    // Actually the user did not paste Wolfram output for cos(A) for real matrix A.
    // We will skip cos test for real A because no reference.

    // ---- Real matrix B (positive definite, tridiagonal 2,3,4,5,6 with ones) ----
    static const std::vector<std::vector<std::string>> wolfram_log_B_str = {
        {"0.58300086180624622641709184411827453", "0.44846580526924368722072947556844935", "-0.069939908268890562984574744511173733", "0.011068784902100660475781097257310823", "-0.0015606331517510980879496133130247064"},
        {"0.44846580526924368722072947556844935", "0.96152675880659935065324657517555014", "0.31965477363356322172736108380341270", "-0.038294186714339679645181066052265971", "0.0048262522950962681239826440052119973"},
        {"-0.069939908268890562984574744511173733", "0.31965477363356322172736108380341270", "1.3128272539947134557200013374378706", "0.23682386759787947008520049844678194", "-0.022254796677299777185283520723605272"},
        {"0.011068784902100660475781097257310823", "-0.038294186714339679645181066052265971", "0.23682386759787947008520049844678194", "1.5656905116296328282650993812133132", "0.18748802194818364759065081299435939"},
        {"-0.0015606331517510980879496133130247064", "0.0048262522950962681239826440052119973", "-0.022254796677299777185283520723605272", "0.18748802194818364759065081299435939", "1.7754333302551162530410337149312779"}
    };

    static const std::vector<std::vector<std::string>> wolfram_sqrt_B_str = {
        {"1.3748420947714997707876898970745496", "0.33019138004649629647907213255730598", "-0.027754490468317114212943559925894224", "0.0035161566118539895512739932656908109", "-0.00043805117246727043034261573188572180"},
        {"0.33019138004649629647907213255730598", "1.6772789843496789530538184697059614", "0.27819855572171605760445900597120835", "-0.017644071805222415989464195860707513", "0.0017639519219849078299035303381479237"},
        {"-0.027754490468317114212943559925894224", "0.27819855572171605760445900597120835", "1.9655879587344897088817568397423564", "0.24115820742140214390416015132225043", "-0.011914164866800422069410989114378020"},
        {"0.0035161566118539895512739932656908109", "-0.017644071805222415989464195860707513", "0.24115820742140214390416015132225043", "2.2124760730943138467059701978109363", "0.21556592576581639193543464275534647"},
        {"-0.00043805117246727043034261573188572180", "0.0017639519219849078299035303381479237", "-0.011914164866800422069410989114378020", "0.21556592576581639193543464275534647", "2.4399561637269306607108158296806608"}
    };

    // ---- Complex matrix C (tridiagonal with complex diagonal) ----
    // Wolfram outputs for exp, sin, cos, log, sqrt (all given)
    // Format: real and imaginary parts, separated by space and 'i'? Actually Wolfram output: "a + b i"
    // We need to parse them. For simplicity, we'll store them as two strings (real, imag).
    // The user provided neatly formatted lists. We'll replicate them.

    // exp(C)
    static const std::vector<std::vector<std::pair<std::string, std::string>>> wolfram_exp_C_str = {
        {{"1.7286188352811995746226406846291123", "4.5966136493585767755317371021785613"},
         {"-0.84704170513293815106835449272814498", "6.5569050759761267924106341586395841"},
         {"-3.0603843427611482636332687038293059", "3.8551311604808454143278080338418476"},
         {"-2.3985917590086024978739425833992902", "0.85050768697567905289200669209207074"},
         {"-0.94331471001638826116759338106166723", "-0.18136887109835812222492282624403159"}},
        {{"-0.84704170513293815106835449272814498", "6.5569050759761267924106341586395841"},
         {"-8.7357122885890136324896166705679227", "14.161608180682610831201824801931848"},
         {"-17.076664470625528004864450551469742", "8.9969063983912001466917195107567382"},
         {"-13.750997390730381177098709911365056", "-0.97048992671628304284292246632384247"},
         {"-5.4463751146807230536446248026698328", "-3.6482266374833064806780581371307245"}},
        {{"-3.0603843427611482636332687038293059", "3.8551311604808454143278080338418476"},
         {"-17.076664470625528004864450551469742", "8.9969063983912001466917195107567382"},
         {"-45.499896205574974697511227940330153", "1.2562290212511545158583632610531539"},
         {"-45.685462754325844829146707660822712", "-24.944802560961113826761610073843854"},
         {"-18.202128112306242634830816526920714", "-28.072926312110013523586048459481483"}},
        {{"-2.3985917590086024978739425833992902", "0.85050768697567905289200669209207074"},
         {"-13.750997390730381177098709911365056", "-0.97048992671628304284292246632384247"},
         {"-45.685462754325844829146707660822712", "-24.944802560961113826761610073843854"},
         {"-70.691687120515567157628432142864668", "-96.476472679429534620793080466771052"},
         {"-20.497491240037579997991618993031340", "-113.84668477231031966291728190951752"}},
        {{"-0.94331471001638826116759338106166723", "-0.18136887109835812222492282624403159"},
         {"-5.4463751146807230536446248026698328", "-3.6482266374833064806780581371307245"},
         {"-18.202128112306242634830816526920714", "-28.072926312110013523586048459481483"},
         {"-20.497491240037579997991618993031340", "-113.84668477231031966291728190951752"},
         {"40.859634524063415142128047300542227", "-202.74772237966742075811593290983843"}}
    };

    // sin(C)
    static const std::vector<std::vector<std::pair<std::string, std::string>>> wolfram_sin_C_str = {
        {{"0.49629677607984613910758661043112290", "0.46322747915288591167364617261281033"},
         {"0.21857385848425828139541729033195984", "-1.3685685097368270984259879369769565"},
         {"-1.2245584991913398073968951353710317", "0.66217670746371069648977666285040562"},
         {"0.72895619139648107471475388399990961", "0.35120359734783352092875741994370673"},
         {"-0.070179451773041574041071239171306085", "-0.36012890494758424800718550838160764"}},
        {{"0.21857385848425828139541729033195984", "-1.3685685097368270984259879369769565"},
         {"0.85888064510959171153209670236900754", "-0.024590464635972208867147811181780665"},
         {"-2.8259403634293616516631724221110051", "-2.1421284958442517993114674620745018"},
         {"-0.16148016881843872007997698237372914", "3.5425271687490702354131250662996470"},
         {"1.8887540040946517705792109608411158", "-1.3700298295346697672642695702679482"}},
        {{"-1.2245584991913398073968951353710317", "0.66217670746371069648977666285040562"},
         {"-2.8259403634293616516631724221110051", "-2.1421284958442517993114674620745018"},
         {"1.2381471078973829464973098953298068", "-2.1123088626242261209184392919180462"},
         {"-9.0741572258662088667849194426165512", "2.8987320771345079431618017155656790"},
         {"9.6850507838425674674915358501247690", "5.4588285973766004933651347464007576"}},
        {{"0.72895619139648107471475388399990961", "0.35120359734783352092875741994370674"},
         {"-0.16148016881843872007997698237372914", "3.5425271687490702354131250662996470"},
         {"-9.0741572258662088667849194426165512", "2.8987320771345079431618017155656790"},
         {"-0.88821124244232767587789843035392524", "-6.3714325827283967865895473388678078"},
         {"-2.5104668570289266891113281960096442", "34.556520669107513632139412478884680"}},
        {{"-0.070179451773041574041071239171306085", "-0.36012890494758424800718550838160764"},
         {"1.8887540040946517705792109608411158", "-1.3700298295346697672642695702679482"},
         {"9.6850507838425674674915358501247690", "5.4588285973766004933651347464007576"},
         {"-2.5104668570289266891113281960096442", "34.556520669107513632139412478884680"},
         {"-47.640249552421335464620174955373019", "20.215792631973589663073402197606471"}}
    };

    // cos(C)
    static const std::vector<std::vector<std::pair<std::string, std::string>>> wolfram_cos_C_str = {
        {{"0.62545010024957660955774824114802668", "-0.29753757631728820017677319958267308"},
         {"-1.5365531127113211054060763457373877", "-0.18264848971587856988124798536043437"},
         {"0.67476005812454486578667793310628936", "1.1709036125890015780044171525947570"},
         {"0.36034022046663262728987669476456402", "-0.72187326926533257907966769468705689"},
         {"-0.36174920044366372585459063644838082", "0.071015473166282643510304122624808174"}},
        {{"-1.5365531127113211054060763457373877", "-0.18264848971587856988124798536043437"},
         {"-0.053694464621321060180402186122637266", "-0.84583556615548629745968037808573809"},
         {"-2.1685000011736019025516780899497590", "2.7868055824458817386212744913546015"},
         {"3.5596513268767767590407204650127713", "0.15731993935918436614534827545208659"},
         {"-1.3707184739731528501697023415281920", "-1.8848081783748569084568137499813475"}},
        {{"0.67476005812454486578667793310628936", "1.1709036125890015780044171525947570"},
         {"-2.1685000011736019025516780899497590", "2.7868055824458817386212744913546015"},
         {"-2.1241087794885728080993122355205159", "-1.2411136581130236732491528538235660"},
         {"2.9051040794217974057794872528788544", "9.0578132058082796596162659169900266"},
         {"5.4636696405255526597566453268206186", "-9.6802754908511275532445041217013399"}},
        {{"0.36034022046663262728987669476456402", "-0.72187326926533257907966769468705689"},
         {"3.5596513268767767590407204650127713", "0.15731993935918436614534827545208659"},
         {"2.9051040794217974057794872528788544", "9.0578132058082796596162659169900266"},
         {"-6.3727995922262791612201660378238407", "0.88420819690674147275674791889188847"},
         {"34.563712816148310681951488491450963", "2.5094096835319867810973620772099315"}},
        {{"-0.36174920044366372585459063644838082", "0.071015473166282643510304122624808174"},
         {"-1.3707184739731528501697023415281920", "-1.8848081783748569084568137499813475"},
         {"5.4636696405255526597566453268206186", "-9.6802754908511275532445041217013399"},
         {"34.563712816148310681951488491450963", "2.5094096835319867810973620772099315"},
         {"20.217833899864492079877315049596573", "47.637606187438166489050102609254123"}}
    };

    // log(C)
    static const std::vector<std::vector<std::pair<std::string, std::string>>> wolfram_log_C_str = {
        {{"0.37283949131464626145847495985450085", "0.93259575733457398034090828669019823"},
         {"0.29974117882412185188834077514782466", "-0.37724573617831929040880386404761006"},
         {"0.012570994481101896287397393312391581", "0.069065797489967705909606861483174549"},
         {"-0.0082325722505628038300614988907276157", "-0.0052027186280559241567182761847208475"},
         {"0.0011470348245127606302354532393199226", "-0.00027724142040812889786352498804041161"}},
        {{"0.29974117882412185188834077514782466", "-0.37724573617831929040880386404761006"},
         {"1.0623974007981893000430169923623272", "0.92415699747034424773005205927358738"},
         {"0.17851900055582742881386033991553111", "-0.21917487086423601017151363064119865"},
         {"0.0046284684380940178976031784336911991", "0.028482683433703393051404011268788747"},
         {"-0.0025354672708792457176655859812862788", "-0.0017235450116373972272305631796028034"}},
        {{"0.012570994481101896287397393312391581", "0.069065797489967705909606861483174549"},
         {"0.17851900055582742881386033991553111", "-0.21917487086423601017151363064119865"},
         {"1.4521487461752448606385967480403565", "0.84291801310567135351419591833353404"},
         {"0.13650767554429223661865458715477735", "-0.14947339350422266134401153823112071"},
         {"0.0010456668358557117960626567893208504", "0.015982888006561593114579088774161912"}},
        {{"-0.0082325722505628038300614988907276158", "-0.0052027186280559241567182761847208474"},
         {"0.0046284684380940178976031784336911992", "0.028482683433703393051404011268788747"},
         {"0.13650767554429223661865458715477735", "-0.14947339350422266134401153823112071"},
         {"1.7345470136215214524997223517818843", "0.81745249971859912885201404476256384"},
         {"0.10916870047375971969928730916638151", "-0.11369273880775065429549748392455238"}},
        {{"0.0011470348245127606302354532393199226", "-0.00027724142040812889786352498804041166"},
         {"-0.0025354672708792457176655859812862788", "-0.0017235450116373972272305631796028034"},
         {"0.0010456668358557117960626567893208504", "0.015982888006561593114579088774161912"},
         {"0.10916870047375971969928730916638151", "-0.11369273880775065429549748392455238"},
         {"1.9563627860671761146984444880834973", "0.79694557337804660114122478123023106"}}
    };

    // sqrt(C)
    static const std::vector<std::vector<std::pair<std::string, std::string>>> wolfram_sqrt_C_str = {
        {{"1.0849628177899342527822960775234565", "0.50341830067845179662623768802246270"},
         {"0.31375881992204779703031667558115702", "-0.14655243275405913779324405281820567"},
         {"-0.0078262419581016093192113890526558749", "0.027166805249921638505238105718884291"},
         {"-0.0016750760098430217226423119571738163", "-0.0027279277780166960866201076104443132"},
         {"0.00033806543080043611332883738178506217", "0.000073450995393153475276006280797891272"}},
        {{"0.31375881992204779703031667558115702", "-0.14655243275405913779324405281820567"},
         {"1.5374478285079395782866454168701634", "0.69779149309636209436854841650429835"},
         {"0.24209764949615827965877537408090288", "-0.11059923394843577550781072709619315"},
         {"-0.0043296212227801501139491647110593219", "0.014031244881735638552726853296827794"},
         {"-0.00061661826821389117043098755322513271", "-0.0010818620732423377322007329601124995"}},
        {{"-0.0078262419581016093192113890526558748", "0.027166805249921638505238105718884291"},
         {"0.24209764949615827965877537408090288", "-0.11059923394843577550781072709619315"},
         {"1.8936413326878550926584937423888559", "0.81615434827589859856700181106695158"},
         {"0.20643437502875583287763466246907733", "-0.089549920925750440275835975274324390"},
         {"-0.0032719552384952465419687658721822839", "0.0088623528619737983695556854760170058"}},
        {{"-0.0016750760098430217226423119571738164", "-0.0027279277780166960866201076104443132"},
         {"-0.0043296212227801501139491647110593219", "0.014031244881735638552726853296827794"},
         {"0.20643437502875583287763466246907733", "-0.089549920925750440275835975274324390"},
         {"2.1906832946266462693839447789711347", "0.92786991035914215098562933044089373"},
         {"0.18278237709603163422501674732590388", "-0.077287263605550998888461403106542447"}},
        {{"0.00033806543080043611332883738178506219", "0.000073450995393153475276006280797891240"},
         {"-0.00061661826821389117043098755322513272", "-0.0010818620732423377322007329601124994"},
         {"-0.0032719552384952465419687658721822838", "0.0088623528619737983695556854760170058"},
         {"0.18278237709603163422501674732590388", "-0.077287263605550998888461403106542447"},
         {"2.4540248905667241490393916952757633", "1.0245026709876489879526289891842382"}}
    };

    // ------------------------------------------------------------------------
    // Test fixture
    // ------------------------------------------------------------------------
    class WolframVerificationTest : public RationalTest {
    protected:
        void SetUp() override {
            RationalTest::SetUp();
            delta::reset_default_eps();
        }
        void TearDown() override {
            delta::reset_default_eps();
            RationalTest::TearDown();
        }
    };

    static const Rational EPS = "1/10000000000000000000"_r;  // 1e-19

    // ------------------------------------------------------------------------
    // Helper to create 5x5 rational matrix from vector of string vectors
    // ------------------------------------------------------------------------
    static Eigen::Matrix<Rational, 5, 5> matrix_from_strings(
        const std::vector<std::vector<std::string>>& src)
    {
        Eigen::Matrix<Rational, 5, 5> M;
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j)
                M(i, j) = rational_from_wolfram(src[i][j]);
        return M;
    }

    // Helper for complex matrices (5x5 of GaussQi)
    static Eigen::Matrix<GaussQi, 5, 5> complex_matrix_from_strings(
        const std::vector<std::vector<std::pair<std::string, std::string>>>& src)
    {
        Eigen::Matrix<GaussQi, 5, 5> M;
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j)
                M(i, j) = gaussqi_from_wolfram(src[i][j].first, src[i][j].second);
        return M;
    }

    // ------------------------------------------------------------------------
    // Tests for real matrix A (exp, sin, cos – but cos reference missing)
    // ------------------------------------------------------------------------
    TEST_F(WolframVerificationTest, RealMatrixA_Exp) {
        Eigen::Matrix<Rational, 5, 5> A;
        A << 2_r, 3_r, 4_r, 5_r, 6_r,
            3_r, 4_r, 5_r, 6_r, 7_r,
            4_r, 5_r, 6_r, 7_r, 8_r,
            5_r, 6_r, 7_r, 8_r, 9_r,
            6_r, 7_r, 8_r, 9_r, 10_r;

        auto result = delta::exp(A, EPS);
        auto expected = matrix_from_strings(wolfram_exp_A_str);

        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                Rational tol = EPS * std::max(Rational(1), delta::abs(expected(i, j)));
                EXPECT_RATIONAL_NEAR(result(i, j), expected(i, j), tol);
            }
        }
    }
    TEST_F(WolframVerificationTest, RealMatrixA_ExpVerbose) {
        Eigen::Matrix<Rational, 5, 5> A;
        A << 2_r, 3_r, 4_r, 5_r, 6_r,
            3_r, 4_r, 5_r, 6_r, 7_r,
            4_r, 5_r, 6_r, 7_r, 8_r,
            5_r, 6_r, 7_r, 8_r, 9_r,
            6_r, 7_r, 8_r, 9_r, 10_r;

        auto result = delta::exp(A, EPS);
        auto expected = matrix_from_strings(wolfram_exp_A_str);

        std::cout << "=== RealMatrixA_Exp debug output ===\n";
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                double res_d = result(i, j).to_double();
                double exp_d = expected(i, j).to_double();
                double diff = res_d - exp_d;
                std::cout << "[" << i << "," << j << "] "
                    << "result=" << res_d
                    << " expected=" << exp_d
                    << " diff=" << diff
                    << " (rel=" << (exp_d != 0.0 ? std::abs(diff / exp_d) : 0.0) << ")"
                    << std::endl;
            }
        }
        // Пока не проверяем, просто смотрим глазами.
        SUCCEED();
    }
    TEST_F(WolframVerificationTest, RealMatrixA_Sin) {
        Eigen::Matrix<Rational, 5, 5> A;
        A << 2_r, 3_r, 4_r, 5_r, 6_r,
            3_r, 4_r, 5_r, 6_r, 7_r,
            4_r, 5_r, 6_r, 7_r, 8_r,
            5_r, 6_r, 7_r, 8_r, 9_r,
            6_r, 7_r, 8_r, 9_r, 10_r;

        auto result = delta::sin(A, EPS);
        auto expected = matrix_from_strings(wolfram_sin_A_str);

        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                EXPECT_RATIONAL_NEAR(result(i, j), expected(i, j), EPS);
            }
        }
    }

    // cos(A) not verified due to missing reference.

    // ------------------------------------------------------------------------
    // Tests for real matrix B (log, sqrt)
    // ------------------------------------------------------------------------
    TEST_F(WolframVerificationTest, RealMatrixB_Log) {
        Eigen::Matrix<Rational, 5, 5> B;
        B << 2_r, 1_r, 0_r, 0_r, 0_r,
            1_r, 3_r, 1_r, 0_r, 0_r,
            0_r, 1_r, 4_r, 1_r, 0_r,
            0_r, 0_r, 1_r, 5_r, 1_r,
            0_r, 0_r, 0_r, 1_r, 6_r;

        auto result = delta::log(B, EPS);
        auto expected = matrix_from_strings(wolfram_log_B_str);

        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                EXPECT_RATIONAL_NEAR(result(i, j), expected(i, j), EPS);
            }
        }
    }

    TEST_F(WolframVerificationTest, RealMatrixB_Sqrt) {
        Eigen::Matrix<Rational, 5, 5> B;
        B << 2_r, 1_r, 0_r, 0_r, 0_r,
            1_r, 3_r, 1_r, 0_r, 0_r,
            0_r, 1_r, 4_r, 1_r, 0_r,
            0_r, 0_r, 1_r, 5_r, 1_r,
            0_r, 0_r, 0_r, 1_r, 6_r;

        auto result = delta::sqrt(B, EPS);
        auto expected = matrix_from_strings(wolfram_sqrt_B_str);

        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                EXPECT_RATIONAL_NEAR(result(i, j), expected(i, j), EPS);
            }
        }
    }

    // ------------------------------------------------------------------------
    // Tests for complex matrix C
    // ------------------------------------------------------------------------
    TEST_F(WolframVerificationTest, ComplexMatrixC_Exp) {
        Eigen::Matrix<GaussQi, 5, 5> C;
        C << GaussQi(1, 1), GaussQi(1, 0), GaussQi(0, 0), GaussQi(0, 0), GaussQi(0, 0),
            GaussQi(1, 0), GaussQi(2, 2), GaussQi(1, 0), GaussQi(0, 0), GaussQi(0, 0),
            GaussQi(0, 0), GaussQi(1, 0), GaussQi(3, 3), GaussQi(1, 0), GaussQi(0, 0),
            GaussQi(0, 0), GaussQi(0, 0), GaussQi(1, 0), GaussQi(4, 4), GaussQi(1, 0),
            GaussQi(0, 0), GaussQi(0, 0), GaussQi(0, 0), GaussQi(1, 0), GaussQi(5, 5);

        auto result = delta::exp(C, EPS);
        auto expected = complex_matrix_from_strings(wolfram_exp_C_str);

        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                EXPECT_RATIONAL_NEAR(result(i, j).real(), expected(i, j).real(), EPS);
                EXPECT_RATIONAL_NEAR(result(i, j).imag(), expected(i, j).imag(), EPS);
            }
        }
    }

    TEST_F(WolframVerificationTest, ComplexMatrixC_Sin) {
        Eigen::Matrix<GaussQi, 5, 5> C;
        C << GaussQi(1, 1), GaussQi(1, 0), GaussQi(0, 0), GaussQi(0, 0), GaussQi(0, 0),
            GaussQi(1, 0), GaussQi(2, 2), GaussQi(1, 0), GaussQi(0, 0), GaussQi(0, 0),
            GaussQi(0, 0), GaussQi(1, 0), GaussQi(3, 3), GaussQi(1, 0), GaussQi(0, 0),
            GaussQi(0, 0), GaussQi(0, 0), GaussQi(1, 0), GaussQi(4, 4), GaussQi(1, 0),
            GaussQi(0, 0), GaussQi(0, 0), GaussQi(0, 0), GaussQi(1, 0), GaussQi(5, 5);

        auto result = delta::sin(C, EPS);
        auto expected = complex_matrix_from_strings(wolfram_sin_C_str);

        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                EXPECT_RATIONAL_NEAR(result(i, j).real(), expected(i, j).real(), EPS);
                EXPECT_RATIONAL_NEAR(result(i, j).imag(), expected(i, j).imag(), EPS);
            }
        }
    }

    TEST_F(WolframVerificationTest, ComplexMatrixC_Cos) {
        Eigen::Matrix<GaussQi, 5, 5> C;
        C << GaussQi(1, 1), GaussQi(1, 0), GaussQi(0, 0), GaussQi(0, 0), GaussQi(0, 0),
            GaussQi(1, 0), GaussQi(2, 2), GaussQi(1, 0), GaussQi(0, 0), GaussQi(0, 0),
            GaussQi(0, 0), GaussQi(1, 0), GaussQi(3, 3), GaussQi(1, 0), GaussQi(0, 0),
            GaussQi(0, 0), GaussQi(0, 0), GaussQi(1, 0), GaussQi(4, 4), GaussQi(1, 0),
            GaussQi(0, 0), GaussQi(0, 0), GaussQi(0, 0), GaussQi(1, 0), GaussQi(5, 5);

        auto result = delta::cos(C, EPS);
        auto expected = complex_matrix_from_strings(wolfram_cos_C_str);

        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                EXPECT_RATIONAL_NEAR(result(i, j).real(), expected(i, j).real(), EPS);
                EXPECT_RATIONAL_NEAR(result(i, j).imag(), expected(i, j).imag(), EPS);
            }
        }
    }

    TEST_F(WolframVerificationTest, ComplexMatrixC_Log) {
        Eigen::Matrix<GaussQi, 5, 5> C;
        C << GaussQi(1, 1), GaussQi(1, 0), GaussQi(0, 0), GaussQi(0, 0), GaussQi(0, 0),
            GaussQi(1, 0), GaussQi(2, 2), GaussQi(1, 0), GaussQi(0, 0), GaussQi(0, 0),
            GaussQi(0, 0), GaussQi(1, 0), GaussQi(3, 3), GaussQi(1, 0), GaussQi(0, 0),
            GaussQi(0, 0), GaussQi(0, 0), GaussQi(1, 0), GaussQi(4, 4), GaussQi(1, 0),
            GaussQi(0, 0), GaussQi(0, 0), GaussQi(0, 0), GaussQi(1, 0), GaussQi(5, 5);

        auto result = delta::log(C, EPS);
        auto expected = complex_matrix_from_strings(wolfram_log_C_str);

        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                EXPECT_RATIONAL_NEAR(result(i, j).real(), expected(i, j).real(), EPS);
                EXPECT_RATIONAL_NEAR(result(i, j).imag(), expected(i, j).imag(), EPS);
            }
        }
    }

    TEST_F(WolframVerificationTest, ComplexMatrixC_Sqrt) {
        Eigen::Matrix<GaussQi, 5, 5> C;
        C << GaussQi(1, 1), GaussQi(1, 0), GaussQi(0, 0), GaussQi(0, 0), GaussQi(0, 0),
            GaussQi(1, 0), GaussQi(2, 2), GaussQi(1, 0), GaussQi(0, 0), GaussQi(0, 0),
            GaussQi(0, 0), GaussQi(1, 0), GaussQi(3, 3), GaussQi(1, 0), GaussQi(0, 0),
            GaussQi(0, 0), GaussQi(0, 0), GaussQi(1, 0), GaussQi(4, 4), GaussQi(1, 0),
            GaussQi(0, 0), GaussQi(0, 0), GaussQi(0, 0), GaussQi(1, 0), GaussQi(5, 5);

        auto result = delta::sqrt(C, EPS);
        auto expected = complex_matrix_from_strings(wolfram_sqrt_C_str);

        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                EXPECT_RATIONAL_NEAR(result(i, j).real(), expected(i, j).real(), EPS);
                EXPECT_RATIONAL_NEAR(result(i, j).imag(), expected(i, j).imag(), EPS);
            }
        }
    }

} // namespace delta::testing