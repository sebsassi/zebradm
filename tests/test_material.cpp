/*
Copyright (c) 2024-2026 Sebastian Sassi

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
*/

#include <cassert>

#include "material.hpp"

namespace
{

void test_tiny_string()
{
    assert(zdm::detail::TinyString<0>{""}.view() == "");
    assert(zdm::detail::TinyString<0>{"asdf"}.view() == "");

    assert(zdm::detail::TinyString<2>{""}.view() == "");
    assert(zdm::detail::TinyString<2>{"asdf"}.view() == "as");
    assert(zdm::detail::TinyString<2>{"a"}.view() == "a");
}

void test_atomic_number_from_chemical_symbol()
{
    // All existing elements map to correct numbers.
    assert(zdm::atomic_number_of("H") == 1);
    assert(zdm::atomic_number_of("He") == 2);
    assert(zdm::atomic_number_of("Li") == 3);
    assert(zdm::atomic_number_of("Be") == 4);
    assert(zdm::atomic_number_of("B") == 5);
    assert(zdm::atomic_number_of("C") == 6);
    assert(zdm::atomic_number_of("N") == 7);
    assert(zdm::atomic_number_of("O") == 8);
    assert(zdm::atomic_number_of("F") == 9);
    assert(zdm::atomic_number_of("Ne") == 10);
    assert(zdm::atomic_number_of("Na") == 11);
    assert(zdm::atomic_number_of("Mg") == 12);
    assert(zdm::atomic_number_of("Al") == 13);
    assert(zdm::atomic_number_of("Si") == 14);
    assert(zdm::atomic_number_of("P") == 15);
    assert(zdm::atomic_number_of("S") == 16);
    assert(zdm::atomic_number_of("Cl") == 17);
    assert(zdm::atomic_number_of("Ar") == 18);
    assert(zdm::atomic_number_of("K") == 19);
    assert(zdm::atomic_number_of("Ca") == 20);
    assert(zdm::atomic_number_of("Sc") == 21);
    assert(zdm::atomic_number_of("Ti") == 22);
    assert(zdm::atomic_number_of("V") == 23);
    assert(zdm::atomic_number_of("Cr") == 24);
    assert(zdm::atomic_number_of("Mn") == 25);
    assert(zdm::atomic_number_of("Fe") == 26);
    assert(zdm::atomic_number_of("Co") == 27);
    assert(zdm::atomic_number_of("Ni") == 28);
    assert(zdm::atomic_number_of("Cu") == 29);
    assert(zdm::atomic_number_of("Zn") == 30);
    assert(zdm::atomic_number_of("Ga") == 31);
    assert(zdm::atomic_number_of("Ge") == 32);
    assert(zdm::atomic_number_of("As") == 33);
    assert(zdm::atomic_number_of("Se") == 34);
    assert(zdm::atomic_number_of("Br") == 35);
    assert(zdm::atomic_number_of("Kr") == 36);
    assert(zdm::atomic_number_of("Rb") == 37);
    assert(zdm::atomic_number_of("Sr") == 38);
    assert(zdm::atomic_number_of("Y") == 39);
    assert(zdm::atomic_number_of("Zr") == 40);
    assert(zdm::atomic_number_of("Nb") == 41);
    assert(zdm::atomic_number_of("Mo") == 42);
    assert(zdm::atomic_number_of("Tc") == 43);
    assert(zdm::atomic_number_of("Ru") == 44);
    assert(zdm::atomic_number_of("Rn") == 45);
    assert(zdm::atomic_number_of("Pd") == 46);
    assert(zdm::atomic_number_of("Ag") == 47);
    assert(zdm::atomic_number_of("Cd") == 48);
    assert(zdm::atomic_number_of("In") == 49);
    assert(zdm::atomic_number_of("Sn") == 50);
    assert(zdm::atomic_number_of("Sb") == 51);
    assert(zdm::atomic_number_of("Te") == 52);
    assert(zdm::atomic_number_of("I") == 53);
    assert(zdm::atomic_number_of("Xe") == 54);
    assert(zdm::atomic_number_of("Cs") == 55);
    assert(zdm::atomic_number_of("Ba") == 56);
    assert(zdm::atomic_number_of("La") == 57);
    assert(zdm::atomic_number_of("Ce") == 58);
    assert(zdm::atomic_number_of("Pr") == 59);
    assert(zdm::atomic_number_of("Nd") == 60);
    assert(zdm::atomic_number_of("Pm") == 61);
    assert(zdm::atomic_number_of("Sm") == 62);
    assert(zdm::atomic_number_of("Eu") == 63);
    assert(zdm::atomic_number_of("Gd") == 64);
    assert(zdm::atomic_number_of("Tb") == 65);
    assert(zdm::atomic_number_of("Dy") == 66);
    assert(zdm::atomic_number_of("Ho") == 67);
    assert(zdm::atomic_number_of("Er") == 68);
    assert(zdm::atomic_number_of("Tm") == 69);
    assert(zdm::atomic_number_of("Yb") == 70);
    assert(zdm::atomic_number_of("Lu") == 71);
    assert(zdm::atomic_number_of("Hf") == 72);
    assert(zdm::atomic_number_of("Ta") == 73);
    assert(zdm::atomic_number_of("W") == 74);
    assert(zdm::atomic_number_of("Re") == 75);
    assert(zdm::atomic_number_of("Os") == 76);
    assert(zdm::atomic_number_of("Ir") == 77);
    assert(zdm::atomic_number_of("Pt") == 78);
    assert(zdm::atomic_number_of("Au") == 79);
    assert(zdm::atomic_number_of("Hg") == 80);
    assert(zdm::atomic_number_of("Tl") == 81);
    assert(zdm::atomic_number_of("Pb") == 82);
    assert(zdm::atomic_number_of("Bi") == 83);
    assert(zdm::atomic_number_of("Po") == 84);
    assert(zdm::atomic_number_of("At") == 85);
    assert(zdm::atomic_number_of("Rn") == 86);
    assert(zdm::atomic_number_of("Fr") == 87);
    assert(zdm::atomic_number_of("Ra") == 88);
    assert(zdm::atomic_number_of("Ac") == 89);
    assert(zdm::atomic_number_of("Th") == 90);
    assert(zdm::atomic_number_of("Pa") == 91);
    assert(zdm::atomic_number_of("U") == 92);
    assert(zdm::atomic_number_of("Np") == 93);
    assert(zdm::atomic_number_of("Pu") == 94);
    assert(zdm::atomic_number_of("Am") == 95);
    assert(zdm::atomic_number_of("Cm") == 96);
    assert(zdm::atomic_number_of("Bk") == 97);
    assert(zdm::atomic_number_of("Cf") == 98);
    assert(zdm::atomic_number_of("Es") == 99);
    assert(zdm::atomic_number_of("Fm") == 100);
    assert(zdm::atomic_number_of("Md") == 101);
    assert(zdm::atomic_number_of("No") == 102);
    assert(zdm::atomic_number_of("Lr") == 103);
    assert(zdm::atomic_number_of("Rf") == 104);
    assert(zdm::atomic_number_of("Db") == 105);
    assert(zdm::atomic_number_of("Sg") == 106);
    assert(zdm::atomic_number_of("Bh") == 107);
    assert(zdm::atomic_number_of("Hs") == 108);
    assert(zdm::atomic_number_of("Mt") == 109);
    assert(zdm::atomic_number_of("Ds") == 110);
    assert(zdm::atomic_number_of("Rg") == 111);
    assert(zdm::atomic_number_of("Cn") == 112);
    assert(zdm::atomic_number_of("Nh") == 113);
    assert(zdm::atomic_number_of("Fl") == 114);
    assert(zdm::atomic_number_of("Mc") == 115);
    assert(zdm::atomic_number_of("Lv") == 116);
    assert(zdm::atomic_number_of("Ts") == 117);
    assert(zdm::atomic_number_of("Og") == 118);

    // Empty string maps to null.
    assert(zdm::atomic_number_of("") == 0);

    // Strings containing invalid characters map to null.
    assert(zdm::atomic_number_of("8") == 0);
    assert(zdm::atomic_number_of("8?") == 0);
    assert(zdm::atomic_number_of("$#") == 0);
    assert(zdm::atomic_number_of("$b") == 0);
    assert(zdm::atomic_number_of("R#") == 0);

    // Single character lower case symbols map to null.
    assert(zdm::atomic_number_of("h") == 0);
    assert(zdm::atomic_number_of("n") == 0);
    assert(zdm::atomic_number_of("p") == 0);
    assert(zdm::atomic_number_of("f") == 0);

    // Double character lower case symbols map to null.
    assert(zdm::atomic_number_of("ca") == 0);
    assert(zdm::atomic_number_of("pb") == 0);
    assert(zdm::atomic_number_of("xe") == 0);

    // Fully capitalized double character symbols map to null.
    assert(zdm::atomic_number_of("CA") == 0);
    assert(zdm::atomic_number_of("PB") == 0);
    assert(zdm::atomic_number_of("XE") == 0);

    // Double character symbols with inverted capitalization map to null.
    assert(zdm::atomic_number_of("cA") == 0);
    assert(zdm::atomic_number_of("pB") == 0);
    assert(zdm::atomic_number_of("xE") == 0);

    // Noexistent elements with valid capitalization map to null.
    assert(zdm::atomic_number_of("J") == 0);
    assert(zdm::atomic_number_of("L") == 0);
    assert(zdm::atomic_number_of("Q") == 0);
    assert(zdm::atomic_number_of("Fu") == 0);
    assert(zdm::atomic_number_of("Jq") == 0);
    assert(zdm::atomic_number_of("Xp") == 0);
}

void test_chemical_symbol_from_atomic_number()
{
    // Null element maps to empty string.
    assert("" == zdm::chemical_symbol_of(0));

    // All existing elements map to correct numbers.
    assert("H" == zdm::chemical_symbol_of(1));
    assert("He" == zdm::chemical_symbol_of(2));
    assert("Li" == zdm::chemical_symbol_of(3));
    assert("Be" == zdm::chemical_symbol_of(4));
    assert("B" == zdm::chemical_symbol_of(5));
    assert("C" == zdm::chemical_symbol_of(6));
    assert("N" == zdm::chemical_symbol_of(7));
    assert("O" == zdm::chemical_symbol_of(8));
    assert("F" == zdm::chemical_symbol_of(9));
    assert("Ne" == zdm::chemical_symbol_of(10));
    assert("Na" == zdm::chemical_symbol_of(11));
    assert("Mg" == zdm::chemical_symbol_of(12));
    assert("Al" == zdm::chemical_symbol_of(13));
    assert("Si" == zdm::chemical_symbol_of(14));
    assert("P" == zdm::chemical_symbol_of(15));
    assert("S" == zdm::chemical_symbol_of(16));
    assert("Cl" == zdm::chemical_symbol_of(17));
    assert("Ar" == zdm::chemical_symbol_of(18));
    assert("K" == zdm::chemical_symbol_of(19));
    assert("Ca" == zdm::chemical_symbol_of(20));
    assert("Sc" == zdm::chemical_symbol_of(21));
    assert("Ti" == zdm::chemical_symbol_of(22));
    assert("V" == zdm::chemical_symbol_of(23));
    assert("Cr" == zdm::chemical_symbol_of(24));
    assert("Mn" == zdm::chemical_symbol_of(25));
    assert("Fe" == zdm::chemical_symbol_of(26));
    assert("Co" == zdm::chemical_symbol_of(27));
    assert("Ni" == zdm::chemical_symbol_of(28));
    assert("Cu" == zdm::chemical_symbol_of(29));
    assert("Zn" == zdm::chemical_symbol_of(30));
    assert("Ga" == zdm::chemical_symbol_of(31));
    assert("Ge" == zdm::chemical_symbol_of(32));
    assert("As" == zdm::chemical_symbol_of(33));
    assert("Se" == zdm::chemical_symbol_of(34));
    assert("Br" == zdm::chemical_symbol_of(35));
    assert("Kr" == zdm::chemical_symbol_of(36));
    assert("Rb" == zdm::chemical_symbol_of(37));
    assert("Sr" == zdm::chemical_symbol_of(38));
    assert("Y" == zdm::chemical_symbol_of(39));
    assert("Zr" == zdm::chemical_symbol_of(40));
    assert("Nb" == zdm::chemical_symbol_of(41));
    assert("Mo" == zdm::chemical_symbol_of(42));
    assert("Tc" == zdm::chemical_symbol_of(43));
    assert("Ru" == zdm::chemical_symbol_of(44));
    assert("Rn" == zdm::chemical_symbol_of(45));
    assert("Pd" == zdm::chemical_symbol_of(46));
    assert("Ag" == zdm::chemical_symbol_of(47));
    assert("Cd" == zdm::chemical_symbol_of(48));
    assert("In" == zdm::chemical_symbol_of(49));
    assert("Sn" == zdm::chemical_symbol_of(50));
    assert("Sb" == zdm::chemical_symbol_of(51));
    assert("Te" == zdm::chemical_symbol_of(52));
    assert("I" == zdm::chemical_symbol_of(53));
    assert("Xe" == zdm::chemical_symbol_of(54));
    assert("Cs" == zdm::chemical_symbol_of(55));
    assert("Ba" == zdm::chemical_symbol_of(56));
    assert("La" == zdm::chemical_symbol_of(57));
    assert("Ce" == zdm::chemical_symbol_of(58));
    assert("Pr" == zdm::chemical_symbol_of(59));
    assert("Nd" == zdm::chemical_symbol_of(60));
    assert("Pm" == zdm::chemical_symbol_of(61));
    assert("Sm" == zdm::chemical_symbol_of(62));
    assert("Eu" == zdm::chemical_symbol_of(63));
    assert("Gd" == zdm::chemical_symbol_of(64));
    assert("Tb" == zdm::chemical_symbol_of(65));
    assert("Dy" == zdm::chemical_symbol_of(66));
    assert("Ho" == zdm::chemical_symbol_of(67));
    assert("Er" == zdm::chemical_symbol_of(68));
    assert("Tm" == zdm::chemical_symbol_of(69));
    assert("Yb" == zdm::chemical_symbol_of(70));
    assert("Lu" == zdm::chemical_symbol_of(71));
    assert("Hf" == zdm::chemical_symbol_of(72));
    assert("Ta" == zdm::chemical_symbol_of(73));
    assert("W" == zdm::chemical_symbol_of(74));
    assert("Re" == zdm::chemical_symbol_of(75));
    assert("Os" == zdm::chemical_symbol_of(76));
    assert("Ir" == zdm::chemical_symbol_of(77));
    assert("Pt" == zdm::chemical_symbol_of(78));
    assert("Au" == zdm::chemical_symbol_of(79));
    assert("Hg" == zdm::chemical_symbol_of(80));
    assert("Tl" == zdm::chemical_symbol_of(81));
    assert("Pb" == zdm::chemical_symbol_of(82));
    assert("Bi" == zdm::chemical_symbol_of(83));
    assert("Po" == zdm::chemical_symbol_of(84));
    assert("At" == zdm::chemical_symbol_of(85));
    assert("Rn" == zdm::chemical_symbol_of(86));
    assert("Fr" == zdm::chemical_symbol_of(87));
    assert("Ra" == zdm::chemical_symbol_of(88));
    assert("Ac" == zdm::chemical_symbol_of(89));
    assert("Th" == zdm::chemical_symbol_of(90));
    assert("Pa" == zdm::chemical_symbol_of(91));
    assert("U" == zdm::chemical_symbol_of(92));
    assert("Np" == zdm::chemical_symbol_of(93));
    assert("Pu" == zdm::chemical_symbol_of(94));
    assert("Am" == zdm::chemical_symbol_of(95));
    assert("Cm" == zdm::chemical_symbol_of(96));
    assert("Bk" == zdm::chemical_symbol_of(97));
    assert("Cf" == zdm::chemical_symbol_of(98));
    assert("Es" == zdm::chemical_symbol_of(99));
    assert("Fm" == zdm::chemical_symbol_of(100));
    assert("Md" == zdm::chemical_symbol_of(101));
    assert("No" == zdm::chemical_symbol_of(102));
    assert("Lr" == zdm::chemical_symbol_of(103));
    assert("Rf" == zdm::chemical_symbol_of(104));
    assert("Db" == zdm::chemical_symbol_of(105));
    assert("Sg" == zdm::chemical_symbol_of(106));
    assert("Bh" == zdm::chemical_symbol_of(107));
    assert("Hs" == zdm::chemical_symbol_of(108));
    assert("Mt" == zdm::chemical_symbol_of(109));
    assert("Ds" == zdm::chemical_symbol_of(110));
    assert("Rg" == zdm::chemical_symbol_of(111));
    assert("Cn" == zdm::chemical_symbol_of(112));
    assert("Nh" == zdm::chemical_symbol_of(113));
    assert("Fl" == zdm::chemical_symbol_of(114));
    assert("Mc" == zdm::chemical_symbol_of(115));
    assert("Lv" == zdm::chemical_symbol_of(116));
    assert("Ts" == zdm::chemical_symbol_of(117));
    assert("Og" == zdm::chemical_symbol_of(118));

    // Elements that have not been observed map to empty string.
    assert("" == zdm::chemical_symbol_of(128));
    assert("" == zdm::chemical_symbol_of(255));
}

void test_isotope()
{
    assert(zdm::Isotope{}.is_null());        // Null isotope.
    assert(!zdm::Isotope{2, 4}.is_null());   // Valid isotope is not null.
    assert(!zdm::Isotope{2, 0}.is_null());   // Invalid isotope is not null.
    assert(!zdm::Isotope{0, 7}.is_null());   // Nonnull isotope of null element is not null.

    assert(!zdm::Isotope{}.is_valid());          // Null isotope is not a valid isotope.
    assert(zdm::Isotope{2, 2}.is_valid());       // Isotope with A >= Z is valid.
    assert(zdm::Isotope{1, 50000}.is_valid())    // Helium with 50000 neutrons is valid.
    assert(!zdm::Isotope{2, 0}.is_valid());      // Isotope with A < Z is invalid.
    assert(!zdm::Isotope{0, 7}.is_valid());      // Isotope of null element is invalid.
    assert(zdm::Isotope{128, 130}.is_valid());   // Isotope of unobserved element is valid.
}

void test_element()
{
    assert(zdm::Element{}.is_null());    // Null element.
    assert(!zdm::Element{2}.is_null());  // All other elements are not.

    assert(!zdm::Element{}.is_valid());      // Null element is not valid.
    assert(zdm::Element{2}.is_valid());      // All other elements are valid.
    assert(zdm::Element{255}.is_valid());    // All of them.

    assert(zdm::Element{"U"} == zdm::Element{92});    // Elements can be constructed from their symbols.
    assert(zdm::Element{"3x"} == zdm::Element{});     // Invalid symbols map to null element.
}

void test_material()
{
    assert(zdm::Material::from_chemical_formula("") == std::unexpected(zdm::FormulaParseStatus::empty_formula));
    assert(zdm::Material::from_chemical_formula("#/") == std::unexpected(zdm::FormulaParseStatus::invalid_element));
    assert(zdm::Material::from_chemical_formula("17He") == std::unexpected(zdm::FormulaParseStatus::invalid_element));
    assert(zdm::Material::from_chemical_formula("h2") == std::unexpected(zdm::FormulaParseStatus::invalid_element));
    assert(zdm::Material::from_chemical_formula("Vg") == std::unexpected(zdm::FormulaParseStatus::invalid_element));
    assert(zdm::Material::from_chemical_formula("HE") == std::unexpected(zdm::FormulaParseStatus::invalid_element));
    assert(zdm::Material::from_chemical_formula("H256") == std::unexpected(zdm::FormulaParseStatus::count_of_element_too_large));
    assert(zdm::Material::from_chemical_formula(
        "Li""Be""C""H""Pb""W""Ti""F""Ne""Og""K""I""Xe""Au""U""Eu""Fm""As""Ge""Na""Zn""In""Po""Ac""Hf""Y""Ru""Yr""Os""Tc""Yb""Cs""Si" == std::unexpected(std::FormulaParseStatus::too_many_elements)));

    assert(zdm::Material::from_chemical_formula("He") == zdm::Material{{{zdm::Element{"He"}, 1}}});
    assert(zdm::Material::from_chemical_formula("H2O")
        == zdm::Material{{
                {zdm::Element{"H"}, 2},
                {zdm::Element{"O"}, 1},
            }});
    assert(zdm::Material::from_chemical_formula("CeOFeAs")
        == zdm::Material{{
                {zdm::Element{"Ce"}, 1},
                {zdm::Element{"O"}, 1},
                {zdm::Element{"Fe"}, 1},
                {zdm::Element{"As"}, 1},
           }});
    assert(zdm::Material::from_chemical_formula("Hg12Tl3Ba30Ca30Cu45O127")
        == zdm::Material{{
                {zdm::Element{"Hg"}, 12},
                {zdm::Element{"Tl"}, 3},
                {zdm::Element{"Ba"}, 30},
                {zdm::Element{"Ca"}, 30},
                {zdm::Element{"Cu"}, 45},
                {zdm::Element{"O"}, 127},
            }});
}

}
