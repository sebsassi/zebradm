#pragma once

#include "multi_span.hpp"
#include "zest/real_sh_expansion.hpp"

template <typename ElementType>
using SHExpansionCollectionSpan = SuperSpan<SHExpansionSpan<ElementType>>;