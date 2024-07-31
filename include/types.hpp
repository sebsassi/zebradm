#pragma once

#include "multi_span.hpp"
#include "zest/real_sh_expansion.hpp"

template <typename ElementType>
using SHExpansionSpan = zest::st::RealSHExpansionSpanGeo<ElementType>;

template <typename ElementType>
using SHExpansionCollectionSpan = SuperSpan<SHExpansionSpan<ElementType>>;