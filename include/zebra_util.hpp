#include "zest/sh_glq_transformer.hpp"

namespace zebra {

template <typename ElementType>
using SHExpansionCollectionSpan = SuperSpan<SHExpansionSpan<ElementType>>;

class ResponseTransformer
{
public:
    ResponseTransformer() = default;
    explicit ResponseTransformer(std::size_t order): m_transformer(order) {}

    template <typename RespType>
    void transform(
        RespType&& resp, std::span<const double> min_speeds, SHExpansionCollectionSpan<std::array<double, 2>> out)
    {
        for (std::size_t i = 0; i < min_speeds.size(); ++i)
        {
            const double min_speed = min_speeds[i];
            auto surface_func = [&](double lon, double colat) -> double
            {
                return resp(min_speed, lon, colat);
            };
            m_transformer.transform(surface_func, out[i]);
        }
    }
private:
    zest::st::SHTransformerGeo<> m_transformer;
};

}; // namespace zebra