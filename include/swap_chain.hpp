#pragma once

#include <cstddef>
#include <span>
#include <array>

#include <zest/md_array.hpp>

namespace zdm::util
{

template <typename ElementType, std::size_t N>
class SwapChain
{
public:
    SwapChain() = default;
    SwapChain(std::size_t size): m_buffer{size}
    {
        for (std::size_t i = 0; i < N; ++i)
            m_chain[i] = i;
    }

    void resize(std::size_t size) { m_buffer.reshape(size); }

    [[nodiscard]] std::size_t buffer_size() const noexcept { return m_buffer.extent(1); }

    template <std::size_t index>
        requires (index < N)
    [[nodiscard]] std::span<double> previous() noexcept { return m_buffer[m_chain[index]].flatten(); }

    template <std::size_t index>
        requires (index < N)
    [[nodiscard]] std::span<const ElementType> previous() const noexcept { return m_buffer[m_chain[index]].flatten(); }

    [[nodiscard]] std::span<ElementType> current() noexcept { return m_buffer[m_chain[0]].flatten(); }
    [[nodiscard]] std::span<const ElementType> current() const noexcept { return m_buffer[m_chain[0]].flatten(); }

    [[nodiscard]] std::span<ElementType> next() noexcept { return m_buffer[m_chain.back()].flatten(); }
    [[nodiscard]] std::span<const ElementType> next() const noexcept { return m_buffer[m_chain.back()].flatten(); }

    void advance()
    {
        const std::size_t back = m_chain.back();
        for (std::size_t i = N - 1; i > 0; --i)
            m_chain[i] = m_chain[i - 1];

        m_chain[0] = back;
    }

private:
    zest::MDArray<ElementType, N, std::dynamic_extent> m_buffer;
    std::array<std::size_t, N> m_chain;
};

} // namespace zdm::util
