/*
Copyright (c) 2026 Sebastian Sassi

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
#pragma once

#include <cstdint>
#include <vector>
#include <span>

namespace zdm
{

template <typename T>
class RaggedTable
{
public:
    enum OffsetSource: std::uint8_t
    {
        size_source,
        offset_source,
    };

    using size_type = std::size_t;

    RaggedTable() = default;

    template <OffsetSource>
    [[nodiscard]] RaggedTable from(std::span<const size_type> source);

    template <>
    [[nodiscard]] RaggedTable from<size_source>(std::span<const size_type> sizes)
    {
        return RaggedTable{from_sizes_tag{}, sizes};
    }

    template <>
    [[nodiscard]] RaggedTable from<offset_source>(std::span<const size_type> offsets)
    {
        return RaggedTable{from_offsets_tag{}, offsets};
    }

    [[nodiscard]] size_type size() const noexcept { return m_data.size(); }
    [[nodiscard]] size_type extent() const noexcept
    {
        return std::max(1UL, m_offsets.size()) - 1;
    }

    [[nodiscard]] std::span<const size_type> offsets() const noexcept { return m_data.offsets(); }

    void clear()
    {
        m_data.clear();
        m_offsets.clear();
    }

    template <OffsetSource>
    void reshape(std::span<const size_type> source);

    template <>
    void reshape<size_source>(std::span<const size_type> sizes)
    {
        m_data.resize(std::ranges::fold_left(sizes, 0, std::plus{}));
        m_offsets.resize(sizes.size() + 1);

        initialize_offsets(sizes);
    }

    template <>
    void reshape<offset_source>(std::span<const size_type> offsets)
    {
        m_data.resize(offsets.back());
        m_offsets.resize(offsets.size());
        std::ranges::copy(offsets, m_offsets.begin());
    }

    std::span<T> append(size_type size)
    {
        m_data.resize(m_data.size() + size);
        m_offsets.emplace_back(m_data.size());
        return back();
    }

    std::span<T> append(std::span<const T> row)
    {
        m_data.append_range(row);
        m_offsets.emplace_back(m_data.size());
        return back();
    }

    [[nodiscard]] std::span<T> flatten() noexcept
    {
        return m_data;
    }

    [[nodiscard]] std::span<const T> flatten() const noexcept
    {
        return m_data;
    }

    [[nodiscard]] std::span<T> front() noexcept
    {
        return {
            m_data.data(),
            m_offsets[1] - m_offsets.front(),
        };
    }

    [[nodiscard]] std::span<const T> front() const noexcept
    {
        return {
            m_data.data(),
            m_offsets[1] - m_offsets.front(),
        };
    }

    [[nodiscard]] std::span<T> back() noexcept
    {
        return {
            m_data.data() + m_offsets[m_data.size() - 1],
            m_offsets.back() - m_offsets[m_data.size() - 1],
        };
    }

    [[nodiscard]] std::span<const T> back() const noexcept
    {
        return {
            m_data.data() + m_offsets[m_data.size() - 1],
            m_offsets.back() - m_offsets[m_data.size() - 1],
        };
    }

    [[nodiscard]] std::span<T> operator[](size_type i) noexcept
    {
        return {
            m_data.data() + m_offsets[i],
            m_offsets[i + 1] - m_offsets[i],
        };
    }

    [[nodiscard]] std::span<const T> operator[](size_type i) const noexcept
    {
        return {
            m_data.data() + m_offsets[i],
            m_offsets[i + 1] - m_offsets[i],
        };
    }

private:
    struct from_offsets_tag {};
    struct from_sizes_tag {};

    RaggedTable(from_sizes_tag, std::span<const size_type> sizes):
        m_offsets(sizes.size() + 1)
    {
        initialize_offsets(sizes);
        m_data.resize(m_offsets.back());
    }

    RaggedTable(from_offsets_tag, std::span<const size_type> offsets):
        m_data(offsets.back()), m_offsets(std::ranges::to<std::vector>(offsets)) {}

    void initialize_offsets(std::span<const size_type> sizes)
    {
        size_type offset = 0;
        for (std::size_t i = 0; i < sizes.size(); ++i)
        {
            m_offsets[i] = offset;
            offset += sizes[i];
        }
        m_offsets.back() = offset;
    }

    std::vector<T> m_data;
    std::vector<size_type> m_offsets;
};

}
