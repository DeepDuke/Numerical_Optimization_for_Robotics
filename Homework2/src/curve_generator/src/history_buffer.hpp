#pragma once

#include <deque>
#include <vector>

template <typename Datatype>
class HistoryBuffer
{
   public:
    HistoryBuffer() : max_buffer_size_(30) {}

    HistoryBuffer(size_t max_buffer_size) : max_buffer_size_(max_buffer_size) {}

    ~HistoryBuffer() = default;

    void AddData(const Datatype& data)
    {
        // Pop old data
        while (history_.size() > max_buffer_size_ - 1)
        {
            history_.pop_front();
        }
        // Add new data
        history_.push_back(data);
    }

    std::vector<Datatype> GetHistory()
    {
        std::vector<Datatype> v{history_.begin(), history_.end()};
        return v;
    }

    void Clear()
    {
        history_.clear();
    }

    size_t Size()
    {
        return history_.size();
    }

   private:
    size_t max_buffer_size_;
    std::deque<Datatype> history_;
};
