#include "gsparc/timer.hpp"
#include <omp.h>
#include <iostream>

namespace gsparc
{

    Timer::Timer() : start_time_(0), total_time_(0), running_(false) {}

    void Timer::start()
    {
        if (!running_)
        {
            start_time_ = omp_get_wtime();
            
            running_ = true;
        }
        else
        {
            std::fprintf(stderr, "Timer is already running!\n");
        }
    }

    void Timer::stop()
    {
        if (running_)
        {
            double elapsed = omp_get_wtime() - start_time_;
            total_time_ += elapsed;
            running_ = false;
        }
        else
        {
            std::fprintf(stderr, "Timer is not running!\n");
        }
    }

    double Timer::currentElapsed() const
    {
        if (running_)
        {
            return omp_get_wtime() - start_time_;
        }
        return 0.0;
    }

    void Timer::printElapsed(const char *msg) const
    {
        double elapsed = omp_get_wtime() - start_time_;
        printf("%s: %f s\n", msg, elapsed);
    }

    void Timer::printTotal(const char *msg) const
    {
        printf("%s: %f s\n", msg, total_time_);
    }

    double Timer::getTotalTime() const
    {
        return total_time_;
    }

    void Timer::reset()
    {
        total_time_ = 0.0;
        running_ = false;
    }

} // namespace common
