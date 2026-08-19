#ifndef TIMER_HPP
#define TIMER_HPP

#include <cstdio>

namespace gsparc
{
    class Timer
    {
    private:
        double start_time_; 
        double total_time_;
        bool running_;     

    public:
        Timer();

        void start();

        void stop();

        double currentElapsed() const;

        void printElapsed(const char *msg = "Elapsed time") const;

        void printTotal(const char *msg = "Total elapsed time") const;

        double getTotalTime() const;

        void reset();
    };

} // namespace common

#endif // TIMER_HPP
