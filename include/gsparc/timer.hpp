#ifndef TIMER_HPP
#define TIMER_HPP

#include <cstdio>

namespace gsparc
{
    class Timer
    {
    private:
        double start_time_; // 마지막 start() 호출 시각 저장
        double total_time_; // 누적된 총 시간
        bool running_;      // 타이머 실행 중 여부

    public:
        // 기본 생성자: 총 시간 0, 실행중 아님.
        Timer();

        // 타이머 시작: 이미 실행중이면 경고 메시지 출력.
        void start();

        // 타이머 종료: 마지막 시작부터 경과한 시간만큼 누적.
        void stop();

        // 현재 사이클의 경과 시간을 반환 (타이머가 실행중일 때)
        double currentElapsed() const;

        // 현재 사이클의 경과 시간을 출력 (실행 중이면 그 시점까지의 시간)
        void printElapsed(const char *msg = "Elapsed time") const;

        // 누적된 총 시간을 출력
        void printTotal(const char *msg = "Total elapsed time") const;

        // 누적된 총 시간을 반환
        double getTotalTime() const;

        // 타이머 초기화: 총 시간을 0으로 리셋하고, 실행 중인 경우 정지.
        void reset();
    };

} // namespace common

#endif // TIMER_HPP
