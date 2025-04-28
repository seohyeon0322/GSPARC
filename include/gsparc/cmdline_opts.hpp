#ifndef CMDLINE_OPTS_HPP_
#define CMDLINE_OPTS_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <inttypes.h>
#include <getopt.h>


namespace gsparc
{
    class CommandLineOptions
    {
    public:
        CommandLineOptions();
        ~CommandLineOptions();

        void Parse(int argc, char **argv);

        const char *get_first_input_path() const { return _tensor_X; }
        const char *get_second_input_path() const { return _tensor_Y; }
        const int *get_first_input_cmodes() const { return _cmodes_X; }
        const int *get_second_input_cmodes() const { return _cmodes_Y; }
        const int *get_third_cmodes() const { return _cmodes_Z; }
        const int get_num_cmodes() const { return _num_cmodes; }
        const int get_gpu_count() const { return _gpu_count; }
        const int get_quantum() const { return _qunatum; }
        const int get_num_iter() const { return _num_iter; }
        const int get_add_num_cmodes() const { return _num_acmodes; }
        const int is_multi_sptc() const { return _multi_sptc; }
        const int is_dense_accumulator() const { return _dense_accumulator; }
        
        void Initialize();

    private:
        option *long_opt;
        const char *const short_opt;

        char * _tensor_X;
        char * _tensor_Y;
        int *_cmodes_X, *_cmodes_Y, *_cmodes_Z;
        int _num_cmodes;
        int _gpu_count;
        int _qunatum;
        int _num_iter;
        int _multi_sptc;
        int _num_acmodes;
        int _dense_accumulator;

    }; // class CommandLineOptions
} // namespace gsparc

#endif // CMDLINE_OPTS_HPP_