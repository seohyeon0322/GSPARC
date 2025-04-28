#ifndef CMDLINE_PARSER_HPP
#define CMDLINE_PARSER_HPP

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <getopt.h>
#include <omp.h>
#include <time.h>
#include <memory>
#include <iostream>

#include "common.hpp"


class CommandLineParser {
    public:

    CommandLineParser();
    ~CommandLineParser();
    inline std::string get_tensor_X() const { return _tensor_X; }
    inline std::string get_tensor_Y() const { return _tensor_Y; }
    inline IType* get_cmodes_X() const { return _cmodes_X; }
    inline IType* get_cmodes_Y() const { return _cmodes_Y; }
    inline int get_num_cmodes() const { return _num_cmodes; }
    inline int get_k() const { return _k; }
    inline int get_g() const { return _g; }
    inline int get_q() const { return _q; }
    inline int get_p() const { return _p; }
    inline std::string get_cmode_name() const {return std::to_string(_num_cmodes) + "_" + std::to_string(_k);}
    void Initialize();
    int Parse(int argc, char** argv);
    private:
    option* long_opt;
    const char* const short_opt;
    std::string _tensor_X;
    std::string _tensor_Y;
    IType* _cmodes_X, *_cmodes_Y;
    int _num_cmodes;
    int _k;
    int _g;
    int _q;
    int _p;
};

#endif