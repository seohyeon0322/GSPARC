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
#include "cmdline_parser.hpp"

CommandLineParser::CommandLineParser() : short_opt("X:Y:c:x:y:k:g:p:qhv"), _g(1), _p(0), _q(0)
{
    Initialize();
}

CommandLineParser::~CommandLineParser()
{
}

void CommandLineParser::Initialize()
{

    const struct option long_opt[] = {
        {"X", required_argument, 0, 'X'},
        {"Y", required_argument, 0, 'Y'},
        {"contract-mode", required_argument, 0, 'c'},
        {"x", required_argument, 0, 'x'},
        {"y", required_argument, 0, 'y'},
        {"k", required_argument, 0, 'k'},
        {"g", required_argument, 0, 'g'},
        {"p", required_argument, 0, 'p'},
        {"q", 0, NULL, 'q'},
        {"help", 0, NULL, 'h'},
        {NULL, 0, NULL, 0}};
}

int CommandLineParser::Parse(int argc, char **argv)
{
    int c;
    std::cout << "\n\n"
              << std::endl;
    while ((c = getopt_long(argc, argv, this->short_opt, this->long_opt, NULL)) != -1)
    {
        switch (c)
        {
        case 'X':
            std::cout << "tensor X: " << std::string(optarg) << std::endl;
            this->_tensor_X = std::string(optarg);
            break;
        case 'Y':
            std::cout << "tensor Y: " << std::string(optarg) << std::endl;
            this->_tensor_Y = std::string(optarg);
            break;
        case 'c':
            this->_num_cmodes = (IType)atoll(optarg);
            std::cout << "num_cmodes: " << this->_num_cmodes << std::endl;
            this->_cmodes_X = (IType *)AlignedMalloc(this->_num_cmodes * sizeof(IType));
            this->_cmodes_Y = (IType *)AlignedMalloc(this->_num_cmodes * sizeof(IType));
            break;
        case 'x':
            fprintf(stderr, "num_cmodes: %d, cmodes: ", this->_num_cmodes);
            for (IType i = 0; i < this->_num_cmodes; ++i)
            {
                sscanf(argv[optind - 1], "%llu", &(this->_cmodes_X[i]));
                printf("%llu ", this->_cmodes_X[i]);
                fprintf(stderr, "%llu ", this->_cmodes_X[i]);
                ++optind;
            }
            fprintf(stderr, "\n");
            optind -= this->_num_cmodes;
            printf("\n");
            break;
        case 'y':
            fprintf(stderr, "num_cmodes: %d, cmodes: ", this->_num_cmodes);

            for (IType i = 0; i < this->_num_cmodes; ++i)
            {
                sscanf(argv[optind - 1], "%llu", &(this->_cmodes_Y[i]));
                printf("%llu ", this->_cmodes_Y[i]);
                fprintf(stderr, "%llu ", this->_cmodes_X[i]);
                ++optind;
            }
            fprintf(stderr, "\n");

            printf("\n");
            optind -= this->_num_cmodes;
            break;
        case 'k':
            this->_k = (IType)atoll(optarg);
            break;
        case 'g':
            this->_g = (IType)atoll(optarg);
            break;
        case 'q':
            this->_q = true;
            break;
        case 'p':
            this->_p = (IType)atoll(optarg);
            break;
        case 'h':
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -X, --X <int>          Number of rows in X\n");
            printf("  -Y, --Y <int>          Number of rows in Y\n");
            printf("  -c, --contract-mode <int>  Contract mode\n");
            printf("  -x, --x <int>          Number of columns in X\n");
            printf("  -y, --y <int>          Number of columns in Y\n");
            printf("  -h, --help             Display this information\n");
            printf("  -v, --version          Display version information\n");
            exit(0);
        default:
            printf("Unknown option\n");
            exit(1);
        }
    }
    return 0;
}
