#include <getopt.h>

#include "gsparc/helper.hpp"
#include "gsparc/cmdline_opts.hpp"

namespace gsparc
{
    CommandLineOptions::CommandLineOptions()
        : short_opt("X:Y:c:a:m:x:y:z:n:g:d:qbhv")
    {
        const struct option long_opt[] = {
            {"X", required_argument, 0, 'X'},
            {"Y", required_argument, 0, 'Y'},
            {"contract-mode", required_argument, 0, 'c'},
            {"add-contract-mode", required_argument, 0, 'a'},
            {"multi-sptc", required_argument, 0, 'm'},
            {"x", required_argument, 0, 'x'},
            {"y", required_argument, 0, 'y'},
            {"z", required_argument, 0, 'z'},
            {"n", required_argument, 0, 'n'},
            {"g", required_argument, 0, 'g'},
            {"d", required_argument, 0, 'd'},
            {"q", no_argument, NULL, 'q'},
            {"b", no_argument, NULL, 'b'},
            {"help", 0, NULL, 'h'},
            {NULL, 0, NULL, 0}};

        _tensor_X = "";
        _tensor_Y = "";
        _cmodes_X = nullptr;
        _cmodes_Y = nullptr;
        _num_cmodes = 0;
        _gpu_count = 0;
        _qunatum = 0;
        _num_iter = 1;
        _multi_sptc = 0;
        _dense_accumulator = 0;

        Initialize();
    }

    CommandLineOptions::~CommandLineOptions()
    {
        delete[] long_opt;
    }

    void CommandLineOptions::Initialize()
    {
        this->_gpu_count = 1;   // Default for g flag
        this->_qunatum = false; // Default for q flag
        this->_num_iter = 1;    // Default for n flag
        this->_dense_accumulator = 0; // Default for d flag
        // Allocate default memory for cmodes if needed
        this->_cmodes_X = gsparc::allocate<int>(this->_num_cmodes);
        this->_cmodes_Y = gsparc::allocate<int>(this->_num_cmodes);
        for (int i = 0; i < this->_num_cmodes; ++i)
        {
            this->_cmodes_X[i] = 0; // Default values for cmodes_X
            this->_cmodes_Y[i] = 0; // Default values for cmodes_Y
        }

        const struct option long_opt[] = {
            {"X", required_argument, 0, 'X'},
            {"Y", required_argument, 0, 'Y'},
            {"contract-mode", required_argument, 0, 'c'},
            {"add-contract-mode", required_argument, 0, 'a'},

            {"multi-sptc", required_argument, 0, 'm'},
            {"x", required_argument, 0, 'x'},
            {"y", required_argument, 0, 'y'},
            {"z", required_argument, 0, 'z'},
            {"n", required_argument, 0, 'n'},
            {"g", required_argument, 0, 'g'},
            {"d", required_argument, 0, 'd'},
            {"q", no_argument, NULL, 'q'},
            {"b", no_argument, NULL, 'b'},
            {"help", 0, NULL, 'h'},
            {NULL, 0, NULL, 0}};
    }

    void CommandLineOptions::Parse(int argc, char **argv)
    {
        int c;
        std::cout << "\n\n"
                  << std::endl;
        while ((c = getopt_long(argc, argv, this->short_opt, this->long_opt, NULL)) != -1)
        {
            switch (c)
            {
            case 'X':
                std::cout << "tensor X: " << (optarg) << std::endl;
                fprintf(stderr, "tensor X: %s\n", optarg);
                this->_tensor_X = (optarg);
                break;
            case 'Y':
                std::cout << "tensor Y: " << std::string(optarg) << std::endl;
                fprintf(stderr, "tensor Y: %s\n", optarg);
                this->_tensor_Y = (optarg);
                break;
            case 'c':
                this->_num_cmodes = (int)atoll(optarg);
                std::cout << "# contract modes: " << this->_num_cmodes << std::endl;
                this->_cmodes_X = gsparc::allocate<int>(this->_num_cmodes);
                this->_cmodes_Y = gsparc::allocate<int>(this->_num_cmodes);
                break;
            case 'a':
                this->_num_acmodes = (int)atoll(optarg);
                std::cout << "# contract modes of second SpTC: " << this->_num_acmodes << std::endl;
                this->_cmodes_Z = gsparc::allocate<int>(this->_num_acmodes);
                break;

            case 'x':
                fprintf(stderr, "# contract modes of X: %d, contract modes: ", this->_num_cmodes);
                for (int i = 0; i < this->_num_cmodes; ++i)
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
                fprintf(stderr, "# contract modes of Y: %d, contract modes: ", this->_num_cmodes);

                for (int i = 0; i < this->_num_cmodes; ++i)
                {
                    sscanf(argv[optind - 1], "%llu", &(this->_cmodes_Y[i]));
                    printf("%llu ", this->_cmodes_Y[i]);
                    fprintf(stderr, "%llu ", this->_cmodes_Y[i]);
                    ++optind;
                }
                fprintf(stderr, "\n");

                printf("\n");
                optind -= this->_num_cmodes;
                break;
            case 'z':
                fprintf(stderr, "# contract modes of Z (second SpTC): %d, contract modes: ", this->_num_acmodes);

                for (int i = 0; i < this->_num_acmodes; ++i)
                {
                    sscanf(argv[optind - 1], "%llu", &(this->_cmodes_Z[i]));
                    printf("%llu ", this->_cmodes_Z[i]);
                    fprintf(stderr, "%llu ", this->_cmodes_Z[i]);

                    ++optind;
                }
                fprintf(stderr, "\n");

                printf("\n");
                optind -= this->_num_acmodes;
                break;
            case 'm':
                this->_multi_sptc = (int)atoll(optarg);
                fprintf(stderr, "multi_sptc: %d\n", this->_multi_sptc);
                break;

            case 'n':
                this->_num_iter = (int)atoll(optarg);
                // fprintf(stderr, "num_iter: %d\n", this->_num_iter);
                break;
            case 'g':
                this->_gpu_count = (int)atoll(optarg);
                fprintf(stderr, "num_gpus: %d\n", this->_gpu_count);
                break;
            case 'd':
                this->_dense_accumulator = (int)atoll(optarg);
                fprintf(stderr, "dense_accumulator: %d\n", this->_dense_accumulator);
                break;
            case 'q':
                this->_qunatum = true;
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
        return;
    }
} // namespace gsparc
