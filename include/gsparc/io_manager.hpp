#ifndef IO_MANAGER_HPP_
#define IO_MANAGER_HPP_

namespace gsparc
{

#define IO_MANAGER_TEMPLATE template <typename TensorType>
#define IO_MANAGER_ARGS TensorType

    IO_MANAGER_TEMPLATE
    class IOManager
    {

        using sptensor_t = TensorType;
        using index_t = typename sptensor_t::index_t;
        using value_t = typename sptensor_t::value_t;

    private:
        bool _ReadData(const char *file_path, sptensor_t **tensor, int based);
        bool _ReadQuantumData(const char *file_path, sptensor_t **tensor, int based);
        int get_nmodes(const char *buffer);
        void read_txt_data_parallel(FILE *index_fin, FILE *value_fin, sptensor_t *tensor, int num_modes, index_t nnz);
        void read_txt_dims_parallel(FILE *fin, index_t *num_modes, uint64_t *nnz, index_t **dims);
        void parse_metadata_and_data(const char *buffer, const size_t buffer_length, sptensor_t *tensor, int based);

    public:
        IOManager();
        ~IOManager();

        bool ParseFromFile(const char *file_path, sptensor_t **tensor, int based, bool is_quantum);
    };

} // namespace gsparc
#include "gsparc/io_manager.tpp"
#endif // IO_MANAGER_HPP_