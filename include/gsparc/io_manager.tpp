#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <string>
#include <iostream>
#include <omp.h>
#include <vector>

#include "gsparc/helper.hpp"
#include "gsparc/io_manager.hpp"

namespace gsparc
{
    IO_MANAGER_TEMPLATE
    IOManager<IO_MANAGER_ARGS>::IOManager() {}

    IO_MANAGER_TEMPLATE
    IOManager<IO_MANAGER_ARGS>::~IOManager() {}

    IO_MANAGER_TEMPLATE
    bool IOManager<IO_MANAGER_ARGS>::ParseFromFile(const char *file_path, sptensor_t **tensor, int based, bool is_quantum)
    {
        if (is_quantum)
        {
            return _ReadQuantumData(file_path, tensor, based);
        }
        else
        {
            return _ReadData(file_path, tensor, based);
        }
    }

    IO_MANAGER_TEMPLATE
    bool IOManager<IO_MANAGER_ARGS>::_ReadData(const char *file_path, sptensor_t **tensor, int based)
    {

        FILE *fin = fopen(file_path, "rb");
        if (fin == NULL)
        {
            std::string err_msg = "[ERROR] Cannot open file \"" + std::string(file_path) + "\" for reading...";
            throw std::runtime_error(err_msg);
        }
        sptensor_t *_tensor = new sptensor_t((*tensor)->input_number);

        // Get the file size
        fseek(fin, 0, SEEK_END);
        size_t file_size = ftell(fin);
        rewind(fin);
        assert(file_size > 0);

        // Allocate buffer
        char *buffer = gsparc::allocate<char>(file_size);
        assert(buffer != NULL);

        size_t bytes_read = fread(buffer, sizeof(char), file_size, fin);
        assert(bytes_read == file_size);

        fclose(fin);

        // Create temporal tensor

        parse_metadata_and_data(buffer, file_size, _tensor, based);
        *tensor = _tensor;
        // free the buffer
        // buffer.clear();
        free(buffer);

        return true;
    }

    IO_MANAGER_TEMPLATE
    void IOManager<IO_MANAGER_ARGS>::parse_metadata_and_data(const char *buffer, const size_t buffer_length, sptensor_t *sptensor, int based)
    {

        int _nmodes = get_nmodes(buffer);

        int thread_count = omp_get_max_threads();
        std::vector<ssize_t> local_nnzs(thread_count, 0);
        std::vector<std::pair<ssize_t, ssize_t>> buff_ranges(thread_count);

        // get the number of non-zeros
        uint64_t _nnzs = 0;
#pragma omp parallel
        {
            // Multi-threaded processing with OpenMP
            int tid = omp_get_thread_num();

            // Divide the buffer among thread by assigning chunks
            ssize_t chunk_size = buffer_length / thread_count;
            ssize_t start = tid * chunk_size;
            ssize_t end = (tid == thread_count - 1) ? buffer_length : start + chunk_size;

            // Adjust the start and end positions to the nearest newline character
            if (tid != 0)
            {
                while (start < buffer_length && buffer[start] != '\n')
                {
                    ++start;
                }
                ++start; // move to the next character
            }

            // Adjust end to the first newline after chunk ends
            if (tid != thread_count - 1)
            {
                while (end < buffer_length && buffer[end] != '\n')
                {
                    ++end;
                }
                ++end; // move to the next character
            }

            // Store the start and end positions of the chunk
            buff_ranges[tid] = std::make_pair(start, end);

            // Process the buffer for each thread
            for (ssize_t i = start; i < end; ++i)
            {
                if (buffer[i] == '\n')
                {
                    ++local_nnzs[tid];
                }
            }
#pragma omp atomic
            _nnzs += local_nnzs[tid];

        } // end of parallel region

        sptensor->order = _nmodes;
        sptensor->nnz = _nnzs;

        std::vector<ssize_t> nnz_offsets(thread_count, 0); // start position of each thread
        for (int tid = 1; tid < thread_count; ++tid)
        {
            nnz_offsets[tid] = nnz_offsets[tid - 1] + local_nnzs[tid - 1];
        }
        assert(nnz_offsets[thread_count - 1] + local_nnzs[thread_count - 1] == _nnzs);

        sptensor->dims = gsparc::allocate<index_t>(_nmodes);
        assert(sptensor->dims != NULL);
        for (index_t i = 0; i < _nmodes; ++i)
        {
            sptensor->dims[i] = 0;
        }

        sptensor->indices = gsparc::allocate<index_t *>(_nmodes);
        assert(sptensor->indices != NULL);
        for (index_t i = 0; i < _nmodes; ++i)
        {
            sptensor->indices[i] = gsparc::allocate<index_t>(_nnzs);
            assert(sptensor->indices[i] != NULL);
        }

        sptensor->values = gsparc::allocate<value_t>(_nnzs);
        assert(sptensor->values != NULL);

        // read the data
        std::vector<index_t> *local_dims = new std::vector<index_t>[thread_count];
        double _norm = 0.0;
#pragma omp parallel reduction(+ : _norm)
        {
            int tid = omp_get_thread_num();

            // Initialize the local dimensions
            local_dims[tid].resize(_nmodes, std::numeric_limits<index_t>::min());

            // Process the buffer for each thread
            size_t start = buff_ranges[tid].first;
            size_t end = buff_ranges[tid].second;

            size_t buff_ptr = start;

            while (buff_ptr < end)
            {

                char *line = const_cast<char *>(&buffer[buff_ptr]);

                // Find the end of the current line
                char *line_end = strchr(line, '\n');
                if (!line_end)
                {
                    line_end = const_cast<char *>(&buffer[buffer_length]);
                    buff_ptr = end;
                }
                else
                {
                    buff_ptr = line_end - buffer + 1; // move to the next character
                }

                // Temporal buffer for the current line
                char saved_char = *line_end;
                *line_end = '\0';

                // Tokenize the buffer by newline characters
                char *rest_token = nullptr;
                char *token = strtok_r(line, " \t", &rest_token);

                if (token != NULL)
                {
                    size_t offset = nnz_offsets[tid];
                    assert(offset < _nnzs);

                    int axis = 0;
                    // Process each token separated by space
                    while (token != NULL && (axis < _nmodes))
                    {
                        index_t idx = strtoull(token, NULL, 10) - based; // 0-based indexing

                        // Update the maximum and minimum indices for the current axis
                        local_dims[tid][axis] = std::max<index_t>(local_dims[tid][axis], idx + 1);
                        // Store the current index in the global indices array
                        sptensor->indices[axis][offset] = idx;

                        ++axis;
                        token = strtok_r(NULL, " \t", &rest_token);
                    }

                    // Ensure there is a valid token left for the value
                    if (token != NULL && *token != '\0')
                    {
                        try
                        {
                            value_t val = (value_t)std::stod(token);
                            _norm += val * val;
                            sptensor->values[offset] = val;
                        }
                        catch (const std::invalid_argument &e)
                        {
                            std::cerr << "Error: Invalid value found at line " << nnz_offsets[tid] << "\n";
                            std::cerr << "Token: " << token << "\n";
                            throw;
                        }
                    }
                    else
                    {
                        std::cerr << "Error: Missing value at line " << nnz_offsets[tid] << "\n";
                        throw std::logic_error("Missing value in input data");
                    }
                    nnz_offsets[tid]++;
                }
                // Restore the saved character
                *line_end = saved_char;
            }
        }

        // Update the dimensions
        for (index_t i = 0; i < _nmodes; ++i)
        {
            for (int tid = 0; tid < thread_count; ++tid)
            {
                sptensor->dims[i] = std::max(sptensor->dims[i], local_dims[tid][i]);
            }
        }

        delete[] local_dims;
    }

    /**
     * @brief Get the number of modes
     * @param buffer buffer containing the tensor data
     * @return the number of modes
     */
    IO_MANAGER_TEMPLATE
    int IOManager<IO_MANAGER_ARGS>::get_nmodes(const char *buffer)
    {
        // get the number of modes
        const char *line_start = buffer;                 // Pointer to the start of the buffer
        const char *line_end = strchr(line_start, '\n'); // Find the first newline

        if (line_end == nullptr)
        {
            line_end = line_start + strlen(line_start); // Handle the case where there's no newline
        }

        // Count the number of tokens (separated by space or tab)
        size_t tokenCount = 0;
        const char *token_start = line_start;

        // Iterate through the first line to count tokens
        while (token_start < line_end)
        {
            // Skip leading spaces or tabs
            while (token_start < line_end && (*token_start == ' ' || *token_start == '\t'))
            {
                ++token_start;
            }

            // Find the end of the token (space or tab)
            const char *token_end = token_start;
            while (token_end < line_end && *token_end != ' ' && *token_end != '\t')
            {
                ++token_end;
            }

            // If a valid token is found, count it
            if (token_start < token_end)
            {
                ++tokenCount;
                token_start = token_end;
            }
        }

        // exclude the last token which is the nnz (last token = value)
        return tokenCount - 1;
    }

    IO_MANAGER_TEMPLATE
    bool IOManager<IO_MANAGER_ARGS>::_ReadQuantumData(const char *file_path, sptensor_t **tensor, int based)
    {
        char index_file[256], value_file[256];
        strcpy(index_file, file_path);
        strcat(index_file, "/indices.txt");
        strcpy(value_file, file_path);
        strcat(value_file, "/values.txt");

        FILE *index_fp = fopen(index_file, "r");
        FILE *value_fp = fopen(value_file, "r");
        assert(index_fp && value_fp);

        sptensor_t *_tensor = new sptensor_t((*tensor)->input_number);

        index_t nnz = 0, nmodes = 0;
        index_t *dims = nullptr;

        read_txt_dims_parallel(index_fp, &nmodes, &nnz, &dims);

        _tensor->dims = gsparc::allocate<index_t>(nmodes);
        memcpy(_tensor->dims, dims, sizeof(index_t) * nmodes);
        _tensor->order = nmodes;
        _tensor->nnz = nnz;
        read_txt_data_parallel(index_fp, value_fp, _tensor, nmodes, nnz);

        *tensor = _tensor;  
        free(dims);

        fclose(index_fp);
        fclose(value_fp);
        return true;
    }

    IO_MANAGER_TEMPLATE
    void IOManager<IO_MANAGER_ARGS>::read_txt_dims_parallel(FILE *fin, index_t *num_modes, uint64_t *nnz, index_t **dims)
    {
        // **파일 크기 확인 및 읽기**
        fseek(fin, 0, SEEK_END);
        size_t file_size = ftell(fin);
        rewind(fin);

        char *buffer = gsparc::allocate<char>(file_size);
        assert(buffer);
        fread(buffer, sizeof(char), file_size, fin);

        // **Step 1: `nmodes`는 `indices.txt`의 행 개수 (즉, 줄 개수)**
        index_t nmodes_count = 0;
        char *ptr = buffer;
        while (*ptr)
        {
            if (*ptr == '\n')
                ++nmodes_count;
            ++ptr;
        }

        *num_modes = nmodes_count; // 최종 nmodes 저장
        char **lines = gsparc::allocate<char *>(nmodes_count);
        // **Step 2: `nnz`는 `indices.txt`의 첫 번째 행의 요소 개수**
        char *saveptr;
        char *line = strtok_r(buffer, "\n", &saveptr);
        for (index_t i = 0; i < nmodes_count; ++i)
        {
            if (!line)
                break;
            lines[i] = line;
            line = strtok_r(NULL, "\n", &saveptr);
        }

        // **Step 3: 각 mode(행)의 최대 값 찾기**
        *dims = gsparc::allocate<index_t>(*num_modes);
        assert(*dims);
        memset(*dims, 0, sizeof(index_t) * (*num_modes));

        // **Thread 개수 제한 설정**
        int max_threads = omp_get_max_threads();
        int thread_count = (*num_modes < max_threads) ? *num_modes : max_threads;
        // **Thread 별 로컬 최대값 저장**
        std::vector<std::vector<index_t>> local_dims(thread_count, std::vector<index_t>(*num_modes, 0));

#pragma omp parallel for
        for (index_t row = 0; row < *num_modes; ++row)
        {
            index_t nnz_count = 0;
            int tid = omp_get_thread_num();

            char *line = lines[row];
            char *rest_token = nullptr;
            char *token = strtok_r(line, " \t", &rest_token);
            while (token != NULL)
            {
                // index_t index = static_cast<index_t>(strtol(token, nullptr, 10));
                index_t index = static_cast<index_t>(strtod(token, &token));

                // printf("index: %llu\n", index);
                local_dims[tid][row] = std::max(local_dims[tid][row], index);
                token = strtok_r(nullptr, " \t", &rest_token);
                ++nnz_count;
                // printf('second token: %s\n', token);
            }

            // **Thread 별 최댓값을 `dims`에 병합**
#pragma omp critical
            {
                for (index_t i = 0; i < *num_modes; ++i)
                {
                    (*dims)[i] = std::max((*dims)[i], local_dims[tid][i]);
                }
                *nnz = nnz_count;
            }
        }
        free(buffer);
    }

    IO_MANAGER_TEMPLATE
    void IOManager<IO_MANAGER_ARGS>::read_txt_data_parallel(FILE *index_fin, FILE *value_fin, sptensor_t *tensor, int num_modes, index_t nnz)
    {
        fseek(index_fin, 0, SEEK_END);
        size_t file_size = ftell(index_fin);
        rewind(index_fin);

        char *buffer = gsparc::allocate<char>(file_size);
        assert(buffer);
        fread(buffer, sizeof(char), file_size, index_fin);

        tensor->indices = gsparc::allocate<index_t *>(num_modes);
        assert(tensor->indices);

        for (index_t i = 0; i < num_modes; i++)
        {
            tensor->indices[i] = gsparc::allocate<index_t>(nnz);
            assert(tensor->indices[i]);
        }

        tensor->values = gsparc::allocate<value_t>(nnz);
        assert(tensor->values);

        char **lines = gsparc::allocate<char *>(num_modes);
        // **Step 2: `nnz`는 `indices.txt`의 첫 번째 행의 요소 개수**
        char *saveptr;
        char *line = strtok_r(buffer, "\n", &saveptr);
        for (index_t i = 0; i < num_modes; ++i)
        {
            if (!line)
                break;
            lines[i] = line;
            line = strtok_r(NULL, "\n", &saveptr);
        }
        // **각 Thread가 "행 단위"로 읽고, "열 단위"로 저장**
#pragma omp parallel for
        for (index_t row = 0; row < num_modes; ++row)
        {
            char *line = lines[row];
            char *rest_buffer = nullptr;
            char *token = strtok_r(line, " \t", &rest_buffer);
            index_t col_idx = 0;
            while (token && col_idx < nnz)
            {
                index_t index = static_cast<index_t>(strtod(token, &token));

                tensor->indices[row][col_idx] = index - 1; //
                token = strtok_r(nullptr, " \t", &rest_buffer);
                ++col_idx;
            }
        }

        // **values.txt 처리 (변경 없음)**
        rewind(value_fin);
        for (index_t i = 0; i < nnz; i++)
        {
            fscanf(value_fin, "%le", &tensor->values[i]);
        }

        free(buffer);
    }

} // namespace gsparc