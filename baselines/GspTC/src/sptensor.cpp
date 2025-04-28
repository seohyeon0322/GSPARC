#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <string>
#include <iostream>
#include <vector>
#include "sptensor.hpp"
#include "sort.hpp"
#include "partition.hpp"
#include "pinned_memory.cuh"

void PrintTensorInfo(SparseTensor *X)
{
  IType *dims = X->orig_dims;
  IType nnz = X->nnz;
  int nmodes = X->nmodes;

  IType tmp = 1;
  for (int i = 0; i < nmodes; i++)
  {
    tmp *= dims[i];
  }
  double sparsity = ((double)nnz) / tmp;
  fprintf(stderr, "# Modes         = %u\n", nmodes);
  fprintf(stderr, "Sparsity        = %f\n", sparsity);
  fprintf(stderr, "Dimensions      = [%llu", dims[0]);
  for (int i = 1; i < nmodes; i++)
  {
    fprintf(stderr, " X %llu", dims[i]);
  }
  fprintf(stderr, "]\n");
  fprintf(stderr, "NNZ             = %llu\n", nnz);
}

void DestroySparseTensor(SparseTensor *X)
{
  AlignedFree(X->orig_dims);
  AlignedFree(X->vals);
  for (int i = 0; i < X->nmodes; i++)
  {
    AlignedFree(X->orig_cidx[i]);
  }
  AlignedFree(X->orig_cidx);
  AlignedFree(X);
}

void ExportSparseTensor(
    const char *file_path,
    FileFormat f,
    SparseTensor *X)
{
  assert(f == TEXT_FORMAT || f == BINARY_FORMAT);

  IType nmodes = X->nmodes;
  IType nnz = X->nnz;
  char fn[1024];
  FILE *fp = NULL;

  if (file_path == NULL)
  {
    if (f == TEXT_FORMAT)
    {
      sprintf(fn, "%lluD_%llu.tns", nmodes, nnz);
    }
    else if (f == BINARY_FORMAT)
    {
      sprintf(fn, "%lluD_%llu.bin", nmodes, nnz);
    }
    fp = fopen(fn, "w");
    assert(fp != NULL);
  }
  else
  {
    fp = fopen(file_path, "w");
    assert(fp != NULL);
  }

  if (f == TEXT_FORMAT)
  {
    IType **orig_cidx = X->orig_cidx;
    FType *vals = X->vals;
    for (IType i = 0; i < nnz; i++)
    {
      for (IType n = 0; n < nmodes; n++)
      {
        fprintf(fp, "%llu ", (orig_cidx[n][i] + 1));
      }
      fprintf(fp, "%g\n", vals[i]);
    }
  }
  else if (f == BINARY_FORMAT)
  {
    // first write the number of modes
    fwrite(&(X->nmodes), sizeof(IType), 1, fp);
    // then, the dimensions
    fwrite(X->orig_dims, sizeof(IType), X->nmodes, fp);
    fwrite(&(X->nnz), sizeof(IType), 1, fp);

    // write the indices and the values
    for (int i = 0; i < X->nmodes; i++)
    {
      fwrite(X->orig_cidx[i], sizeof(IType), X->nnz, fp);
    }
    fwrite(X->vals, sizeof(FType), X->nnz, fp);
  }

  fclose(fp);
}

void read_tns_dims(
    FILE *fin,
    IType *num_modes,
    IType *nnz,
    IType **dims)
{
  // count the number of modes
  IType nmodes = 0;
  ssize_t nread;
  char *line = NULL;
  size_t len = 0;
  while ((nread = getline(&line, &len, fin)) != -1)
  {
    // if line is not empty or a comment (#)
    if (nread > 1 && line[0] != '#')
    {
      char *ptr = strtok(line, " \t");
      while (ptr != NULL)
      {
        ++nmodes;
        ptr = strtok(NULL, " \t");
      }
      break;
    }
  }
  --nmodes;
  *num_modes = nmodes;

  // Calculate the tensor dimensions
  IType *tmp_dims = (IType *)malloc(sizeof(IType) * nmodes);
  assert(tmp_dims);
  for (IType i = 0; i < nmodes; i++)
  {
    tmp_dims[i] = 0;
  }
  IType tmp_nnz = 0;

  rewind(fin);
  while ((nread = getline(&line, &len, fin)) != -1)
  {
    // if line is not empty or a comment
    if (nread > 1 && line[0] != '#')
    {
      char *ptr = line;
      for (IType i = 0; i < nmodes; i++)
      {
        IType index = strtoull(ptr, &ptr, 10);
        if (index > tmp_dims[i])
        {
          tmp_dims[i] = index;
        }
      }
      strtod(ptr, &ptr);
      ++tmp_nnz;
    }
  }
  *dims = tmp_dims;
  *nnz = tmp_nnz;

  rewind(fin);
  free(line);
}

void read_tns_data(
    FILE *fin,
    SparseTensor *tensor,
    IType num_modes,
    IType nnz)
{
  IType tmp_nnz = 0;
  ssize_t nread;
  char *line = NULL;
  size_t len = 0;

  while ((nread = getline(&line, &len, fin)) != -1)
  {
    // if line is not empty or a comment
    if (nread > 1 && line[0] != '#')
    {
      char *ptr = line;
      for (IType i = 0; i < num_modes; i++)
      {
        tensor->orig_cidx[i][tmp_nnz] = (IType)strtoull(ptr, &ptr, 10) - 1;
      }
      tensor->vals[tmp_nnz] = (FType)strtod(ptr, &ptr);
      ++tmp_nnz;
    }
  }
  assert(tmp_nnz == nnz);

  free(line);
}

void read_txt_dims(
    FILE *fin,
    IType *num_modes,
    IType *nnz,
    IType **dims)
{
  IType nmodes = 0;
  ssize_t nread;
  char *line = NULL;
  size_t len = 0;

  // 첫 번째 유효한 라인을 읽어서 차원 수 계산
  while ((nread = getline(&line, &len, fin)) != -1)
  {
    if (nread > 1 && line[0] != '#')
    {
      nmodes++;
    }
  }

  *num_modes = nmodes;

  // 텐서의 차원 크기를 계산
  IType *tmp_dims = (IType *)malloc(sizeof(IType) * nmodes);
  assert(tmp_dims);
  for (IType i = 0; i < nmodes; i++)
  {
    tmp_dims[i] = 0;
  }

  rewind(fin);
  IType tmp_nnz = 0;
  IType line_num = 0;

  while ((nread = getline(&line, &len, fin)) != -1)
  {
    if (nread > 1 && line[0] != '#')
    {
      char *ptr = strtok(line, " \t");
      IType index_count = 0;
      while (ptr != NULL)
      {
        IType index = static_cast<IType>(strtod(ptr, &ptr));
        if (index > tmp_dims[line_num])
        {
          tmp_dims[line_num] = index;
        }
        if (line_num == 0)
        {
          // 첫 번째 줄에서 nnz를 결정
          tmp_nnz++;
        }
        ptr = strtok(NULL, " \t");
        index_count++;
      }
      line_num++;
    }
  }

  *dims = tmp_dims;
  *nnz = tmp_nnz;

  rewind(fin);
  free(line);
}

void read_txt_data(
    FILE *index_fin,
    FILE *value_fin,
    SparseTensor *tensor,
    IType num_modes,
    IType nnz)
{
  IType tmp_nnz = 0;
  ssize_t nread;
  char *line = NULL;
  size_t len = 0;

  rewind(index_fin);
  IType line_num = 0;

  while ((nread = getline(&line, &len, index_fin)) != -1)
  {
    if (nread > 1 && line[0] != '#')
    {
      char *ptr = strtok(line, " \t");
      IType index_count = 0;
      while (ptr != NULL)
      {
        double index = strtod(ptr, &ptr);
        tensor->orig_cidx[line_num][index_count] = (IType)index - 1; // 1-based indexing assumed in input
        ptr = strtok(NULL, " \t");
        index_count++;
      }
      line_num++;
    }
  }

  // value.txt를 처음부터 읽음
  rewind(value_fin);
  for (IType i = 0; i < nnz; i++)
  {
    fscanf(value_fin, "%le", &tensor->vals[i]);
  }

  free(line);
}

void read_txt_dims_parallel(FILE *fin, IType *num_modes, IType *nnz, IType **dims)
{
  // **파일 크기 확인 및 읽기**
  fseek(fin, 0, SEEK_END);
  size_t file_size = ftell(fin);
  rewind(fin);

  char *buffer = (char *)malloc(file_size);
  assert(buffer);
  fread(buffer, sizeof(char), file_size, fin);

  // **Step 1: `nmodes`는 `indices.txt`의 행 개수 (즉, 줄 개수)**
  IType nmodes_count = 0;
  char *ptr = buffer;
  while (*ptr)
  {
    if (*ptr == '\n')
      ++nmodes_count;
    ++ptr;
  }

  *num_modes = nmodes_count; // 최종 nmodes 저장
  char **lines = (char **)malloc(sizeof(char *) * nmodes_count);
  // **Step 2: `nnz`는 `indices.txt`의 첫 번째 행의 요소 개수**
  char *saveptr;
  char *line = strtok_r(buffer, "\n", &saveptr);
  for (IType i = 0; i < nmodes_count; ++i)
  {
    if (!line)
      break;
    lines[i] = line;
    line = strtok_r(NULL, "\n", &saveptr);
  }

  // **Step 3: 각 mode(행)의 최대 값 찾기**
  *dims = (IType *)malloc(sizeof(IType) * (*num_modes));
  assert(*dims);
  memset(*dims, 0, sizeof(IType) * (*num_modes));

  // **Thread 개수 제한 설정**
  int max_threads = omp_get_max_threads();
  int thread_count = (*num_modes < max_threads) ? *num_modes : max_threads;
  // **Thread 별 로컬 최대값 저장**
  std::vector<std::vector<IType>> local_dims(thread_count, std::vector<IType>(*num_modes, 0));

#pragma omp parallel for
  for (IType row = 0; row < *num_modes; ++row)
  {
    IType nnz_count = 0;
    int tid = omp_get_thread_num();

    char *line = lines[row];
    char *rest_token = nullptr;
    char *token = strtok_r(line, " \t", &rest_token);
    while (token != NULL)
    {
      // IType index = static_cast<IType>(strtol(token, nullptr, 10));
      IType index = static_cast<IType>(strtod(token, &token));

      // printf("index: %llu\n", index);
      local_dims[tid][row] = std::max(local_dims[tid][row], index);
      token = strtok_r(nullptr, " \t", &rest_token);
      ++nnz_count;
      // printf('second token: %s\n', token);
    }

    // **Thread 별 최댓값을 `dims`에 병합**
#pragma omp critical
    {
      for (IType i = 0; i < *num_modes; ++i)
      {
        (*dims)[i] = std::max((*dims)[i], local_dims[tid][i]);
      }
      *nnz = nnz_count;
    }
  }
  free(buffer);
}

void read_txt_data_parallel(FILE *index_fin, FILE *value_fin, SparseTensor *tensor, IType num_modes, IType nnz, PinnedMemoryPool *pinned_pool)
{
  fseek(index_fin, 0, SEEK_END);
  size_t file_size = ftell(index_fin);
  rewind(index_fin);

  char *buffer = (char *)malloc(file_size);
  assert(buffer);
  fread(buffer, sizeof(char), file_size, index_fin);

  tensor->orig_cidx = (IType **)malloc(sizeof(IType *) * num_modes);
  assert(tensor->orig_cidx);

  for (IType i = 0; i < num_modes; i++)
  {
    tensor->orig_cidx[i] = (IType *)malloc(sizeof(IType) * nnz);
    assert(tensor->orig_cidx[i]);
  }

  tensor->orig_vals = (FType *)malloc(sizeof(FType) * nnz);
  assert(tensor->orig_vals);

  char **lines = (char **)malloc(sizeof(char *) * num_modes);
  // **Step 2: `nnz`는 `indices.txt`의 첫 번째 행의 요소 개수**
  char *saveptr;
  char *line = strtok_r(buffer, "\n", &saveptr);
  for (IType i = 0; i < num_modes; ++i)
  {
    if (!line)
      break;
    lines[i] = line;
    line = strtok_r(NULL, "\n", &saveptr);
  }
  // **각 Thread가 "행 단위"로 읽고, "열 단위"로 저장**
#pragma omp parallel for
  for (IType row = 0; row < num_modes; ++row)
  {
    char *line = lines[row];
    char *rest_buffer = nullptr;
    char *token = strtok_r(line, " \t", &rest_buffer);
    IType col_idx = 0;
    while (token && col_idx < nnz)
    {
      IType index = static_cast<IType>(strtod(token, &token));

      tensor->orig_cidx[row][col_idx] = index - 1; //
      token = strtok_r(nullptr, " \t", &rest_buffer);
      ++col_idx;
    }
  }

  // **values.txt 처리 (변경 없음)**
  rewind(value_fin);
  for (IType i = 0; i < nnz; i++)
  {
    fscanf(value_fin, "%le", &tensor->orig_vals[i]);
  }

  free(buffer);
}

void ImportQuantumTensor(const char *filepath, SparseTensor **X_, PinnedMemoryPool *pinned_pool)
{
  char index_file[256], value_file[256];
  strcpy(index_file, filepath);
  strcat(index_file, "/indices.txt");
  strcpy(value_file, filepath);
  strcat(value_file, "/values.txt");

  FILE *index_fp = fopen(index_file, "r");
  FILE *value_fp = fopen(value_file, "r");
  assert(index_fp && value_fp);

  IType nnz = 0, nmodes = 0;
  IType *dims = nullptr;

  read_txt_dims_parallel(index_fp, &nmodes, &nnz, &dims);

  SparseTensor *ten = (SparseTensor *)pinned_pool->allocate(sizeof(SparseTensor));
  assert(ten);

  ten->orig_dims = (IType *)pinned_pool->allocate(sizeof(IType) * nmodes);
  memcpy(ten->orig_dims, dims, sizeof(IType) * nmodes);
  ten->nmodes = nmodes;
  ten->nnz = nnz;
  read_txt_data_parallel(index_fp, value_fp, ten, nmodes, nnz, pinned_pool);

  *X_ = ten;
  free(dims);

  fclose(index_fp);
  fclose(value_fp);
}

void read_sparse_tensor(const char *file_path, SparseTensor **sptensor, int based, PinnedMemoryPool *pinned_pool)
{
  // @note: input tensors must follow base-1 indexing and outputs are based on base-0 ndexing.

  FILE *fin = fopen(file_path, "rb");
  if (fin == NULL)
  {
    std::string err_msg = "[ERROR] Cannot open file \"" + std::string(file_path) + "\" for reading...";
    throw std::runtime_error(err_msg);
  }

  // Get the file size
  fseek(fin, 0, SEEK_END);
  size_t file_size = ftell(fin);
  rewind(fin);
  assert(file_size > 0);

  // Allocate buffer
  char *buffer = (char *)malloc(file_size * sizeof(char));
  assert(buffer != NULL);

  size_t bytes_read = fread(buffer, sizeof(char), file_size, fin);
  assert(bytes_read == file_size);

  fclose(fin);

  // Create temporal tensor
  SparseTensor *_sptensor = (SparseTensor *)malloc(sizeof(SparseTensor));
  assert(_sptensor != NULL);

  parse_metadata_and_data(buffer, file_size, _sptensor, based, pinned_pool);

  *sptensor = _sptensor;

  // free the buffer
  // buffer.clear();
  free(buffer);
}

/**
 * @brief Parse metadata and data
 * @param buffer buffer containing the tensor data
 * @param buffer_length length of the buffer
 * @param sptensor SparseTensor object
 */
void parse_metadata_and_data(const char *buffer, const size_t buffer_length, SparseTensor *sptensor, int based, PinnedMemoryPool *pinned_pool)
{

  int _nmodes = get_nmodes(buffer);

  int thread_count = omp_get_max_threads();
  std::vector<ssize_t> local_nnzs(thread_count, 0);
  std::vector<std::pair<ssize_t, ssize_t>> buff_ranges(thread_count);

  // get the number of non-zeros
  IType _nnzs = 0;
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

  sptensor->nmodes = _nmodes;
  sptensor->nnz = _nnzs;

  std::vector<ssize_t> nnz_offsets(thread_count, 0); // start position of each thread
  for (int tid = 1; tid < thread_count; ++tid)
  {
    nnz_offsets[tid] = nnz_offsets[tid - 1] + local_nnzs[tid - 1];
  }
  assert(nnz_offsets[thread_count - 1] + local_nnzs[thread_count - 1] == _nnzs);

  sptensor->orig_dims = (IType *)malloc(_nmodes * sizeof(IType));
  assert(sptensor->orig_dims != NULL);
  for (IType i = 0; i < _nmodes; ++i)
  {
    sptensor->orig_dims[i] = 0;
  }

  sptensor->orig_cidx = (IType **)malloc(_nmodes * sizeof(IType *));
  assert(sptensor->orig_cidx != NULL);
  for (IType i = 0; i < _nmodes; ++i)
  {
    sptensor->orig_cidx[i] = (IType *)malloc(_nnzs * sizeof(IType));
    assert(sptensor->orig_cidx[i] != NULL);
  }

  sptensor->orig_vals = (FType *)malloc(_nnzs * sizeof(FType));
  // cudaHostAlloc((void **)&sptensor->vals, _nnzs * sizeof(FType), cudaHostAllocPortable);
  // assert(sptensor->vals != NULL);

  // read the data
  std::vector<IType> *local_dims = new std::vector<IType>[thread_count];
  double _norm = 0.0;
#pragma omp parallel reduction(+ : _norm)
  {
    int tid = omp_get_thread_num();

    // Initialize the local dimensions
    local_dims[tid].resize(_nmodes, std::numeric_limits<IType>::min());

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
          IType idx = strtoull(token, NULL, 10) - based; // 0-based indexing

          // Update the maximum and minimum indices for the current axis
          local_dims[tid][axis] = std::max<IType>(local_dims[tid][axis], idx + 1);
          // Store the current index in the global indices array
          sptensor->orig_cidx[axis][offset] = idx;

          ++axis;
          token = strtok_r(NULL, " \t", &rest_token);
        }

        // Ensure there is a valid token left for the value
        if (token != NULL && *token != '\0')
        {
          try
          {
            FType val = (FType)std::stod(token);
            _norm += val * val;
            sptensor->orig_vals[offset] = val;
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
  for (IType i = 0; i < _nmodes; ++i)
  {
    for (int tid = 0; tid < thread_count; ++tid)
    {
      sptensor->orig_dims[i] = std::max(sptensor->orig_dims[i], local_dims[tid][i]);
    }
  }

  delete[] local_dims;
}
/**
 * @brief Get the number of modes
 * @param buffer buffer containing the tensor data
 * @return the number of modes
 */
int get_nmodes(const char *buffer)
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

// void ImportSparseTensor(
//     const char *file_path,
//     FileFormat f,
//     SparseTensor **X_)
// {
//   FILE *fp = fopen(file_path, "r");
//   assert(fp != NULL);

//   IType nnz = 0;
//   IType nmodes = 0;
//   IType *dims = NULL;
//   // FType* vals = NULL;
//   // IType** cidx = NULL;

//   if (f == TEXT_FORMAT)
//   {
//     // read dims and nnz info from file
//     read_tns_dims(fp, &nmodes, &nnz, &dims);

//     // allocate memory to the data structures
//     SparseTensor *ten = (SparseTensor *)AlignedMalloc(sizeof(SparseTensor));
//     assert(ten);

//     ten->dims = (IType *)AlignedMalloc(sizeof(IType) * nmodes);
//     assert(ten->dims);
//     ten->cidx = (IType **)AlignedMalloc(sizeof(IType *) * nmodes);
//     assert(ten->cidx);
//     for (IType i = 0; i < nmodes; i++)
//     {
//       ten->cidx[i] = (IType *)AlignedMalloc(sizeof(IType) * nnz);
//       assert(ten->cidx[i]);
//     }
//     ten->vals = (FType *)AlignedMalloc(sizeof(FType) * nnz);
//     assert(ten->vals);

//     // populate the data structures
//     ten->nmodes = nmodes;
//     for (IType i = 0; i < nmodes; i++)
//     {
//       ten->dims[i] = dims[i];
//     }
//     ten->nnz = nnz;
//     read_tns_data(fp, ten, nmodes, nnz);

//     *X_ = ten;
//   }
//   else if (f == BINARY_FORMAT)
//   {
//     // first read the number of modes
//     nmodes = 0;
//     fread(&nmodes, sizeof(IType), 1, fp);

//     // use this information to read the dimensions of the tensor
//     IType *dims = (IType *)AlignedMalloc(sizeof(IType) * nmodes);
//     assert(dims);
//     fread(dims, sizeof(IType), nmodes, fp);
//     // read the nnz
//     nnz = 0;
//     fread(&nnz, sizeof(IType), 1, fp);
//     // use this information to read the index and the values
//     IType **cidx = (IType **)AlignedMalloc(sizeof(IType *) * nmodes);
//     assert(cidx);
//     for (IType i = 0; i < nmodes; i++)
//     {
//       cidx[i] = (IType *)AlignedMalloc(sizeof(IType) * nnz);
//       assert(cidx[i]);
//       fread(cidx[i], sizeof(IType), nnz, fp);
//     }
//     FType *vals = (FType *)malloc(sizeof(FType) * nnz);
//     assert(vals);
//     fread(vals, sizeof(FType), nnz, fp);

//     // create the sptensor
//     SparseTensor *ten = (SparseTensor *)AlignedMalloc(sizeof(SparseTensor));
//     assert(ten);
//     ten->nmodes = nmodes;
//     ten->dims = dims;
//     ten->nnz = nnz;
//     ten->cidx = cidx;
//     ten->vals = vals;

//     *X_ = ten;
//   }

//   if (dims)
//   {
//     free(dims);
//   }

//   fclose(fp);
// }

// void CreateSparseTensor(
//     IType nmodes,
//     IType *dims,
//     IType nnz,
//     IType *cidx,
//     FType *vals,
//     SparseTensor **X_)
// {
//   assert(nmodes > 0);
//   assert(nnz > 0);
//   for (IType n = 0; n < nmodes; n++)
//   {
//     assert(dims[n] > 0);
//   }
//   for (IType n = 0; n < nmodes; n++)
//   {
//     for (IType i = 0; i < nnz; i++)
//     {
//       assert(cidx[i * nmodes + n] < dims[n]);
//     }
//   }

//   // create tensor
//   SparseTensor *X = (SparseTensor *)AlignedMalloc(sizeof(SparseTensor));
//   assert(X);
//   X->nmodes = nmodes;
//   X->nnz = nnz;
//   X->dims = (IType *)AlignedMalloc(sizeof(IType) * nmodes);
//   assert(X->dims);
//   memcpy(X->dims, dims, sizeof(IType) * nmodes);

//   X->cidx = (IType **)AlignedMalloc(sizeof(IType *) * nmodes);
//   assert(X->cidx);
//   for (IType n = 0; n < nmodes; n++)
//   {
//     X->cidx[n] = (IType *)AlignedMalloc(sizeof(IType) * nnz);
//     assert(X->cidx[n]);
//   }
//   for (IType i = 0; i < nnz; i++)
//   {
//     for (IType n = 0; n < nmodes; n++)
//     {
//       X->cidx[n][i] = cidx[i * nmodes + n];
//     }
//   }

//   X->vals = (FType *)AlignedMalloc(sizeof(FType) * nnz);
//   assert(X->vals);
//   memcpy(X->vals, vals, sizeof(FType) * nnz);

//   *X_ = X;
// }

void permute_tensor(SparseTensor *X, IType *mode_order_X)
{

  IType **temp_ind = (IType **)malloc(X->nmodes * sizeof(IType *));
  IType *temp_ndims = (IType *)malloc(X->nmodes * sizeof(IType));
  X->mode = (IType *)malloc(X->nmodes * sizeof(IType));
  for (IType m = 0; m < X->nmodes; m++)
  {
    temp_ind[m] = X->orig_cidx[m];
    temp_ndims[m] = X->orig_dims[m];
  }
  for (IType m = 0; m < X->nmodes; ++m)
  {
    IType pm = mode_order_X[m];

    X->dims[m] = temp_ndims[pm];
    X->orig_cidx[m] = temp_ind[pm];
    X->mode[m] = pm;
  }
  for (int m = 0; m < X->fnmodes; m++)
  {
    X->fmode[m] = X->cnmodes + m;
  }
  free(temp_ind);
  free(temp_ndims);
}

int compare_cmodes(IType cmode_len, IType *tmp_ind, IType **X_ind, IType idx)
{
  for (int i = 0; i < cmode_len; i++)
  { 
    if (tmp_ind[i] > X_ind[i][idx])
    {
      return 1;
    }
    else if (tmp_ind[i] < X_ind[i][idx])
    {
      return -1;
    }
  }
  return 0;
}

int compare_cmodes_tmp(IType cmode_len, IType *tmp_ind, IType *X_ind)
{
  for (int i = 0; i < cmode_len; i++)
  {

    if (tmp_ind[i] > X_ind[i])
    {
      return 1;
    }
    else if (tmp_ind[i] < X_ind[i])
    {
      return -1;
    }
  }
  return 0;
}

void get_mode_order(SparseTensor *spt, IType *mode_order, IType *cmode_pos, IType cnmodes, IType fnmodes)
{
  int ci = 0;
  for (ci = 0; ci < cnmodes; ci++)
  {
    mode_order[ci] = cmode_pos[ci];
  }
  int fi = ci;
  spt->fmode = (IType *)malloc((fnmodes) * sizeof(IType));
  for (int n = 0; n < (cnmodes + fnmodes); ++n)
  {
    bool in_cmode = 0;
    for (int i = 0; i < cnmodes; ++i)
    {
      if (n == cmode_pos[i])
      {
        in_cmode = 1;
      }
    }
    if (!in_cmode)
    {
      mode_order[fi] = n;
      spt->fmode[fi - ci] = n;
      fi++;
    }
  }
}
void preprocess_tensor(SparseTensor *X, SparseTensor *Y, IType *dims_acc, tmp_sptensor_struct **tmp_X, PinnedMemoryPool *pinned_pool, CudaMemoryPool *cuda_pool)
{
  double start = omp_get_wtime();
  tmp_SparseTensor *_tmp_X = (tmp_SparseTensor *)malloc(sizeof(tmp_SparseTensor));
  IType *mode_order_X = (IType *)malloc(X->nmodes * sizeof(IType));
  IType *mode_order_Y = (IType *)malloc(Y->nmodes * sizeof(IType));

  // X->cidx = (IType **)malloc(sizeof(IType *) * X->nmodes);
  X->dims = (IType *)pinned_pool->allocate(sizeof(IType) * X->nmodes);

  // for (IType i = 0; i < X->nmodes; i++)
  // {
  //   X->cidx[i] = (IType *)malloc(sizeof(IType) * X->nnz); 
  // }

  // Y->cidx = (IType **)malloc(sizeof(IType *) * Y->nmodes);
  Y->dims = (IType *)pinned_pool->allocate(sizeof(IType) * Y->nmodes);

  // for (IType i = 0; i < Y->nmodes; i++)
  // {
  //   Y->cidx[i] = (IType *)malloc(sizeof(IType) * Y->nnz);
  // }

  get_mode_order(X, mode_order_X, X->cmode, X->cnmodes, X->fnmodes);
  get_mode_order(Y, mode_order_Y, Y->cmode, Y->cnmodes, Y->fnmodes);
  printf("get mode order\n");

  printf("X->fnmodes: %d\n", X->fnmodes);
  IType *fdims = (IType *)malloc((X->fnmodes + Y->fnmodes) * sizeof(IType));
  for (int i = 0; i < X->fnmodes; i++)
  {
    fdims[i] = X->orig_dims[X->fmode[i]];
  }
  for (int i = 0; i < Y->fnmodes; i++)
  {
    fdims[X->fnmodes + i] = Y->orig_dims[Y->fmode[i]];
  }
  for (int i = 0; i < (X->fnmodes + Y->fnmodes); i++)
  {
    dims_acc[i] = 1;
    printf("============i: %d\n", i);
    for (int j = i; j < (X->fnmodes + Y->fnmodes); j++)
    {
      dims_acc[i] *= fdims[j];
      printf("fdims[%d]: %d\n", j, fdims[j]);
    }
  }

  for (int i = 0; i < (X->fnmodes + Y->fnmodes); i++)
  {
    printf("dims_acc[%d]: %d\n", i, dims_acc[i]);
  }
  if (cuda_pool->is_available(dims_acc[0] * sizeof(FType)) == false){
    fprintf(stderr, "Out of memory\n");
    printf("not available\n");
    exit(1);
  }
  printf("before sort time: %f\n", omp_get_wtime() - start);
  start = omp_get_wtime();
  permute_tensor(X, mode_order_X);
  permute_tensor(Y, mode_order_Y);

  // for (int i = 0; i < X->nmodes; i++)
  // {
  //   printf("orig_dim: %d, dim: %d\n", X->orig_dims[i], X->dims[i]);
  //   printf("X->cidx[%d][0]: %d\n", i, X->cidx[i][0]);
  // }

  printf("permute time: %f\n", omp_get_wtime() - start);
  start = omp_get_wtime();

#pragma omp parallel for schedule(static) num_threads(2)
  for (int i = 0; i < 2; i++)
  {
    if (i == 0)
      sort_index(X, 1, 12);
    else
      sort_index(Y, 1, 12);
  }

  printf("sort time: %f\n", omp_get_wtime() - start);

  _tmp_X->XO = (tmp_SparseTensor_element **)malloc(sizeof(tmp_SparseTensor_element *) * X->nnz);
  *tmp_X = _tmp_X;

  return;
}

void write_back(FType *ZT_vals, SparseTensor *Z, IType *acc_dims, PinnedMemoryPool *pinned_pool)
{ // dim
  IType max_dim = 1;

  for (IType i = 0; i < Z->nmodes; i++)
  {
    max_dim *= Z->dims[i];
  }
  printf("Z->nmode: %d\n", Z->nmodes);

  IType nnz = 0;
  printf("Z->nmodes: %d, max_dim: %d\n", Z->nmodes, max_dim);

#pragma omp parallel for reduction(+ : nnz)
  for (IType f = 0; f < max_dim; f++)
  {
    // if (ZT_vals[f] != static_cast<FType>(0))
    // {
    //   printf("ZT_vals[%d]: %f\n", f, ZT_vals[f]);
    // }
    if (ZT_vals[f] != static_cast<FType>(0))
    {
      nnz++;
    }
  }
  printf("nnz: %d\n", nnz);

  Z->nnz = nnz;
  Z->cidx = (IType **)pinned_pool->allocate(sizeof(IType *) * Z->nmodes);
  for (IType i = 0; i < Z->nmodes; i++)
  {
    Z->cidx[i] = (IType *)pinned_pool->allocate(sizeof(IType) * Z->nnz);
  }
  Z->vals = (FType *)pinned_pool->allocate(sizeof(FType) * Z->nnz);

  IType id = 0;
  for (IType f = 0; f < max_dim; f++)
  {
    if (ZT_vals[f] > 0)
    {
      IType index = f;
      for (int j = 0; j < Z->nmodes - 1; j++)
      {
        Z->vals[id] = ZT_vals[f];
        Z->cidx[j][id] = index / acc_dims[j + 1];
        index = index % acc_dims[j + 1];
      }
      Z->cidx[Z->nmodes - 1][id] = index;
      id++;
    }
  }
  return;
}