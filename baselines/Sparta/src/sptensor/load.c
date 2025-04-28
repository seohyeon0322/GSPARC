/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <ParTI.h>
#include "sptensor.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <numa.h>
#include <string.h>

typedef unsigned long long IType;

struct ftype
{
  char * extension;
  int type;
};

static struct ftype file_extensions[] = {
  { ".tns", 0 },
  { ".coo", 0 },
  { ".bin", 1 },
  { NULL, 0}
};


static int get_file_type(
    char const * const fname)
{
  /* find last . in filename */
  char const * const suffix = strrchr(fname, '.');
  if(suffix == NULL) {
    goto NOT_FOUND;
  }

  size_t idx = 0;
  do {
    if(strcmp(suffix, file_extensions[idx].extension) == 0) {
      return file_extensions[idx].type;
    }
  } while(file_extensions[++idx].extension != NULL);


  /* default to text coordinate format */
  NOT_FOUND:
  fprintf(stderr, "extension for '%s' not recognized. "
                  "Defaulting to ASCII coordinate form.\n", fname);
  return 0;
}
void read_tns_dims(
  FILE* fin,
  IType* num_modes,
  IType* nnz,
  IType** dims
)
{
  // count the number of modes
  IType nmodes = 0;
  ssize_t nread;
  char* line = NULL;
  size_t len = 0;
  while((nread = getline(&line, &len, fin)) != -1) {
    // if line is not empty or a comment (#)
    if(nread > 1 && line[0] != '#') {
      char* ptr = strtok(line, " \t");
      while(ptr != NULL) {
        ++nmodes;
        ptr = strtok(NULL, " \t");
      }
      break;
    }
  }
  --nmodes;
  *num_modes = nmodes;


  // Calculate the tensor dimensions
  IType* tmp_dims = (IType*) malloc(sizeof(IType) * nmodes);
  assert(tmp_dims);
  for(IType i = 0; i < nmodes; i++) {
    tmp_dims[i] = 0;
  }
  IType tmp_nnz = 0;

  rewind(fin);
  while((nread = getline(&line, &len, fin)) != -1) {
    // if line is not empty or a comment
    if(nread > 1 && line[0] != '#') {
      char* ptr = line;
      for(IType i = 0; i < nmodes; i++) {
        IType index = strtoull(ptr, &ptr, 10);
        if(index > tmp_dims[i]) {
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

static int p_tt_read_file(sptSparseTensor *tsr, sptIndex start_index, FILE *fp)
{
    int iores, retval;
    sptIndex mode;
    IType nnz = 0;
    IType nmodes = 0;
    IType* dims = NULL;
    read_tns_dims(fp, &nmodes, &nnz, &dims);
    // iores = fscanf(fp, "%u", &tsr->nmodes);
    tsr->nmodes = nmodes;
    printf("nmodes: %d\n", tsr->nmodes);
    spt_CheckOSError(iores < 0, "SpTns Load");
    /* Only allocate space for sortorder, initialized to 0s. */
    tsr->sortorder = malloc(tsr->nmodes * sizeof tsr->sortorder[0]);
    spt_CheckOSError(!tsr->sortorder, "SpTns Load");
    memset(tsr->sortorder, 0, tsr->nmodes * sizeof tsr->sortorder[0]);
    tsr->ndims = malloc(tsr->nmodes * sizeof *tsr->ndims);
    // spt_CheckOSError(!tsr->ndims, "SpTns Load");
    for(mode = 0; mode < tsr->nmodes; ++mode) {
        tsr->ndims[mode] = dims[mode];
    }

    // tsr->nnz = nnz;

    tsr->inds = malloc(tsr->nmodes * sizeof *tsr->inds);
    spt_CheckOSError(!tsr->inds, "SpTns Load");
    for(mode = 0; mode < tsr->nmodes; ++mode) {
        retval = sptNewIndexVector(&tsr->inds[mode], 0, 0);
        spt_CheckError(retval, "SpTns Load", NULL);
    }
    retval = sptNewValueVector(&tsr->values, 0, 0);
    spt_CheckError(retval, "SpTns Load", NULL);
    while(retval == 0) {
        double value;
        for(mode = 0; mode < tsr->nmodes; ++mode) {
            sptIndex index;
            iores = fscanf(fp, "%u", &index);
            if(iores != 1) {
                retval = -1;
                break;
            }
            if(index < start_index) {
                spt_CheckError(SPTERR_VALUE_ERROR, "SpTns Load", "index < start_index");
            }
            sptAppendIndexVector(&tsr->inds[mode], index-start_index);
        }
        if(retval == 0) {
            iores = fscanf(fp, "%lf", &value);
            if(iores != 1) {
                retval = -1;
                break;
            }
            sptAppendValueVector(&tsr->values, value);
            ++tsr->nnz;
        }
    }
    for(mode = 0; mode < tsr->nmodes; ++mode) {
        tsr->inds[mode].len = tsr->nnz;
    }
    // sptSparseTensorCollectZeros(tsr);
    
    return 0;
}

void read_txt_dims(
  FILE* fin,
  IType* num_modes,
  IType* nnz,
  IType** dims
)
{
  IType nmodes = 0;
  ssize_t nread;
  char* line = NULL;
  size_t len = 0;

  // 첫 번째 유효한 라인을 읽어서 차원 수 계산
  while((nread = getline(&line, &len, fin)) != -1) {
    if(nread > 1 && line[0] != '#') {
      nmodes++;
    }
  }

  *num_modes = nmodes;

  // 텐서의 차원 크기를 계산
  IType* tmp_dims = (IType*) malloc(sizeof(IType) * nmodes);
  assert(tmp_dims);
  for(IType i = 0; i < nmodes; i++) {
    tmp_dims[i] = 0;
  }

  rewind(fin);
  IType tmp_nnz = 0;
  IType line_num = 0;

  while((nread = getline(&line, &len, fin)) != -1) {
    if(nread > 1 && line[0] != '#') {
      char* ptr = strtok(line, " \t");
      IType index_count = 0;
      while(ptr != NULL) {
        IType index = (IType)(strtod(ptr, &ptr));
        if (index > tmp_dims[line_num]) {
          tmp_dims[line_num] = index;
        }
        if (line_num == 0) {
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
  FILE* index_fin,
  FILE* value_fin,
  sptSparseTensor* tensor,
  IType num_modes,
  IType nnz
)
{
  IType tmp_nnz = 0;
  ssize_t nread;
  char* line = NULL;
  size_t len = 0;

  rewind(index_fin);
  IType line_num = 0;

  while((nread = getline(&line, &len, index_fin)) != -1) {
    if(nread > 1 && line[0] != '#') {
      char* ptr = strtok(line, " \t");
      while(ptr != NULL) {
        double index = strtod(ptr, &ptr);
        sptAppendIndexVector(&tensor->inds[line_num], (IType)index-1);
        ptr = strtok(NULL, " \t");
      }
      line_num++;
    }
  }

  // value.txt를 처음부터 읽음
  rewind(value_fin);
  double value; 

  for (IType i = 0; i < nnz; i++) {
    fscanf(value_fin, "%f", &value);
    sptAppendValueVector(&tensor->values, value);
    ++tensor->nnz;
  }

  free(line);
  return;
}


int ImportQuantumTensor(
  sptSparseTensor* tsr,
  char const * const filepath
)
{
  printf("filepath: %s\n", filepath);
  char index_file[256];
  char value_file[256];

  snprintf(index_file, sizeof(index_file), "%s/indices.txt", filepath);
  snprintf(value_file, sizeof(value_file), "%s/values.txt", filepath);

  printf("index_file: %s\n", index_file);
  printf("value_file: %s\n", value_file);

  FILE* index_fp = fopen(index_file, "r");
  FILE* value_fp = fopen(value_file, "r");
  assert(index_fp != NULL && value_fp != NULL);

  int iores, retval;
  sptIndex mode;
  IType nnz = 0;
  IType nmodes = 0;
  IType* dims = NULL;

    // 차원 및 nnz 정보를 읽어옴
  read_txt_dims(index_fp, &nmodes, &nnz, &dims);
  tsr->nmodes = nmodes; 
  printf("nmodes: %d\n", tsr->nmodes);
  spt_CheckOSError(iores < 0, "SpTns Load");

  tsr->sortorder = malloc(tsr->nmodes * sizeof tsr->sortorder[0]);
  spt_CheckOSError(!tsr->sortorder, "SpTns Load");
  memset(tsr->sortorder, 0, tsr->nmodes * sizeof tsr->sortorder[0]);
  tsr->ndims = malloc(tsr->nmodes * sizeof *tsr->ndims);

  for (mode = 0; mode < tsr->nmodes; ++mode) {
    tsr->ndims[mode] = dims[mode];
  }

  tsr->inds = malloc(tsr->nmodes * sizeof *tsr->inds);
  spt_CheckOSError(!tsr->inds, "SpTns Load");
  for(mode = 0; mode < tsr->nmodes; ++mode) {
    retval = sptNewIndexVector(&tsr->inds[mode], 0, 0);
    spt_CheckError(retval, "SpTns Load", NULL);
  }

  retval = sptNewValueVector(&tsr->values, 0, 0);
  spt_CheckError(retval, "SpTns Load", NULL);
  
  read_txt_data(index_fp, value_fp, tsr, nmodes, nnz);

    for(mode = 0; mode < tsr->nmodes; ++mode) {
        tsr->inds[mode].len = tsr->nnz;
    }

  fclose(index_fp);
  fclose(value_fp);
  return 0;
}

static void read_binary_header(
  FILE * fin,
  bin_header * header)
{
  fread(&(header->magic), sizeof(header->magic), 1, fin);
  fread(&(header->idx_width), sizeof(header->idx_width), 1, fin);
  fread(&(header->val_width), sizeof(header->val_width), 1, fin);

  if(header->idx_width > PARTI_INDEX_TYPEWIDTH / 8) {
    fprintf(stderr, "ERROR input has %lu-bit integers. "
                    "Build with PARTI_INDEX_TYPEWIDTH %lu\n",
                    header->idx_width * 8, header->idx_width * 8);
    exit(-1);
  }

  if(header->val_width > PARTI_VALUE_TYPEWIDTH / 8) {
    fprintf(stderr, "WARNING input has %lu-bit floating-point values. "
                    "Build with PARTI_VALUE_TYPEWIDTH %lu for full precision\n",
                    header->val_width * 8, header->val_width * 8);
  }
}

static void fill_binary_idx(
    sptIndex * const buffer,
    sptIndex const count,
    bin_header const * const header,
    FILE * fin)
{
  if(header->idx_width == sizeof(sptIndex)) {
    fread(buffer, sizeof(sptIndex), count, fin);
  } else {
    /* read in uint32_t in a buffered fashion */
    sptIndex const BUF_LEN = 1024*1024;
    uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    for(sptIndex n=0; n < count; n += BUF_LEN) {
      sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
      fread(ubuf, sizeof(*ubuf), read_count, fin);
      #pragma omp parallel for schedule(static)
      for(sptIndex i=0; i < read_count; ++i) {
        buffer[n + i] = ubuf[i];
      }
    }
    free(ubuf);
  }
}


static void fill_binary_nnzidx(
    sptNnzIndex * const buffer,
    sptIndex const count,
    bin_header const * const header,
    FILE * fin)
{
  if(header->idx_width == sizeof(sptNnzIndex)) {
    fread(buffer, sizeof(sptNnzIndex), count, fin);
  } else {
    /* read in uint32_t in a buffered fashion */
    sptIndex const BUF_LEN = 1024*1024;
    uint32_t * ubuf = (uint32_t*)malloc(BUF_LEN * sizeof(*ubuf));
    for(sptIndex n=0; n < count; n += BUF_LEN) {
      sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
      fread(ubuf, sizeof(*ubuf), read_count, fin);
      #pragma omp parallel for schedule(static)
      for(sptIndex i=0; i < read_count; ++i) {
        buffer[n + i] = ubuf[i];
      }
    }
    free(ubuf);
  }
}


static void fill_binary_val(
    sptValue * const buffer,
    sptIndex const count,
    bin_header const * const header,
    FILE * fin)
{
  if(header->val_width == sizeof(sptValue)) {
    fread(buffer, sizeof(sptValue), count, fin);
  } else {
    /* read in float in a buffered fashion */
    sptIndex const BUF_LEN = 1024*1024;

    /* select whichever *is not* configured with. */
#if PARTI_VALUE_TYPEWIDTH == 64
    float * ubuf = (float*)malloc(BUF_LEN * sizeof(*ubuf));
#else
    double * ubuf = (double*)malloc(BUF_LEN * sizeof(*ubuf));
#endif

    for(sptIndex n=0; n < count; n += BUF_LEN) {
      sptIndex const read_count = BUF_LEN < count - n ? BUF_LEN : count - n;
      fread(ubuf, sizeof(*ubuf), read_count, fin);
      #pragma omp parallel for schedule(static)
      for(sptIndex i=0; i < read_count; ++i) {
        buffer[n + i] = ubuf[i];
      }
    }
    free(ubuf);
  }
}

/**
* @brief Read a COORD tensor from a binary file, converting from smaller idx or
*        val precision if necessary.
*
* @param fin The file to read from.
*
* @return The parsed tensor.
*/
static int p_tt_read_binary_file(sptSparseTensor *tsr, FILE * fin)
{
  int result;
  bin_header header;
  read_binary_header(fin, &header);
  // printf("header.magic: %d\n", header.magic);
  // printf("header.idx_width: %lu\n", header.idx_width);
  // printf("header.val_width: %lu\n", header.val_width);

  sptNnzIndex nnz = 0;
  sptIndex nmodes = 0;

  fill_binary_idx(&nmodes, 1, &header, fin);

  sptIndex * dims = (sptIndex *) malloc (nmodes * sizeof(*dims));
  fill_binary_idx(dims, nmodes, &header, fin);
  fill_binary_nnzidx(&nnz, 1, &header, fin);

  /* allocate structures */
  sptNewSparseTensor(tsr, nmodes, dims);
  tsr->nnz = nnz;
  for(sptIndex m=0; m < nmodes; ++m) {
    result = sptResizeIndexVector(&tsr->inds[m], nnz);
    spt_CheckError(result, "SpTns Read", NULL);
  }
  result = sptResizeValueVector(&tsr->values, nnz);
  spt_CheckError(result, "SpTns Read", NULL);

  /* fill in tensor data */
  for(sptIndex m=0; m < nmodes; ++m) {
    fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
  }
  fill_binary_val(tsr->values.data, nnz, &header, fin);

  return 0;
}

//numa
static int p_tt_read_binary_file_numa(sptSparseTensor *tsr, FILE * fin, int numa_node)
{
  int result;
  bin_header header;
  read_binary_header(fin, &header);
  // printf("header.magic: %d\n", header.magic);
  // printf("header.idx_width: %lu\n", header.idx_width);
  // printf("header.val_width: %lu\n", header.val_width);

  sptNnzIndex nnz = 0;
  sptIndex nmodes = 0;

  fill_binary_idx(&nmodes, 1, &header, fin);

  sptIndex * dims = (sptIndex *) numa_alloc_onnode (nmodes * sizeof(*dims), numa_node);
  fill_binary_idx(dims, nmodes, &header, fin);
  fill_binary_nnzidx(&nnz, 1, &header, fin);

  /* allocate structures */
  sptNewSparseTensorNuma(tsr, nmodes, dims, numa_node);
  tsr->nnz = nnz;
  for(sptIndex m=0; m < nmodes; ++m) {
    result = sptResizeIndexVectorNuma(&tsr->inds[m], nnz);
    spt_CheckError(result, "SpTns Read", NULL);
  }
  result = sptResizeValueVectorNuma(&tsr->values, nnz);
  spt_CheckError(result, "SpTns Read", NULL);

  /* fill in tensor data */
  for(sptIndex m=0; m < nmodes; ++m) {
    fill_binary_idx(tsr->inds[m].data, nnz, &header, fin);
  }
  fill_binary_val(tsr->values.data, nnz, &header, fin);

  return 0;
}


/**
 * Load the contents of a sparse tensor fro a text file
 * @param tsr         th sparse tensor to store into
 * @param start_index the index of the first element in array. Set to 1 for MATLAB compability, else set to 0
 * @param fp          the file to read from
 */
int sptLoadSparseTensor(sptSparseTensor *tsr, sptIndex start_index, char const * const fname) 
{
    FILE * fp = fopen(fname, "r");
    sptAssert(fp != NULL);

    int iores;
    switch(get_file_type(fname)) {
        case 0:
            iores = p_tt_read_file(tsr, start_index, fp);
            spt_CheckOSError(iores != 0, "SpTns Load");
            break;
        case 1:
            iores = p_tt_read_binary_file(tsr, fp);
            spt_CheckOSError(iores != 0, "SpTns Load");
            break;
          
    }

    fclose(fp);

    return 0;
}

//numa
int sptLoadSparseTensorNuma(sptSparseTensor *tsr, sptIndex start_index, char const * const fname, int numa_node) 
{
    FILE * fp = fopen(fname, "r");
    sptAssert(fp != NULL);

    int iores;
    switch(get_file_type(fname)) {
        case 0:
            iores = p_tt_read_file(tsr, start_index, fp);
            spt_CheckOSError(iores != 0, "SpTns Load");
            break;
        case 1:
            iores = p_tt_read_binary_file_numa(tsr, fp, numa_node);
            spt_CheckOSError(iores != 0, "SpTns Load");
            break;
    }

    fclose(fp);

    return 0;
}

void sptLoadShuffleFile(sptSparseTensor *tsr, FILE *fs, sptIndex ** map_inds)
{
    sptNnzIndex line_count = 0;
    sptNnzIndex dim_count = 0;
    // int iores;
    for(sptIndex mode = 0; mode < tsr->nmodes; ++mode) {
        dim_count += tsr->ndims[mode];
        for(sptIndex i = 0; i < tsr->ndims[mode]; ++i) {
            fscanf(fs, "%u", &(map_inds[mode][i]));
            -- map_inds[mode][i];
            ++ line_count;
        }
    }
    sptAssert(dim_count == line_count);    
    
    return;
}
