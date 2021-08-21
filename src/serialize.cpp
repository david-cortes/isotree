/*    Isolation forests and variations thereof, with adjustments for incorporation
*     of categorical variables and missing values.
*     Writen for C++11 standard and aimed at being used in R and Python.
*     
*     This library is based on the following works:
*     [1] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou.
*         "Isolation forest."
*         2008 Eighth IEEE International Conference on Data Mining. IEEE, 2008.
*     [2] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou.
*         "Isolation-based anomaly detection."
*         ACM Transactions on Knowledge Discovery from Data (TKDD) 6.1 (2012): 3.
*     [3] Hariri, Sahand, Matias Carrasco Kind, and Robert J. Brunner.
*         "Extended Isolation Forest."
*         arXiv preprint arXiv:1811.02141 (2018).
*     [4] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou.
*         "On detecting clustered anomalies using SCiForest."
*         Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, Berlin, Heidelberg, 2010.
*     [5] https://sourceforge.net/projects/iforest/
*     [6] https://math.stackexchange.com/questions/3388518/expected-number-of-paths-required-to-separate-elements-in-a-binary-tree
*     [7] Quinlan, J. Ross. C4. 5: programs for machine learning. Elsevier, 2014.
*     [8] Cortes, David. "Distance approximation using Isolation Forests." arXiv preprint arXiv:1910.12362 (2019).
*     [9] Cortes, David. "Imputing missing values with unsupervised random trees." arXiv preprint arXiv:1911.06646 (2019).
* 
*     BSD 2-Clause License
*     Copyright (c) 2019-2021, David Cortes
*     All rights reserved.
*     Redistribution and use in source and binary forms, with or without
*     modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright notice, this
*       list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright notice,
*       this list of conditions and the following disclaimer in the documentation
*       and/or other materials provided with the distribution.
*     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
*     AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
*     IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
*     DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
*     FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
*     DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
*     SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*     CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
*     OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
*     OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "isotree.hpp"


using std::uint8_t;
using std::int8_t;
using std::uint16_t;
using std::int16_t;
using std::uint32_t;
using std::int32_t;
using std::uint64_t;
using std::int64_t;

/* https://stackoverflow.com/questions/16696297/ftell-at-a-position-past-2gb */
/* TODO: do CLANG and ICC have similar functionality? */
#if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)) && (SIZE_MAX >= UINT64_MAX)
#   ifdef _MSC_VER
#       include <stdio.h>
#       define fseek_ _fseeki64
#       define ftell_ _ftelli64
#       define fpos_t_ __int64
#   elif defined(__GNUG__) || defined(__GNUC__)
#       ifndef _FILE_OFFSET_BITS
#           define _FILE_OFFSET_BITS 64
#       endif
#       include <stdio.h>
#       define fseek_ fseeko
#       define ftell_ ftello
#       define fpos_t_ off_t
#   else
        using std::feof;
        using std::fwrite;
        using std::fread;
        using std::fopen;
        using std::fclose;
        using std::ftell;
        using std::fseek;
#       define fseek_ fseek
#       define ftell_ ftell
#       define fpos_t_ long /* <- might overflow with large files */
#   endif
#else
    using std::feof;
    using std::fwrite;
    using std::fread;
    using std::fopen;
    using std::fclose;
    using std::ftell;
    using std::fseek;
#   define fseek_ fseek
#   define ftell_ ftell
#   define fpos_t_ long
#endif

#if defined(DBL_MANT_DIG) && (DBL_MANT_DIG == 53) && (FLT_RADIX == 2)
    #define HAS_NORMAL_DOUBLE
#endif

#if INT_MAX == INT16_MAX
    #define HAS_INT16
#elif INT_MAX == INT32_MAX
    #define HAS_INT32
#elif INT_MAX == INT64_MAX
    #define HAS_INT64
#else
    #define HAS_INT_OTHER
#endif

#if SIZE_MAX == UINT32_MAX
    #define HAS_SIZE32
#elif SIZE_MAX == UINT64_MAX
    #define HAS_SIZE64
#else
    #define HAS_SIZE_OTHER
#endif

const char *watermark = "isotree_model";
const char *incomplete_watermark = "incomplete___";
static const size_t SIZE_WATERMARK = 13;
enum DoubleTypeStructure {IsNormalDouble=1, IsAbnormalDouble=2};
enum PlatformSize {Is16Bit=1, Is32Bit=2, Is64Bit=3, IsOther=4};
enum PlatformEndianness {PlatformLittleEndian=1, PlatformBigEndian=2};
enum ModelTypes {IsoForestModel=1, ExtIsoForestModel=2, ImputerModel=3, AllObjectsCombined=4};
enum EndingIndicator {
    EndsHere=0,
    HasSingleVarModelNext=1,
    HasExtModelNext=2,
    HasImputerNext=3,
    HasSingleVarModelPlusImputerNext=4,
    HasExtModelPlusImputerNext=5,
    HasSingleVarModelPlusMetadataNext=6,
    HasExtModelPlusMetadataNext=7,
    HasSingleVarModelPlusImputerPlusMetadataNext=8,
    HasExtModelPlusImputerPlusMetadataNext=9,
    HasMoreTreesNext=10
};

#ifdef _MSC_VER
#include <stdlib.h>
void swap16b(char *bytes)
{
    if (sizeof(unsigned short) == 2) {
        unsigned short temp;
        memcpy(&temp, bytes, sizeof(unsigned short));
        temp = _byteswap_ushort(temp);
        memcpy(bytes, &temp, sizeof(unsigned short));
    }
    
    else {
        std::swap(bytes[0], bytes[1]);
    }
}
void swap32b(char *bytes)
{
    if (sizeof(unsigned long) == 4) {
        unsigned long temp;
        memcpy(&temp, bytes, sizeof(unsigned long));
        temp = _byteswap_ulong(temp);
        memcpy(bytes, &temp, sizeof(unsigned long));
    }
    
    else {
        std::swap(bytes[0], bytes[3]);
        std::swap(bytes[1], bytes[2]);
    }
}
void swap64b(char *bytes)
{
    unsigned __int64 temp;
    memcpy(&temp, bytes, sizeof(unsigned __int64));
    temp = _byteswap_uint64(temp);
    memcpy(bytes, &temp, sizeof(unsigned __int64));
}
#elif defined(__GNUC__) && (__GNUC__ >= 5) && !defined(_WIN32)
void swap16b(char *bytes)
{
    uint16_t temp;
    memcpy(&temp, bytes, sizeof(uint16_t));
    temp = __builtin_bswap16(temp);
    memcpy(bytes, &temp, sizeof(uint16_t));
}
void swap32b(char *bytes)
{
    uint32_t temp;
    memcpy(&temp, bytes, sizeof(uint32_t));
    temp = __builtin_bswap32(temp);
    memcpy(bytes, &temp, sizeof(uint32_t));
}
void swap64b(char *bytes)
{
    uint64_t temp;
    memcpy(&temp, bytes, sizeof(uint64_t));
    temp = __builtin_bswap64(temp);
    memcpy(bytes, &temp, sizeof(uint64_t));
}
#else
void swap16b(char *bytes)
{
    std::swap(bytes[0], bytes[1]);
}
void swap32b(char *bytes)
{
    std::swap(bytes[0], bytes[3]);
    std::swap(bytes[1], bytes[2]);
}
void swap64b(char *bytes)
{
    std::swap(bytes[0], bytes[7]);
    std::swap(bytes[1], bytes[6]);
    std::swap(bytes[2], bytes[5]);
    std::swap(bytes[3], bytes[4]);
}
#endif
void endian_swap(float &bytes)
{
    #ifdef HAS_NORMAL_DOUBLE
    swap32b((char*)&bytes);
    #else
    std::reverse((char*)&bytes, (char*)&bytes + sizeof(float));
    #endif
}
void endian_swap(double &bytes)
{
    #ifdef HAS_NORMAL_DOUBLE
    swap64b((char*)&bytes);
    #else
    std::reverse((char*)&bytes, (char*)&bytes + sizeof(double));
    #endif
}
void endian_swap(uint8_t &bytes)
{
    return;
}
void endian_swap(uint16_t &bytes)
{
    swap16b((char*)&bytes);
}
void endian_swap(uint32_t &bytes)
{
    swap32b((char*)&bytes);
}
void endian_swap(uint64_t &bytes)
{
    swap64b((char*)&bytes);
}
void endian_swap(int8_t &bytes)
{
    return;
}
void endian_swap(int16_t &bytes)
{
    swap16b((char*)&bytes);
}
void endian_swap(int32_t &bytes)
{
    swap32b((char*)&bytes);
}
void endian_swap(int64_t &bytes)
{
    swap64b((char*)&bytes);
}
/* Note: on macOS, some compilers will  take 'size_t' as different from 'uin64_t',
   hence it needs a separate one. However, in other compiler and platforms this
   leads to a a duplicated function definition, and thus needs this separation
   in names (otherwise, compilers such as GCC will not compile it). */
void endian_swap_size_t(char *bytes)
{
    #if (SIZE_MAX == UINT32_MAX)
    swap32b(bytes);
    #elif (SIZE_MAX == UINT64_MAX)
    swap64b(bytes);
    #else
    std::reverse(bytes, bytes + sizeof(size_t));
    #endif
}
void endian_swap_int(char *bytes)
{
    #if (INT_MAX == INT16_MAX)
    swap16b(bytes);
    #elif (INT_MAX == INT32_MAX)
    swap32b(bytes);
    #elif (SIZE_MAX == INT64_MAX)
    swap64b(bytes);
    #else
    std::reverse(bytes, bytes + sizeof(int));
    #endif
}
template <class T>
void endian_swap(T &bytes)
{
    std::reverse((char*)&bytes, (char*)&bytes + sizeof(T));
}

template <class dtype>
void swap_endianness(dtype *ptr, size_t n_els)
{
    #ifndef __GNUC__
    if (std::is_same<dtype, size_t>::value)
    {
        for (size_t ix = 0; ix < n_els; ix++)
            endian_swap_size_t((char*)&ptr[ix]);
        return;
    }

    else if (std::is_same<dtype, int>::value)
    {
        for (size_t ix = 0; ix < n_els; ix++)
            endian_swap_int((char*)&ptr[ix]);
        return;
    }
    #endif

    for (size_t ix = 0; ix < n_els; ix++)
        endian_swap(ptr[ix]);
}

const char* set_return_position(const char *in)
{
    return in;
}

char* set_return_position(char *in)
{
    return in;
}

fpos_t_ set_return_position(FILE *in)
{
    return ftell_(in);
}

#define pos_type_istream decltype(std::declval<std::istream>().tellg())

pos_type_istream set_return_position(std::istream &in)
{
    return in.tellg();
}

pos_type_istream set_return_position(std::ostream &in)
{
    return in.tellp();
}

void return_to_position(const char *&in, const char *saved_position)
{
    in = saved_position;
}

void return_to_position(char *&in, char *saved_position)
{
    in = saved_position;
}

void return_to_position(FILE *&in, fpos_t_ saved_position)
{
    fseek_(in, saved_position, SEEK_SET);
}

void return_to_position(std::istream &in, pos_type_istream saved_position)
{
    in.seekg(saved_position);
}

void return_to_position(std::ostream &in, pos_type_istream saved_position)
{
    in.seekp(saved_position);
}


bool has_wchar_t_file_serializers()
{
    #ifdef WCHAR_T_FUNS
    return true;
    #else
    return false;
    #endif
}

void throw_errno()
{
    throw std::runtime_error("Error " + std::to_string(errno) + " " + strerror(errno) + "\n");
}

void throw_ferror(FILE *file)
{
    if (!errno) fflush(file);
    throw_errno();
}

void throw_feoferr()
{
    throw std::runtime_error("Error: file ended unexpectedly.\n");
}

template <class dtype, class saved_type>
void convert_dtype(void *ptr_write_, std::vector<char> &buffer, size_t n_els)
{
    dtype *ptr_write = (dtype*)ptr_write_;
    saved_type *ptr_read = (saved_type*)buffer.data();

    if ((sizeof(dtype) <= sizeof(saved_type)) &&
        (saved_type)std::numeric_limits<dtype>::max() < std::numeric_limits<saved_type>::max())
    {
        const saved_type maxval = (saved_type) std::numeric_limits<dtype>::max();
        for (size_t el = 0; el < n_els; el++)
            if (ptr_read[el] > maxval)
                throw std::runtime_error("Error: serialized model has values too large for the current machine's types.\n");
    }

    for (size_t el = 0; el < n_els; el++)
        ptr_write[el] = (dtype)ptr_read[el];
}

template <class dtype>
void write_bytes(const void *ptr, const size_t n_els, char *&out)
{
    if (n_els == 0) return;
    memcpy(out, ptr, n_els * sizeof(dtype));
    out += n_els * sizeof(dtype);
}

template <class dtype>
void write_bytes(const void *ptr, const size_t n_els, std::ostream &out)
{
    if (n_els == 0) return;
    out.write((char*)ptr, n_els * sizeof(dtype));
    if (out.bad()) throw_errno();
}

template <class dtype>
void write_bytes(const void *ptr, const size_t n_els, FILE *&out)
{
    if (n_els == 0) return;
    size_t n_written = fwrite(ptr, sizeof(dtype), n_els, out);
    if (n_written != n_els || ferror(out)) throw_ferror(out);
}

template <class dtype>
void read_bytes(void *ptr, const size_t n_els, const char *&in)
{
    if (n_els == 0) return;
    memcpy(ptr, in, n_els * sizeof(dtype));
    in += n_els * sizeof(dtype);
}

template <class dtype, class saved_type>
void read_bytes(void *ptr, const size_t n_els, const char *&in, std::vector<char> &buffer, const bool diff_endian)
{
    if (std::is_same<dtype, saved_type>::value)
    {
        read_bytes<dtype>(ptr, n_els, in);
        if (diff_endian) swap_endianness((dtype*)ptr, n_els);
        return;
    }
    if (n_els == 0) return;
    if (buffer.size() < n_els * sizeof(saved_type))
        buffer.resize((size_t)2 * n_els * sizeof(saved_type));
    memcpy(buffer.data(), in, n_els * sizeof(saved_type));
    in += n_els * sizeof(saved_type);

    if (diff_endian) swap_endianness((saved_type*)buffer.data(), n_els);
    convert_dtype<dtype, saved_type>(ptr, buffer, n_els);
}

template <class dtype>
void read_bytes(void *ptr, const size_t n_els, char *&in)
{
    if (n_els == 0) return;
    memcpy(ptr, in, n_els * sizeof(dtype));
    in += n_els * sizeof(dtype);
}

template <class dtype, class saved_type>
void read_bytes(void *ptr, const size_t n_els, char *&in, std::vector<char> &buffer, const bool diff_endian)
{
    if (std::is_same<dtype, saved_type>::value)
    {
        read_bytes<dtype>(ptr, n_els, in);
        if (diff_endian) swap_endianness((dtype*)ptr, n_els);
        return;
    }
    if (n_els == 0) return;
    if (buffer.size() < n_els * sizeof(saved_type))
        buffer.resize((size_t)2 * n_els * sizeof(saved_type));
    memcpy(buffer.data(), in, n_els * sizeof(saved_type));
    in += n_els * sizeof(saved_type);

    if (diff_endian) swap_endianness((saved_type*)buffer.data(), n_els);
    convert_dtype<dtype, saved_type>(ptr, buffer, n_els);
}

template <class dtype>
void read_bytes(void *ptr, const size_t n_els, std::istream &in)
{
    if (n_els == 0) return;
    in.read((char*)ptr, n_els * sizeof(dtype));
    if (in.bad()) throw_errno();
}

template <class dtype, class saved_type>
void read_bytes(void *ptr, const size_t n_els, std::istream &in, std::vector<char> &buffer, const bool diff_endian)
{
    if (std::is_same<dtype, saved_type>::value)
    {
        read_bytes<dtype>(ptr, n_els, in);
        if (diff_endian) swap_endianness((dtype*)ptr, n_els);
        return;
    }
    if (n_els == 0) return;
    if (buffer.size() < n_els * sizeof(saved_type))
        buffer.resize((size_t)2 * n_els * sizeof(saved_type));
    in.read((char*)buffer.data(), n_els * sizeof(saved_type));
    if (in.bad()) throw_errno();

    if (diff_endian) swap_endianness((saved_type*)buffer.data(), n_els);
    convert_dtype<dtype, saved_type>(ptr, buffer, n_els);
}

template <class dtype>
void read_bytes(void *ptr, const size_t n_els, FILE *&in)
{
    if (n_els == 0) return;
    if (feof(in)) throw_feoferr();
    size_t n_read = fread(ptr, sizeof(dtype), n_els, in);
    if (n_read != n_els || ferror(in)) throw_ferror(in);
}

template <class dtype, class saved_type>
void read_bytes(void *ptr, const size_t n_els, FILE *&in, std::vector<char> &buffer, const bool diff_endian)
{
    if (std::is_same<dtype, saved_type>::value)
    {
        read_bytes<dtype>(ptr, n_els, in);
        if (diff_endian) swap_endianness((dtype*)ptr, n_els);
        return;
    }
    if (n_els == 0) return;
    if (feof(in)) throw_feoferr();
    if (buffer.size() < n_els * sizeof(saved_type))
        buffer.resize((size_t)2 * n_els * sizeof(saved_type));
    size_t n_read = fread(buffer.data(), sizeof(saved_type), n_els, in);
    if (n_read != n_els || ferror(in)) throw_ferror(in);

    if (diff_endian) swap_endianness((saved_type*)buffer.data(), n_els);
    convert_dtype<dtype, saved_type>(ptr, buffer, n_els);
}

template <class dtype>
void read_bytes(std::vector<dtype> &vec, const size_t n_els, const char *&in)
{
    if (n_els)
        vec.assign((dtype*)in, (dtype*)in + n_els);
    else
        vec.clear();
    vec.shrink_to_fit();
    in += n_els * sizeof(dtype);
}

template <class dtype, class saved_type>
void read_bytes(std::vector<dtype> &vec, const size_t n_els, const char *&in, std::vector<char> &buffer, const bool diff_endian)
{
    if (std::is_same<dtype, saved_type>::value)
    {
        read_bytes<dtype>(vec, n_els, in);
        if (diff_endian) swap_endianness(vec.data(), n_els);
        return;
    }
    if (n_els) {
        if (buffer.size() < n_els * sizeof(saved_type))
            buffer.resize((size_t)2 * n_els * sizeof(saved_type));
        read_bytes<saved_type>(buffer.data(), n_els, in);
        vec.resize(n_els);
        vec.shrink_to_fit();
        
        if (diff_endian) swap_endianness((saved_type*)buffer.data(), n_els);
        convert_dtype<dtype, saved_type>(vec.data(), buffer, n_els);
    }
    
    else {
        vec.clear();
        vec.shrink_to_fit();
    }

    in += n_els * sizeof(saved_type);
}

template <class dtype>
void read_bytes(std::vector<dtype> &vec, const size_t n_els, std::istream &in)
{
    vec.resize(n_els);
    vec.shrink_to_fit();

    if (n_els) {
        in.read((char*)vec.data(), n_els * sizeof(dtype));
        if (in.bad()) throw_errno();
    }
}

template <class dtype, class saved_type>
void read_bytes(std::vector<dtype> &vec, const size_t n_els, std::istream &in, std::vector<char> &buffer, const bool diff_endian)
{
    if (std::is_same<dtype, saved_type>::value)
    {
        read_bytes<dtype>(vec, n_els, in);
        if (diff_endian) swap_endianness(vec.data(), n_els);
        return;
    }
    vec.resize(n_els);
    vec.shrink_to_fit();

    if (n_els) {
        if (buffer.size() < n_els * sizeof(saved_type))
            buffer.resize((size_t)2 * n_els * sizeof(saved_type));
        in.read(buffer.data(), n_els * sizeof(saved_type));
        if (in.bad()) throw_errno();

        if (diff_endian) swap_endianness((saved_type*)buffer.data(), n_els);
        convert_dtype<dtype, saved_type>(vec.data(), buffer, n_els);
    }
}

template <class dtype>
void read_bytes(std::vector<dtype> &vec, const size_t n_els, FILE *&in)
{
    vec.resize(n_els);
    vec.shrink_to_fit();
    
    if (n_els) {
        if (feof(in)) throw_feoferr();
        size_t n_read = fread(vec.data(), sizeof(dtype), n_els, in);
        if (n_read != n_els || ferror(in)) throw_ferror(in);
    }
}

template <class dtype, class saved_type>
void read_bytes(std::vector<dtype> &vec, const size_t n_els, FILE *&in, std::vector<char> &buffer, const bool diff_endian)
{
    if (std::is_same<dtype, saved_type>::value)
    {
        read_bytes<dtype>(vec, n_els, in);
        if (diff_endian) swap_endianness(vec.data(), n_els);
        return;
    }
    vec.resize(n_els);
    vec.shrink_to_fit();

    if (n_els) {
        if (feof(in)) throw_feoferr();
        if (buffer.size() < n_els * sizeof(saved_type))
            buffer.resize((size_t)2 * n_els * sizeof(saved_type));

        size_t n_read = fread(buffer.data(), sizeof(saved_type), n_els, in);
        if (n_read != n_els || ferror(in)) throw_ferror(in);

        if (diff_endian) swap_endianness((saved_type*)buffer.data(), n_els);
        convert_dtype<dtype, saved_type>(vec.data(), buffer, n_els);
    }
}

size_t get_size_node(const IsoTree &node)
{
    size_t n_bytes = 0;
    n_bytes += sizeof(uint8_t);
    n_bytes += sizeof(int);
    n_bytes += sizeof(double) * 6;
    n_bytes += sizeof(size_t) * 4;
    n_bytes += sizeof(signed char) * node.cat_split.size();
    return n_bytes;
}

template <class otype>
void serialize_node(const IsoTree &node, otype &out)
{
    if (interrupt_switch) return;
    
    uint8_t data_en = (uint8_t)node.col_type;
    write_bytes<uint8_t>((void*)&data_en, (size_t)1, out);

    write_bytes<int>((void*)&node.chosen_cat, (size_t)1, out);

    double data_doubles[] = {
        node.num_split,
        node.pct_tree_left,
        node.score,
        node.range_low,
        node.range_high,
        node.remainder
    };
    write_bytes<double>((void*)data_doubles, (size_t)6, out);

    size_t data_sizets[] = {
        node.col_num,
        node.tree_left,
        node.tree_right,
        node.cat_split.size()
    };
    write_bytes<size_t>((void*)data_sizets, (size_t)4, out);

    if (node.cat_split.size())
    write_bytes<signed char>((void*)node.cat_split.data(), node.cat_split.size(), out);
}

template <class itype>
void deserialize_node(IsoTree &node, itype &in)
{
    if (interrupt_switch) return;

    uint8_t data_en;
    read_bytes<uint8_t>((void*)&data_en, (size_t)1, in);
    node.col_type = (ColType)data_en;

    read_bytes<int>((void*)&node.chosen_cat, (size_t)1, in);

    double data_doubles[6];
    read_bytes<double>((void*)data_doubles, (size_t)6, in);
    node.num_split = data_doubles[0];
    node.pct_tree_left = data_doubles[1];
    node.score = data_doubles[2];
    node.range_low = data_doubles[3];
    node.range_high = data_doubles[4];
    node.remainder = data_doubles[5];

    size_t data_sizets[4];
    read_bytes<size_t>((void*)data_sizets, (size_t)4, in);
    node.col_num = data_sizets[0];
    node.tree_left = data_sizets[1];
    node.tree_right = data_sizets[2];
    read_bytes<signed char>(node.cat_split, data_sizets[3], in);
}

template <class itype, class saved_int_t, class saved_size_t>
void deserialize_node(IsoTree &node, itype &in, std::vector<char> &buffer, const bool diff_endian)
{
    if (interrupt_switch) return;

    uint8_t data_en;
    read_bytes<uint8_t>((void*)&data_en, (size_t)1, in);
    node.col_type = (ColType)data_en;

    read_bytes<int, saved_int_t>((void*)&node.chosen_cat, (size_t)1, in, buffer, diff_endian);

    double data_doubles[6];
    read_bytes<double, double>((void*)data_doubles, (size_t)6, in, buffer, diff_endian);
    node.num_split = data_doubles[0];
    node.pct_tree_left = data_doubles[1];
    node.score = data_doubles[2];
    node.range_low = data_doubles[3];
    node.range_high = data_doubles[4];
    node.remainder = data_doubles[5];

    size_t data_sizets[4];
    read_bytes<size_t, saved_size_t>((void*)data_sizets, (size_t)4, in, buffer, diff_endian);
    node.col_num = data_sizets[0];
    node.tree_left = data_sizets[1];
    node.tree_right = data_sizets[2];
    read_bytes<signed char, signed char>(node.cat_split, data_sizets[3], in, buffer, diff_endian);
}

size_t get_size_node(const IsoHPlane &node)
{
    size_t n_bytes = 0;
    n_bytes += sizeof(double) * 5;
    n_bytes += sizeof(size_t) * 10;
    n_bytes += sizeof(size_t) * node.col_num.size();
    if (node.col_type.size()) {
        n_bytes += sizeof(uint8_t)*node.col_type.size();
    }
    n_bytes += sizeof(double)*node.coef.size();
    n_bytes += sizeof(double)*node.mean.size();
    if (node.cat_coef.size()) {
        for (const auto &vec : node.cat_coef) {
            n_bytes += sizeof(size_t);
            n_bytes += sizeof(double) * vec.size();
        }
    }
    n_bytes += sizeof(int)*node.chosen_cat.size();
    n_bytes += sizeof(double)*node.fill_val.size();
    n_bytes += sizeof(double)*node.fill_new.size();
    return n_bytes;
}

template <class otype>
void serialize_node(const IsoHPlane &node, otype &out, std::vector<uint8_t> &buffer)
{
    if (interrupt_switch) return;

    double data_doubles[] = {
        node.split_point,
        node.score,
        node.range_low,
        node.range_high,
        node.remainder
    };
    write_bytes<double>((void*)data_doubles, (size_t)5, out);

    size_t data_sizets[] = {
        node.hplane_left,
        node.hplane_right,
        node.col_num.size(),
        node.col_type.size(),
        node.coef.size(),
        node.mean.size(),
        node.cat_coef.size(),
        node.chosen_cat.size(),
        node.fill_val.size(),
        node.fill_new.size()
    };
    write_bytes<size_t>((void*)data_sizets, (size_t)10, out);

    write_bytes<size_t>((void*)node.col_num.data(), node.col_num.size(), out);

    if (node.col_type.size()) {
        if (buffer.size() < node.col_type.size())
            buffer.resize((size_t)2 * node.col_type.size());
        for (size_t ix = 0; ix < node.col_type.size(); ix++)
            buffer[ix] = (uint8_t)node.col_type[ix];
        write_bytes<uint8_t>((void*)buffer.data(), node.col_type.size(), out);
    }

    write_bytes<double>((void*)node.coef.data(), node.coef.size(), out);

    write_bytes<double>((void*)node.mean.data(), node.mean.size(), out);

    if (node.cat_coef.size()) {
        size_t veclen;
        for (const auto &vec : node.cat_coef) {
            veclen = vec.size();
            write_bytes<size_t>((void*)&veclen, (size_t)1, out);
            write_bytes<double>((void*)vec.data(), vec.size(), out);
        }
    }
    
    write_bytes<int>((void*)node.chosen_cat.data(), node.chosen_cat.size(), out);

    write_bytes<double>((void*)node.fill_val.data(), node.fill_val.size(), out);

    write_bytes<double>((void*)node.fill_new.data(), node.fill_new.size(), out);
}

template <class itype>
void deserialize_node(IsoHPlane &node, itype &in, std::vector<uint8_t> &buffer)
{
    if (interrupt_switch) return;

    double data_doubles[5];
    read_bytes<double>((void*)data_doubles, (size_t)5, in);
    node.split_point = data_doubles[0];
    node.score = data_doubles[1];
    node.range_low = data_doubles[2];
    node.range_high = data_doubles[3];
    node.remainder = data_doubles[4];

    size_t data_sizets[10];
    read_bytes<size_t>((void*)data_sizets, (size_t)10, in);

    node.hplane_left = data_sizets[0];
    node.hplane_right = data_sizets[1];

    read_bytes<size_t>(node.col_num, data_sizets[2], in);

    if (data_sizets[3]) {
        node.col_type.resize(data_sizets[3]);
        node.col_type.shrink_to_fit();
        if (buffer.size() < data_sizets[3])
            buffer.resize((size_t)2 * data_sizets[3]);
        read_bytes<uint8_t>((void*)buffer.data(), data_sizets[3], in);
        for (size_t ix = 0; ix < data_sizets[3]; ix++)
            node.col_type[ix] = (ColType)buffer[ix];
    }

    read_bytes<double>(node.coef, data_sizets[4], in);

    read_bytes<double>(node.mean, data_sizets[5], in);

    if (data_sizets[6]) {
        node.cat_coef.resize(data_sizets[6]);
        node.cat_coef.shrink_to_fit();
        size_t veclen;
        for (auto &vec : node.cat_coef) {
            read_bytes<size_t>((void*)&veclen, (size_t)1, in);
            read_bytes<double>(vec, veclen, in);
        }
    }

    read_bytes<int>(node.chosen_cat, data_sizets[7], in);

    read_bytes<double>(node.fill_val, data_sizets[8], in);

    read_bytes<double>(node.fill_new, data_sizets[9], in);
}

template <class itype, class saved_int_t, class saved_size_t>
void deserialize_node(IsoHPlane &node, itype &in, std::vector<uint8_t> &buffer, std::vector<char> &buffer2, const bool diff_endian)
{
    if (interrupt_switch) return;

    double data_doubles[5];
    read_bytes<double, double>((void*)data_doubles, (size_t)5, in, buffer2, diff_endian);
    node.split_point = data_doubles[0];
    node.score = data_doubles[1];
    node.range_low = data_doubles[2];
    node.range_high = data_doubles[3];
    node.remainder = data_doubles[4];

    size_t data_sizets[10];
    read_bytes<size_t, saved_size_t>((void*)data_sizets, (size_t)10, in, buffer2, diff_endian);

    node.hplane_left = data_sizets[0];
    node.hplane_right = data_sizets[1];

    read_bytes<size_t, saved_size_t>(node.col_num, data_sizets[2], in, buffer2, diff_endian);

    if (data_sizets[3]) {
        node.col_type.resize(data_sizets[3]);
        node.col_type.shrink_to_fit();
        if (buffer.size() < data_sizets[3])
            buffer.resize((size_t)2 * data_sizets[3]);
        read_bytes<uint8_t>((void*)buffer.data(), data_sizets[3], in);
        for (size_t ix = 0; ix < data_sizets[3]; ix++)
            node.col_type[ix] = (ColType)buffer[ix];
    }

    read_bytes<double, double>(node.coef, data_sizets[4], in, buffer2, diff_endian);

    read_bytes<double, double>(node.mean, data_sizets[5], in, buffer2, diff_endian);

    if (data_sizets[6]) {
        node.cat_coef.resize(data_sizets[6]);
        node.cat_coef.shrink_to_fit();
        size_t veclen;
        for (auto &vec : node.cat_coef) {
            read_bytes<size_t, saved_size_t>((void*)&veclen, (size_t)1, in, buffer2, diff_endian);
            read_bytes<double, double>(vec, veclen, in, buffer2, diff_endian);
        }
    }

    read_bytes<int, saved_int_t>(node.chosen_cat, data_sizets[7], in, buffer2, diff_endian);

    read_bytes<double, double>(node.fill_val, data_sizets[8], in, buffer2, diff_endian);

    read_bytes<double, double>(node.fill_new, data_sizets[9], in, buffer2, diff_endian);
}

size_t get_size_node(const ImputeNode &node)
{
    size_t n_bytes = 0;
    n_bytes += sizeof(size_t) * 5;
    n_bytes += sizeof(double) * node.num_sum.size();
    n_bytes += sizeof(double) * node.num_weight.size();
    if (node.cat_sum.size()) {
        for (const auto &v : node.cat_sum) {
            n_bytes += sizeof(size_t);
            n_bytes += sizeof(double) * v.size();
        }
    }
    n_bytes += sizeof(double) * node.cat_weight.size();
    return n_bytes;
}

template <class otype>
void serialize_node(const ImputeNode &node, otype &out)
{
    if (interrupt_switch) return;

    size_t data_sizets[] = {
        node.parent,
        node.num_sum.size(),
        node.num_weight.size(),
        node.cat_sum.size(),
        node.cat_weight.size(),
    };
    write_bytes<size_t>((void*)data_sizets, (size_t)5, out);

    write_bytes<double>((void*)node.num_sum.data(), node.num_sum.size(), out);

    write_bytes<double>((void*)node.num_weight.data(), node.num_weight.size(), out);

    if (node.cat_sum.size()) {
        size_t veclen;
        for (const auto &v : node.cat_sum) {
            veclen = v.size();
            write_bytes<size_t>((void*)&veclen, (size_t)1, out);
            write_bytes<double>((void*)v.data(), veclen, out);
        }
    }

    write_bytes<double>((void*)node.cat_weight.data(), node.cat_weight.size(), out);
}

template <class itype>
void deserialize_node(ImputeNode &node, itype &in)
{
    if (interrupt_switch) return;

    size_t data_sizets[5];
    read_bytes<size_t>((void*)data_sizets, (size_t)5, in);
    node.parent = data_sizets[0];

    read_bytes<double>(node.num_sum, data_sizets[1], in);

    read_bytes<double>(node.num_weight, data_sizets[2], in);

    node.cat_sum.resize(data_sizets[3]);
    if (data_sizets[3]) {
        size_t veclen;
        for (auto &v : node.cat_sum) {
            read_bytes<size_t>((void*)&veclen, (size_t)1, in);
            read_bytes<double>(v, veclen, in);
        }
    }
    node.cat_sum.shrink_to_fit();

    read_bytes<double>(node.cat_weight, data_sizets[4], in);
}

template <class itype, class saved_int_t, class saved_size_t>
void deserialize_node(ImputeNode &node, itype &in, std::vector<char> &buffer, const bool diff_endian)
{
    if (interrupt_switch) return;

    size_t data_sizets[5];
    read_bytes<size_t, saved_size_t>((void*)data_sizets, (size_t)5, in, buffer, diff_endian);
    node.parent = data_sizets[0];

    read_bytes<double, double>(node.num_sum, data_sizets[1], in, buffer, diff_endian);

    read_bytes<double, double>(node.num_weight, data_sizets[2], in, buffer, diff_endian);

    node.cat_sum.resize(data_sizets[3]);
    if (data_sizets[3]) {
        size_t veclen;
        for (auto &v : node.cat_sum) {
            read_bytes<size_t, saved_size_t>((void*)&veclen, (size_t)1, in, buffer, diff_endian);
            read_bytes<double, double>(v, veclen, in, buffer, diff_endian);
        }
    }
    node.cat_sum.shrink_to_fit();

    read_bytes<double, double>(node.cat_weight, data_sizets[4], in, buffer, diff_endian);
}

size_t get_size_model(const IsoForest &model)
{
    size_t n_bytes = 0;
    n_bytes += sizeof(uint8_t) * 3;
    n_bytes += sizeof(double) * 2;
    n_bytes += sizeof(size_t) * 2;
    for (const auto &tree : model.trees) {
        n_bytes += sizeof(size_t);
        for (const auto &node : tree)
            n_bytes += get_size_node(node);
    }
    return n_bytes;
}

template <class otype>
void serialize_model(const IsoForest &model, otype &out)
{
    if (interrupt_switch) return;

    uint8_t data_en[] = {
        (uint8_t)model.new_cat_action,
        (uint8_t)model.cat_split_type,
        (uint8_t)model.missing_action
    };
    write_bytes<uint8_t>((void*)data_en, (size_t)3, out);

    double data_doubles[] = {
        model.exp_avg_depth,
        model.exp_avg_sep
    };
    write_bytes<double>((void*)data_doubles, (size_t)2, out);

    size_t data_sizets[] = {
        model.orig_sample_size,
        model.trees.size()
    };
    write_bytes<size_t>((void*)data_sizets, (size_t)2, out);

    size_t veclen;
    for (const auto &tree : model.trees) {
        veclen = tree.size();
        write_bytes<size_t>((void*)&veclen, (size_t)1, out);
        for (const auto &node : tree)
            serialize_node(node, out);
    }
}

template <class itype>
void deserialize_model(IsoForest &model, itype &in)
{
    if (interrupt_switch) return;

    uint8_t data_en[3];
    read_bytes<uint8_t>((void*)data_en, (size_t)3, in);
    model.new_cat_action = (NewCategAction)data_en[0];
    model.cat_split_type = (CategSplit)data_en[1];
    model.missing_action = (MissingAction)data_en[2];

    double data_doubles[2];
    read_bytes<double>((void*)data_doubles, (size_t)2, in);
    model.exp_avg_depth = data_doubles[0];
    model.exp_avg_sep = data_doubles[1];

    size_t data_sizets[2];
    read_bytes<size_t>((void*)data_sizets, (size_t)2, in);
    model.orig_sample_size = data_sizets[0];
    model.trees.resize(data_sizets[1]);
    model.trees.shrink_to_fit();

    size_t veclen;
    for (auto &tree : model.trees) {
        read_bytes<size_t>((void*)&veclen, (size_t)1, in);
        tree.resize(veclen);
        tree.shrink_to_fit();
        for (auto &node : tree)
            deserialize_node(node, in);
    }
}

template <class itype, class saved_int_t, class saved_size_t>
void deserialize_model(IsoForest &model, itype &in, std::vector<char> &buffer, const bool diff_endian)
{
    if (interrupt_switch) return;

    uint8_t data_en[3];
    read_bytes<uint8_t>((void*)data_en, (size_t)3, in);
    model.new_cat_action = (NewCategAction)data_en[0];
    model.cat_split_type = (CategSplit)data_en[1];
    model.missing_action = (MissingAction)data_en[2];

    double data_doubles[2];
    read_bytes<double, double>((void*)data_doubles, (size_t)2, in, buffer, diff_endian);
    model.exp_avg_depth = data_doubles[0];
    model.exp_avg_sep = data_doubles[1];

    size_t data_sizets[2];
    read_bytes<size_t, saved_size_t>((void*)data_sizets, (size_t)2, in, buffer, diff_endian);
    model.orig_sample_size = data_sizets[0];
    model.trees.resize(data_sizets[1]);
    model.trees.shrink_to_fit();

    size_t veclen;
    for (auto &tree : model.trees) {
        read_bytes<size_t, saved_size_t>((void*)&veclen, (size_t)1, in, buffer, diff_endian);
        tree.resize(veclen);
        tree.shrink_to_fit();
        for (auto &node : tree)
            deserialize_node<itype, saved_int_t, saved_size_t>(node, in, buffer, diff_endian);
    }
}

template <class otype>
void serialize_additional_trees(const IsoForest &model, otype &out, size_t trees_prev)
{
    size_t veclen;
    for (size_t ix = trees_prev; ix < model.trees.size(); ix++) {
        veclen = model.trees[ix].size();
        write_bytes<size_t>((void*)&veclen, (size_t)1, out);
        for (const auto &node : model.trees[ix])
            serialize_node(node, out);
    }
}

size_t determine_serialized_size_additional_trees(const IsoForest &model, size_t old_ntrees)
{
    size_t n_bytes = 0;
    for (size_t ix = 0; ix < model.trees.size(); ix++) {
        n_bytes += sizeof(size_t);
        for (const auto &node : model.trees[ix])
            n_bytes += get_size_node(node);
    }
    return n_bytes;
}

size_t get_size_model(const ExtIsoForest &model)
{
    size_t n_bytes = 0;
    n_bytes += sizeof(uint8_t) * 3;
    n_bytes += sizeof(double) * 2;
    n_bytes += sizeof(size_t) * 2;
    for (const auto &tree : model.hplanes) {
        n_bytes += sizeof(size_t);
        for (const auto &node : tree)
            n_bytes += get_size_node(node);
    }
    return n_bytes;
}

template <class otype>
void serialize_model(const ExtIsoForest &model, otype &out)
{
    if (interrupt_switch) return;

    uint8_t data_en[] = {
        (uint8_t)model.new_cat_action,
        (uint8_t)model.cat_split_type,
        (uint8_t)model.missing_action
    };
    write_bytes<uint8_t>((void*)data_en, (size_t)3, out);

    double data_doubles[] = {
        model.exp_avg_depth,
        model.exp_avg_sep
    };
    write_bytes<double>((void*)data_doubles, (size_t)2, out);

    size_t data_sizets[] = {
        model.orig_sample_size,
        model.hplanes.size()
    };
    write_bytes<size_t>((void*)data_sizets, (size_t)2, out);

    std::vector<uint8_t> buffer;
    size_t veclen;
    for (const auto &tree : model.hplanes) {
        veclen = tree.size();
        write_bytes<size_t>((void*)&veclen, (size_t)1, out);
        for (const auto &node : tree)
            serialize_node(node, out, buffer);
    }
}

template <class itype>
void deserialize_model(ExtIsoForest &model, itype &in)
{
    if (interrupt_switch) return;

    uint8_t data_en[3];
    read_bytes<uint8_t>((void*)data_en, (size_t)3, in);
    model.new_cat_action = (NewCategAction)data_en[0];
    model.cat_split_type = (CategSplit)data_en[1];
    model.missing_action = (MissingAction)data_en[2];

    double data_doubles[2];
    read_bytes<double>((void*)data_doubles, (size_t)2, in);
    model.exp_avg_depth = data_doubles[0];
    model.exp_avg_sep = data_doubles[1];

    size_t data_sizets[2];
    read_bytes<size_t>((void*)data_sizets, (size_t)2, in);
    model.orig_sample_size = data_sizets[0];
    model.hplanes.resize(data_sizets[1]);
    model.hplanes.shrink_to_fit();

    size_t veclen;
    std::vector<uint8_t> buffer;
    for (auto &tree : model.hplanes) {
        read_bytes<size_t>((void*)&veclen, (size_t)1, in);
        tree.resize(veclen);
        tree.shrink_to_fit();
        for (auto &node : tree)
            deserialize_node(node, in, buffer);
    }
}

template <class itype, class saved_int_t, class saved_size_t>
void deserialize_model(ExtIsoForest &model, itype &in, std::vector<char> &buffer, const bool diff_endian)
{
    if (interrupt_switch) return;

    uint8_t data_en[3];
    read_bytes<uint8_t>((void*)data_en, (size_t)3, in);
    model.new_cat_action = (NewCategAction)data_en[0];
    model.cat_split_type = (CategSplit)data_en[1];
    model.missing_action = (MissingAction)data_en[2];

    double data_doubles[2];
    read_bytes<double, double>((void*)data_doubles, (size_t)2, in, buffer, diff_endian);
    model.exp_avg_depth = data_doubles[0];
    model.exp_avg_sep = data_doubles[1];

    size_t data_sizets[2];
    read_bytes<size_t, saved_size_t>((void*)data_sizets, (size_t)2, in, buffer, diff_endian);
    model.orig_sample_size = data_sizets[0];
    model.hplanes.resize(data_sizets[1]);
    model.hplanes.shrink_to_fit();

    size_t veclen;
    std::vector<uint8_t> buffer_;
    for (auto &tree : model.hplanes) {
        read_bytes<size_t, saved_size_t>((void*)&veclen, (size_t)1, in, buffer, diff_endian);
        tree.resize(veclen);
        tree.shrink_to_fit();
        for (auto &node : tree)
            deserialize_node<itype, saved_int_t, saved_size_t>(node, in, buffer_, buffer, diff_endian);
    }
}

template <class otype>
void serialize_additional_trees(const ExtIsoForest &model, otype &out, size_t trees_prev)
{
    if (interrupt_switch) return;

    std::vector<uint8_t> buffer;
    size_t veclen;
    for (size_t ix = trees_prev; ix < model.hplanes.size(); ix++) {
        veclen = model.hplanes[ix].size();
        write_bytes<size_t>((void*)&veclen, (size_t)1, out);
        for (const auto &node : model.hplanes[ix])
            serialize_node(node, out, buffer);
    }
}

size_t determine_serialized_size_additional_trees(const ExtIsoForest &model, size_t old_ntrees)
{
    size_t n_bytes = 0;
    for (size_t ix = 0; ix < model.hplanes.size(); ix++) {
        n_bytes += sizeof(size_t);
        for (const auto &node : model.hplanes[ix])
            n_bytes += get_size_node(node);
    }
    return n_bytes;
}

size_t get_size_model(const Imputer &model)
{
    size_t n_bytes = 0;
    n_bytes += sizeof(size_t) * 6;
    n_bytes += sizeof(int) * model.ncat.size();
    n_bytes += sizeof(double) * model.col_means.size();
    n_bytes += sizeof(int) * model.col_modes.size();
    for (const auto &tree : model.imputer_tree) {
        n_bytes += sizeof(size_t);
        for (const auto &node : tree)
            n_bytes += get_size_node(node);
    }
    return n_bytes;
}

template <class otype>
void serialize_model(const Imputer &model, otype &out)
{
    if (interrupt_switch) return;

    size_t data_sizets[] = {
        model.ncols_numeric,
        model.ncols_categ,
        model.ncat.size(),
        model.imputer_tree.size(),
        model.col_means.size(),
        model.col_modes.size()
    };
    write_bytes<size_t>((void*)data_sizets, (size_t)6, out);

    write_bytes<int>((void*)model.ncat.data(), model.ncat.size(), out);

    write_bytes<double>((void*)model.col_means.data(), model.col_means.size(), out);

    write_bytes<int>((void*)model.col_modes.data(), model.col_modes.size(), out);

    size_t veclen;
    for (const auto &tree : model.imputer_tree) {
        veclen = tree.size();
        write_bytes<size_t>((void*)&veclen, (size_t)1, out);
        for (const auto &node : tree)
            serialize_node(node, out);
    }
}

template <class itype>
void deserialize_model(Imputer &model, itype &in)
{
    if (interrupt_switch) return;

    size_t data_sizets[6];
    read_bytes<size_t>((void*)data_sizets, (size_t)6, in);
    model.ncols_numeric = data_sizets[0];
    model.ncols_categ = data_sizets[1];
    model.ncat.resize(data_sizets[2]);
    model.imputer_tree.resize(data_sizets[3]);
    model.col_means.resize(data_sizets[4]);
    model.col_modes.resize(data_sizets[5]);

    model.ncat.shrink_to_fit();
    model.imputer_tree.shrink_to_fit();
    model.col_means.shrink_to_fit();
    model.col_modes.shrink_to_fit();

    read_bytes<int>(model.ncat, model.ncat.size(), in);

    read_bytes<double>(model.col_means, model.col_means.size(), in);

    read_bytes<int>(model.col_modes, model.col_modes.size(), in);

    size_t veclen;
    for (auto &tree : model.imputer_tree) {
        read_bytes<size_t>((void*)&veclen, (size_t)1, in);
        tree.resize(veclen);
        tree.shrink_to_fit();
        for (auto &node : tree)
            deserialize_node(node, in);
    }
}

template <class itype, class saved_int_t, class saved_size_t>
void deserialize_model(Imputer &model, itype &in, std::vector<char> &buffer, const bool diff_endian)
{
    if (interrupt_switch) return;

    size_t data_sizets[6];
    read_bytes<size_t, saved_size_t>((void*)data_sizets, (size_t)6, in, buffer, diff_endian);
    model.ncols_numeric = data_sizets[0];
    model.ncols_categ = data_sizets[1];
    model.ncat.resize(data_sizets[2]);
    model.imputer_tree.resize(data_sizets[3]);
    model.col_means.resize(data_sizets[4]);
    model.col_modes.resize(data_sizets[5]);

    model.ncat.shrink_to_fit();
    model.imputer_tree.shrink_to_fit();
    model.col_means.shrink_to_fit();
    model.col_modes.shrink_to_fit();

    read_bytes<int, saved_int_t>(model.ncat, model.ncat.size(), in, buffer, diff_endian);

    read_bytes<double, double>(model.col_means, model.col_means.size(), in, buffer, diff_endian);

    read_bytes<int, saved_int_t>(model.col_modes, model.col_modes.size(), in, buffer, diff_endian);

    size_t veclen;
    for (auto &tree : model.imputer_tree) {
        read_bytes<size_t, saved_size_t>((void*)&veclen, (size_t)1, in, buffer, diff_endian);
        tree.resize(veclen);
        tree.shrink_to_fit();
        for (auto &node : tree)
            deserialize_node<itype, saved_int_t, saved_size_t>(node, in, buffer, diff_endian);
    }
}

template <class otype>
void serialize_additional_trees(const Imputer &model, otype &out, size_t trees_prev)
{
    size_t veclen;
    for (size_t ix = trees_prev; ix < model.imputer_tree.size(); ix++) {
        veclen = model.imputer_tree[ix].size();
        write_bytes<size_t>((void*)&veclen, (size_t)1, out);
        for (const auto &node : model.imputer_tree[ix])
            serialize_node(node, out);
    }
}

size_t determine_serialized_size_additional_trees(const Imputer &model, size_t old_ntrees)
{
    size_t n_bytes = 0;
    for (size_t ix = 0; ix < model.imputer_tree.size(); ix++) {
        n_bytes += sizeof(size_t);
        for (const auto &node : model.imputer_tree[ix])
            n_bytes += get_size_node(node);
    }
    return n_bytes;
}

bool get_is_little_endian()
{
    const int one = 1;
    return *((unsigned char*)&one) != 0;
}

size_t get_size_setup_info()
{
    size_t n_bytes = 0;
    n_bytes += sizeof(unsigned char) * SIZE_WATERMARK;
    n_bytes += sizeof(uint8_t) * 9;
    return n_bytes;
}

template <class otype>
void add_setup_info(otype &out, bool full_watermark)
{
    write_bytes<unsigned char>((void*)(full_watermark? watermark: incomplete_watermark), SIZE_WATERMARK, out);
    /*
    0 : endianness
    1-3: isotree version
    4: double type
    5: size_t limit
    6: sizeof(int)
    7: sizeof(size_t)
    8: sizeof(double)
    */
    uint8_t setup_info[] = {
        (uint8_t)get_is_little_endian(),
        (uint8_t)ISOTREE_VERSION_MAJOR,
        (uint8_t)ISOTREE_VERSION_MINOR,
        (uint8_t)ISOTREE_VERSION_PATCH,
        #if defined(HAS_NORMAL_DOUBLE)
        (uint8_t)IsNormalDouble,
        #else
        (uint8_t)IsAbnormalDouble,
        #endif
        #if SIZE_MAX == UINT32_MAX
        (uint8_t)Is32Bit,
        #elif SIZE_MAX == UINT64_MAX
        (uint8_t)Is64Bit,
        #else
        (uint8_t)IsOther,
        #endif
        (uint8_t)sizeof(int),
        (uint8_t)sizeof(size_t),
        (uint8_t)sizeof(double)
    };
    write_bytes<uint8_t>((void*)setup_info, (size_t)9, out);
}

template <class otype>
void add_full_watermark(otype &out)
{
    write_bytes<unsigned char>((void*)watermark, SIZE_WATERMARK, out);
}

template <class itype>
void check_setup_info
(
    itype &in,
    bool &has_watermark,
    bool &has_incomplete_watermark,
    bool &has_same_double,
    bool &has_same_int_size,
    bool &has_same_size_t_size,
    bool &has_same_endianness,
    PlatformSize &saved_int_t,
    PlatformSize &saved_size_t,
    PlatformEndianness &saved_endian,
    bool &is_deserializable
)
{
    is_deserializable = false;
    has_incomplete_watermark = false;

    unsigned char watermark_in[SIZE_WATERMARK];
    read_bytes<unsigned char>((void*)watermark_in, SIZE_WATERMARK, in);
    if (memcmp(watermark_in, (unsigned char*)watermark, SIZE_WATERMARK)) {
        has_watermark = false;
        if (!memcmp(watermark_in, (unsigned char*)incomplete_watermark, SIZE_WATERMARK))
            has_incomplete_watermark = true;
        return;
    }
    else {
        has_watermark = true;
    }

    uint8_t setup_info[9];
    read_bytes<uint8_t>((void*)setup_info, (size_t)9, in);

    bool is_little_endian = get_is_little_endian();
    if ((bool)is_little_endian != (bool)setup_info[0]) {
        has_same_endianness = false;
        saved_endian = is_little_endian? PlatformLittleEndian : PlatformBigEndian;
    }
    else {
        has_same_endianness = true;
    }

    if (setup_info[4] == (uint8_t)IsAbnormalDouble)
        fprintf(stderr, "Warning: input model uses non-standard numeric type, might read correctly.\n");
    
    switch(setup_info[6])
    {
        case 16: {saved_int_t = Is16Bit; break;}
        case 32: {saved_int_t = Is32Bit; break;}
        case 64: {saved_int_t = Is64Bit; break;}
        default: {saved_int_t = IsOther; break;}
    }
    if ((uint8_t)sizeof(int) != setup_info[6]) {
        has_same_int_size = false;
        if (sizeof(uint8_t) != 1) return;
        if (saved_int_t == IsOther) return;
    }
    else {
        has_same_int_size = true;
    }


    if ((uint8_t)sizeof(size_t) != setup_info[7]) {
        has_same_size_t_size = false;
        if (sizeof(uint8_t) != 1) return;
    }
    else {
        has_same_size_t_size = true;
    }


    if ((uint8_t)sizeof(double) != setup_info[8]) {
        has_same_double = false;
        return;
    }
    else {
        has_same_double = true;
    }

    saved_size_t = (PlatformSize)setup_info[5];
    #if SIZE_MAX == UINT32_MAX
    if (setup_info[5] != (uint8_t)Is32Bit)
    #elif SIZE_MAX == UINT64_MAX
    if (setup_info[5] != (uint8_t)Is64Bit)
    #else
    if (setup_info[5] != (uint8_t)IsOther)
    #endif
    {
        has_same_size_t_size = false;
        if (saved_size_t == IsOther)
            return;
    }

    else {
        has_same_size_t_size = true;
    }

    is_deserializable = true;
}

template <class itype>
void check_setup_info(itype &in)
{
    bool has_watermark = false;
    bool has_incomplete_watermark = false;
    bool has_same_double = false;
    bool has_same_int_size = false;
    bool has_same_size_t_size = false;
    bool has_same_endianness = false;
    PlatformSize saved_int_t;
    PlatformSize saved_size_t;
    PlatformEndianness saved_endian;
    bool is_deserializable = false;

    check_setup_info(
        in,
        has_watermark,
        has_incomplete_watermark,
        has_same_double,
        has_same_int_size,
        has_same_size_t_size,
        has_same_endianness,
        saved_int_t,
        saved_size_t,
        saved_endian,
        is_deserializable
    );

    if (!has_watermark) {
        if (has_incomplete_watermark)
            throw std::runtime_error("Error: serialized model is incomplete.\n");
        else
            throw std::runtime_error("Error: input is not an isotree model.\n");
    }
    if (!has_same_double)
        throw std::runtime_error("Error: input model was saved in a machine with different 'double' type.\n");
    if (!has_same_int_size)
        throw std::runtime_error("Error: input model was saved in a machine with different integer type.\n");
    if (!has_same_size_t_size)
        throw std::runtime_error("Error: input model was saved in a machine with different 'size_t' type.\n");
    if (!has_same_endianness)
        throw std::runtime_error("Error: input model was saved in a machine with different endianness.\n");
}

template <class itype>
void check_setup_info
(
    itype &in,
    bool &has_same_int_size,
    bool &has_same_size_t_size,
    bool &has_same_endianness,
    PlatformSize &saved_int_t,
    PlatformSize &saved_size_t,
    PlatformEndianness &saved_endian
)
{
    bool has_watermark = false;
    bool has_incomplete_watermark = false;
    bool has_same_double = false;
    bool is_deserializable = false;

    check_setup_info(
        in,
        has_watermark,
        has_incomplete_watermark,
        has_same_double,
        has_same_int_size,
        has_same_size_t_size,
        has_same_endianness,
        saved_int_t,
        saved_size_t,
        saved_endian,
        is_deserializable
    );

    if (!has_watermark) {
        if (has_incomplete_watermark)
            throw std::runtime_error("Error: serialized model is incomplete.\n");
        else
            throw std::runtime_error("Error: input is not an isotree model.\n");
    }
    if (!has_same_double)
        throw std::runtime_error("Error: input model was saved in a machine with different 'double' type.\n");
    if (!is_deserializable)
        throw std::runtime_error("Error: input format is incompatible.\n");
}

size_t get_size_ending_metadata()
{
    size_t n_bytes = 0;
    n_bytes += sizeof(uint8_t);
    n_bytes += sizeof(size_t);
    return n_bytes;
}

template <class Model>
size_t determine_serialized_size(const Model &model)
{
    size_t n_bytes = 0;
    n_bytes += get_size_setup_info();
    n_bytes += sizeof(uint8_t);
    n_bytes += sizeof(size_t);
    n_bytes += get_size_model(model);
    n_bytes += get_size_ending_metadata();
    return n_bytes;
}

uint8_t get_model_code(const IsoForest &model)
{
    return IsoForestModel;
}

uint8_t get_model_code(const ExtIsoForest &model)
{
    return ExtIsoForestModel;
}

uint8_t get_model_code(const Imputer &model)
{
    return ImputerModel;
}

template <class Model, class otype>
void serialization_pipeline(const Model &model, otype &out)
{
    SignalSwitcher ss = SignalSwitcher();

    auto pos_watermark = set_return_position(out);
    
    add_setup_info(out, false);
    uint8_t model_type = get_model_code(model);
    write_bytes<uint8_t>((void*)&model_type, (size_t)1, out);
    size_t size_model = get_size_model(model);
    write_bytes<size_t>((void*)&size_model, (size_t)1, out);
    serialize_model(model, out);
    check_interrupt_switch(ss);

    /* This last bit will be left open in order to signal if anything follows,
       in case it's decided to change the format in the future or to add
       something additional, along with a 'size_t' slot in case it would need
       to jump ahead or something like that. */
    uint8_t ending_type = (uint8_t)EndsHere;
    write_bytes<uint8_t>((void*)&ending_type, (size_t)1, out);
    size_t jump_ahead = 0;
    write_bytes<size_t>((void*)&jump_ahead, (size_t)1, out);

    auto end_pos = set_return_position(out);
    return_to_position(out, pos_watermark);
    add_full_watermark(out);
    return_to_position(out, end_pos);
}

template <class Model, class itype>
void deserialization_pipeline(Model &model, itype &in)
{
    SignalSwitcher ss = SignalSwitcher();

    bool has_same_int_size;
    bool has_same_size_t_size;
    bool has_same_endianness;
    PlatformSize saved_int_t;
    PlatformSize saved_size_t;
    PlatformEndianness saved_endian;

    check_setup_info(
        in,
        has_same_int_size,
        has_same_size_t_size,
        has_same_endianness,
        saved_int_t,
        saved_size_t,
        saved_endian
    );

    uint8_t model_type = get_model_code(model);
    uint8_t model_in;
    read_bytes<uint8_t>((void*)&model_in, (size_t)1, in);
    if (model_type != model_in)
        throw std::runtime_error("Object to de-serialize does not match with the supplied type.\n");

    size_t size_model;
    if (has_same_int_size && has_same_size_t_size && has_same_endianness)
    {
        read_bytes<size_t>((void*)&size_model, (size_t)1, in);
        deserialize_model(model, in);
    }
    
    else
    {
        std::vector<char> buffer;
        const bool diff_endian = !has_same_endianness;

        if (saved_int_t == Is16Bit && saved_size_t == Is32Bit)
        {
            read_bytes<size_t, uint32_t>((void*)&size_model, (size_t)1, in, buffer, diff_endian);
            deserialize_model<itype, int16_t, uint32_t>(model, in, buffer, diff_endian);
        }

        else if (saved_int_t == Is32Bit && saved_size_t == Is32Bit)
        {
            read_bytes<size_t, uint32_t>((void*)&size_model, (size_t)1, in, buffer, diff_endian);
            deserialize_model<itype, int32_t, uint32_t>(model, in, buffer, diff_endian);
        }

        else if (saved_int_t == Is64Bit && saved_size_t == Is32Bit)
        {
            read_bytes<size_t, uint32_t>((void*)&size_model, (size_t)1, in, buffer, diff_endian);
            deserialize_model<itype, int64_t, uint32_t>(model, in, buffer, diff_endian);
        }

        else if (saved_int_t == Is16Bit && saved_size_t == Is64Bit)
        {
            read_bytes<size_t, uint64_t>((void*)&size_model, (size_t)1, in, buffer, diff_endian);
            deserialize_model<itype, int16_t, uint64_t>(model, in, buffer, diff_endian);
        }

        else if (saved_int_t == Is32Bit && saved_size_t == Is64Bit)
        {
            read_bytes<size_t, uint64_t>((void*)&size_model, (size_t)1, in, buffer, diff_endian);
            deserialize_model<itype, int32_t, uint64_t>(model, in, buffer, diff_endian);
        }

        else if (saved_int_t == Is64Bit && saved_size_t == Is64Bit)
        {
            read_bytes<size_t, uint64_t>((void*)&size_model, (size_t)1, in, buffer, diff_endian);
            deserialize_model<itype, int64_t, uint64_t>(model, in, buffer, diff_endian);
        }

        else
        {
            throw std::runtime_error("Unexpected error.\n");
        }
    }

    check_interrupt_switch(ss);

    /* Not currently used, but left in case the format changes */
    uint8_t ending_type;
    read_bytes<uint8_t>((void*)&ending_type, (size_t)1, in);
    size_t jump_ahead;
    read_bytes<size_t>((void*)&jump_ahead, (size_t)1, in);
}

void re_serialization_pipeline(const IsoForest &model, char *&out)
{
    SignalSwitcher ss = SignalSwitcher();

    check_setup_info(out);
    
    uint8_t model_in;
    memcpy(&model_in, out, sizeof(uint8_t));
    out += sizeof(uint8_t);
    if (model_in != get_model_code(model))
        throw std::runtime_error("Object to incrementally-serialize does not match with the supplied type.\n");

    char *pos_size = out;
    size_t old_size;
    memcpy(&old_size, out, sizeof(size_t));
    out += sizeof(size_t);
    
    char *old_end = out + old_size;
    uint8_t old_ending_type;
    memcpy(&old_ending_type, old_end, sizeof(uint8_t));
    size_t old_jump_ahead;
    memcpy(&old_jump_ahead, old_end + sizeof(uint8_t), sizeof(size_t));

    size_t new_size = get_size_model(model);
    size_t new_ntrees = model.trees.size();

    try
    {
        out += sizeof(uint8_t) * 3;
        out += sizeof(double) * 2;
        out += sizeof(size_t);
        
        char *pos_ntrees = out;
        size_t old_ntrees;
        memcpy(&old_ntrees, out, sizeof(size_t));
        
        serialize_additional_trees(model, old_end, old_ntrees);

        out = old_end;
        uint8_t ending_type = (uint8_t)EndsHere;
        memcpy(out, &ending_type, sizeof(uint8_t));
        out += sizeof(uint8_t);
        size_t jump_ahead = 0;
        memcpy(out, &jump_ahead, sizeof(size_t));
        out += sizeof(size_t);

        /* Leave this for the end in case something fails, so as not to
           render the serialized bytes unusable. */
        memcpy(pos_size, &new_size, sizeof(size_t));
        memcpy(pos_ntrees, &new_ntrees, sizeof(size_t));
    }

    catch(...)
    {
        memcpy(out, &old_ending_type, sizeof(uint8_t));
        memcpy(out + sizeof(uint8_t), &old_jump_ahead, sizeof(size_t));
        throw;
    }

    check_interrupt_switch(ss);
}

void re_serialization_pipeline(const ExtIsoForest &model, char *&out)
{
    SignalSwitcher ss = SignalSwitcher();

    check_setup_info(out);

    uint8_t model_in;
    memcpy(&model_in, out, sizeof(uint8_t));
    out += sizeof(uint8_t);
    if (model_in != get_model_code(model))
        throw std::runtime_error("Object to incrementally-serialize does not match with the supplied type.\n");

    char *pos_size = out;
    size_t old_size;
    memcpy(&old_size, out, sizeof(size_t));
    out += sizeof(size_t);
    
    char *old_end = out + old_size;
    uint8_t old_ending_type;
    memcpy(&old_ending_type, old_end, sizeof(uint8_t));
    size_t old_jump_ahead;
    memcpy(&old_jump_ahead, old_end + sizeof(uint8_t), sizeof(size_t));

    size_t new_size = get_size_model(model);
    size_t new_ntrees = model.hplanes.size();

    try
    {
        out += sizeof(uint8_t) * 3;
        out += sizeof(double) * 2;
        out += sizeof(size_t);
        char *pos_ntrees = out;
        size_t old_ntrees;
        memcpy(&old_ntrees, out, sizeof(size_t));
        out += sizeof(size_t);
        
        serialize_additional_trees(model, old_end, old_ntrees);

        out = old_end;
        uint8_t ending_type = (uint8_t)EndsHere;
        memcpy(out, &ending_type, sizeof(uint8_t));
        out += sizeof(uint8_t);
        size_t jump_ahead = 0;
        memcpy(out, &jump_ahead, sizeof(size_t));
        out += sizeof(size_t);

        /* Leave this for the end in case something fails, so as not to
           render the serialized bytes unusable. */
        memcpy(pos_size, &new_size, sizeof(size_t));
        memcpy(pos_ntrees, &new_ntrees, sizeof(size_t));
    }

    catch(...)
    {
        memcpy(out, &old_ending_type, sizeof(uint8_t));
        memcpy(out + sizeof(uint8_t), &old_jump_ahead, sizeof(size_t));
        throw;
    }

    check_interrupt_switch(ss);
}

void re_serialization_pipeline(const Imputer &model, char *&out)
{
    SignalSwitcher ss = SignalSwitcher();

    check_setup_info(out);

    uint8_t model_in;
    memcpy(&model_in, out, sizeof(uint8_t));
    out += sizeof(uint8_t);
    if (model_in != get_model_code(model))
        throw std::runtime_error("Object to incrementally-serialize does not match with the supplied type.\n");

    char *pos_size = out;
    size_t old_size;
    memcpy(&old_size, out, sizeof(size_t));
    out += sizeof(size_t);
    
    char *old_end = out + old_size;
    uint8_t old_ending_type;
    memcpy(&old_ending_type, old_end, sizeof(uint8_t));
    size_t old_jump_ahead;
    memcpy(&old_jump_ahead, old_end + sizeof(uint8_t), sizeof(size_t));

    size_t new_size = get_size_model(model);
    size_t new_ntrees = model.imputer_tree.size();

    try
    {
        out += sizeof(size_t) * 3;
        
        char *pos_ntrees = out;
        size_t old_ntrees;
        memcpy(&old_ntrees, out, sizeof(size_t));
        
        serialize_additional_trees(model, old_end, old_ntrees);

        out = old_end;
        uint8_t ending_type = (uint8_t)EndsHere;
        memcpy(out, &ending_type, sizeof(uint8_t));
        out += sizeof(uint8_t);
        size_t jump_ahead = 0;
        memcpy(out, &jump_ahead, sizeof(size_t));
        out += sizeof(size_t);

        /* Leave this for the end in case something fails, so as not to
           render the serialized bytes unusable. */
        memcpy(pos_size, &new_size, sizeof(size_t));
        memcpy(pos_ntrees, &new_ntrees, sizeof(size_t));
    }

    catch(...)
    {
        memcpy(out, &old_ending_type, sizeof(uint8_t));
        memcpy(out + sizeof(uint8_t), &old_jump_ahead, sizeof(size_t));
        throw;
    }

    check_interrupt_switch(ss);
}

void incremental_serialize_IsoForest(const IsoForest &model, char *old_bytes_reallocated)
{
    char *out = old_bytes_reallocated;
    re_serialization_pipeline(model, out);
}

void incremental_serialize_ExtIsoForest(const ExtIsoForest &model, char *old_bytes_reallocated)
{
    char *out = old_bytes_reallocated;
    re_serialization_pipeline(model, out);
}

void incremental_serialize_Imputer(const Imputer &model, char *old_bytes_reallocated)
{
    char *out = old_bytes_reallocated;
    re_serialization_pipeline(model, out);
}

template <class Model>
void incremental_serialize_string(const Model &model, std::string &old_bytes)
{
    size_t new_size = determine_serialized_size(model);
    if (old_bytes.size() > new_size)
        throw std::runtime_error("'old_bytes' is not a subset of 'model'.\n");
    if (!new_size)
        throw std::runtime_error("Unexpected error.\n");
    old_bytes.resize(new_size);
    char *out = &old_bytes[0];
    re_serialization_pipeline(model, out);
}

void incremental_serialize_IsoForest(const IsoForest &model, std::string &old_bytes)
{
    incremental_serialize_string(model, old_bytes);
}

void incremental_serialize_ExtIsoForest(const ExtIsoForest &model, std::string &old_bytes)
{
    incremental_serialize_string(model, old_bytes);
}

void incremental_serialize_Imputer(const Imputer &model, std::string &old_bytes)
{
    incremental_serialize_string(model, old_bytes);
}

template <class Model>
std::string serialization_pipeline(const Model &model)
{
    std::string serialized;
    serialized.resize(get_size_model(model));
    char *ptr = &serialized[0];
    serialization_pipeline(model, ptr);
    return serialized;
}

template <class Model>
void serialization_pipeline_ToFile(const Model &model, const char *fname)
{
    FileHandle f(fname, "wb");
    serialization_pipeline(model, f.handle);
}

#ifdef WCHAR_T_FUNS
template <class Model>
void serialization_pipeline_ToFile(const Model &model, const wchar_t *fname)
{
    WFileHandle f(fname, L"wb");
    serialization_pipeline(model, f.handle);
}
#endif

size_t determine_serialized_size(const IsoForest &model)
{
    return determine_serialized_size<IsoForest>(model);
}

size_t determine_serialized_size(const ExtIsoForest &model)
{
    return determine_serialized_size<ExtIsoForest>(model);
}

size_t determine_serialized_size(const Imputer &model)
{
    return determine_serialized_size<Imputer>(model);
}

void serialize_IsoForest(const IsoForest &model, char *out)
{
    serialization_pipeline(model, out);
}

void serialize_IsoForest(const IsoForest &model, FILE *out)
{
    serialization_pipeline(model, out);
}

void serialize_IsoForest(const IsoForest &model, std::ostream &out)
{
    serialization_pipeline(model, out);
}

std::string serialize_IsoForest(const IsoForest &model)
{
    return serialization_pipeline(model);
}

void serialize_IsoForest_ToFile(const IsoForest &model, const char *fname)
{
    serialization_pipeline_ToFile(model, fname);
}

#ifdef WCHAR_T_FUNS
void serialize_IsoForest_ToFile(const IsoForest &model, const wchar_t *fname)
{
    serialization_pipeline_ToFile(model, fname);
}
#endif

void deserialize_IsoForest(IsoForest &model, const char *in)
{
    deserialization_pipeline(model, in);
}

void deserialize_IsoForest(IsoForest &model, FILE *in)
{
    deserialization_pipeline(model, in);
}

void deserialize_IsoForest(IsoForest &model, std::istream &in)
{
    deserialization_pipeline(model, in);
}

void deserialize_IsoForest(IsoForest &model, const std::string &in)
{
    if (!in.size())
        throw std::runtime_error("Invalid input model to deserialize.");
    const char *in_ = &in[0];
    deserialization_pipeline(model, in_);
}

void deserialize_IsoForest_FromFile(IsoForest &model, const char *fname)
{
    FileHandle f(fname, "rb");
    deserialize_IsoForest(model, f.handle);
}

#ifdef WCHAR_T_FUNS
void deserialize_IsoForest_FromFile(IsoForest &model, const wchar_t *fname)
{
    WFileHandle f(fname, L"rb");
    deserialize_IsoForest(model, f.handle);
}
#endif

void serialize_ExtIsoForest(const ExtIsoForest &model, char *out)
{
    serialization_pipeline(model, out);
}

void serialize_ExtIsoForest(const ExtIsoForest &model, FILE *out)
{
    serialization_pipeline(model, out);
}

void serialize_ExtIsoForest(const ExtIsoForest &model, std::ostream &out)
{
    serialization_pipeline(model, out);
}

std::string serialize_ExtIsoForest(const ExtIsoForest &model)
{
    return serialization_pipeline(model);
}

void serialize_ExtIsoForest_ToFile(const ExtIsoForest &model, const char *fname)
{
    serialization_pipeline_ToFile(model, fname);
}

#ifdef WCHAR_T_FUNS
void serialize_ExtIsoForest_ToFile(const ExtIsoForest &model, const wchar_t *fname)
{
    serialization_pipeline_ToFile(model, fname);
}
#endif

void deserialize_ExtIsoForest(ExtIsoForest &model, const char *in)
{
    deserialization_pipeline(model, in);
}

void deserialize_ExtIsoForest(ExtIsoForest &model, FILE *in)
{
    deserialization_pipeline(model, in);
}

void deserialize_ExtIsoForest(ExtIsoForest &model, std::istream &in)
{
    deserialization_pipeline(model, in);
}

void deserialize_ExtIsoForest(ExtIsoForest &model, const std::string &in)
{
    if (!in.size())
        throw std::runtime_error("Invalid input model to deserialize.");
    const char *in_ = &in[0];
    deserialization_pipeline(model, in_);
}

void deserialize_ExtIsoForest_FromFile(ExtIsoForest &model, const char *fname)
{
    FileHandle f(fname, "rb");
    deserialize_ExtIsoForest(model, f.handle);
}

#ifdef WCHAR_T_FUNS
void deserialize_ExtIsoForest_FromFile(ExtIsoForest &model, const wchar_t *fname)
{
    WFileHandle f(fname, L"rb");
    deserialize_ExtIsoForest(model, f.handle);
}
#endif

void serialize_Imputer(const Imputer &model, char *out)
{
    serialization_pipeline(model, out);
}

void serialize_Imputer(const Imputer &model, FILE *out)
{
    serialization_pipeline(model, out);
}

void serialize_Imputer(const Imputer &model, std::ostream &out)
{
    serialization_pipeline(model, out);
}

std::string serialize_Imputer(const Imputer &model)
{
    return serialization_pipeline(model);
}

void serialize_Imputer_ToFile(const Imputer &model, const char *fname)
{
    serialization_pipeline_ToFile(model, fname);
}

#ifdef WCHAR_T_FUNS
void serialize_Imputer_ToFile(const Imputer &model, const wchar_t *fname)
{
    serialization_pipeline_ToFile(model, fname);
}
#endif

void deserialize_Imputer(Imputer &model, const char *in)
{
    deserialization_pipeline(model, in);
}

void deserialize_Imputer(Imputer &model, FILE *in)
{
    deserialization_pipeline(model, in);
}

void deserialize_Imputer(Imputer &model, std::istream &in)
{
    deserialization_pipeline(model, in);
}

void deserialize_Imputer(Imputer &model, const std::string &in)
{
    if (!in.size())
        throw std::runtime_error("Invalid input model to deserialize.");
    const char *in_ = &in[0];
    deserialization_pipeline(model, in_);
}

void deserialize_Imputer_FromFile(Imputer &model, const char *fname)
{
    FileHandle f(fname, "rb");
    deserialize_Imputer(model, f.handle);
}

#ifdef WCHAR_T_FUNS
void deserialize_Imputer_FromFile(Imputer &model, const wchar_t *fname)
{
    WFileHandle f(fname, L"rb");
    deserialize_Imputer(model, f.handle);
}
#endif

/* Shorthands to use in templates (will be used in R) */
void serialize_isotree(const IsoForest &model, char *out)
{
    serialize_IsoForest(model, out);
}

void serialize_isotree(const ExtIsoForest &model, char *out)
{
    serialize_ExtIsoForest(model, out);
}

void serialize_isotree(const Imputer &model, char *out)
{
    serialize_Imputer(model, out);
}

void deserialize_isotree(IsoForest &model, const char *in)
{
    deserialize_IsoForest(model, in);
}

void deserialize_isotree(ExtIsoForest &model, const char *in)
{
    deserialize_ExtIsoForest(model, in);
}

void deserialize_isotree(Imputer &model, const char *in)
{
    deserialize_Imputer(model, in);
}

void incremental_serialize_isotree(const IsoForest &model, char *old_bytes_reallocated)
{
    incremental_serialize_IsoForest(model, old_bytes_reallocated);
}

void incremental_serialize_isotree(const ExtIsoForest &model, char *old_bytes_reallocated)
{
    incremental_serialize_ExtIsoForest(model, old_bytes_reallocated);
}

void incremental_serialize_isotree(const Imputer &model, char *old_bytes_reallocated)
{
    incremental_serialize_Imputer(model, old_bytes_reallocated);
}

template <class itype>
void read_bytes_size_t(void *ptr, const size_t n_els, itype &in, const PlatformSize saved_size_t, const bool has_same_endianness)
{
    std::vector<char> buffer;
    switch(saved_size_t)
    {
        case Is32Bit:
        {
            read_bytes<size_t, uint32_t>(ptr, n_els, in, buffer, !has_same_endianness);
            break;
        }

        case Is64Bit:
        {
            read_bytes<size_t, uint64_t>(ptr, n_els, in, buffer, !has_same_endianness);
            break;
        }

        default:
        {
            throw std::runtime_error("Unexpected error.\n");
        }
    }
}

template <class itype>
void inspect_serialized_object
(
    itype &serialized_bytes,
    bool &is_isotree_model,
    bool &is_compatible,
    bool &has_combined_objects,
    bool &has_IsoForest,
    bool &has_ExtIsoForest,
    bool &has_Imputer,
    bool &has_metadata,
    size_t &size_metadata,
    bool &has_same_int_size,
    bool &has_same_size_t_size,
    bool &has_same_endianness
)
{
    auto saved_position = set_return_position(serialized_bytes);

    is_isotree_model = false;
    is_compatible = false;
    has_combined_objects = false;
    has_IsoForest = false;
    has_ExtIsoForest = false;
    has_Imputer = false;
    has_metadata = false;
    size_metadata = 0;

    bool has_same_double = false;
    bool has_incomplete_watermark = false;
    PlatformSize saved_int_t;
    PlatformSize saved_size_t;
    PlatformEndianness saved_endian;
    check_setup_info(
        serialized_bytes,
        is_isotree_model,
        has_incomplete_watermark,
        has_same_double,
        has_same_int_size,
        has_same_size_t_size,
        has_same_endianness,
        saved_int_t,
        saved_size_t,
        saved_endian,
        is_compatible
    );

    if (!is_isotree_model || !is_compatible)
        return;

    uint8_t model_type;
    read_bytes<uint8_t>((void*)&model_type, (size_t)1, serialized_bytes);

    switch(model_type)
    {
        case IsoForestModel:
        {
            has_IsoForest = true;
            break;
        }

        case ExtIsoForestModel:
        {
            has_ExtIsoForest = true;
            break;
        }

        case ImputerModel:
        {
            has_Imputer = true;
            break;
        }

        case AllObjectsCombined:
        {
            has_combined_objects = true;
            break;
        }

        default:
        {

        }
    }

    if (has_combined_objects)
    {
        size_t size_model[3];

        read_bytes<uint8_t>((void*)&model_type, (size_t)1, serialized_bytes);
        switch(model_type)
        {
            case HasSingleVarModelNext:
            {
                has_IsoForest = true;
                break;
            }
            case HasExtModelNext:
            {
                has_ExtIsoForest = true;
                break;
            }
            case HasSingleVarModelPlusImputerNext:
            {
                has_IsoForest = true;
                has_Imputer = true;
                break;
            }
            case HasExtModelPlusImputerNext:
            {
                has_ExtIsoForest = true;
                has_Imputer = true;
                break;
            }
            case HasSingleVarModelPlusMetadataNext:
            {
                has_IsoForest = true;
                has_metadata = true;
                read_bytes_size_t(size_model, (size_t)3, serialized_bytes, saved_size_t, has_same_endianness);
                size_metadata = size_model[2];
                break;
            }
            case HasExtModelPlusMetadataNext:
            {
                has_ExtIsoForest = true;
                has_metadata = true;
                read_bytes_size_t(size_model, (size_t)3, serialized_bytes, saved_size_t, has_same_endianness);
                size_metadata = size_model[2];
                break;
            }
            case HasSingleVarModelPlusImputerPlusMetadataNext:
            {
                has_IsoForest = true;
                has_Imputer = true;
                has_metadata = true;
                read_bytes_size_t(size_model, (size_t)3, serialized_bytes, saved_size_t, has_same_endianness);
                size_metadata = size_model[2];
                break;
            }
            case HasExtModelPlusImputerPlusMetadataNext:
            {
                has_ExtIsoForest = true;
                has_Imputer = true;
                has_metadata = true;
                read_bytes_size_t(size_model, (size_t)3, serialized_bytes, saved_size_t, has_same_endianness);
                size_metadata = size_model[2];
                break;
            }
            
            default:
            {

            }
        }
    }

    return_to_position(serialized_bytes, saved_position);
}

template <class itype>
void inspect_serialized_object
(
    itype &serialized_bytes,
    bool &is_isotree_model,
    bool &is_compatible,
    bool &has_combined_objects,
    bool &has_IsoForest,
    bool &has_ExtIsoForest,
    bool &has_Imputer,
    bool &has_metadata,
    size_t &size_metadata
)
{
    bool ignored[3];
    inspect_serialized_object(
        serialized_bytes,
        is_isotree_model,
        is_compatible,
        has_combined_objects,
        has_IsoForest,
        has_ExtIsoForest,
        has_Imputer,
        has_metadata,
        size_metadata,
        ignored[0],
        ignored[1],
        ignored[2]
    );
}

void inspect_serialized_object
(
    const char *serialized_bytes,
    bool &is_isotree_model,
    bool &is_compatible,
    bool &has_combined_objects,
    bool &has_IsoForest,
    bool &has_ExtIsoForest,
    bool &has_Imputer,
    bool &has_metadata,
    size_t &size_metadata
)
{
    const char *in = serialized_bytes;
    inspect_serialized_object<const char*>(
        in,
        is_isotree_model,
        is_compatible,
        has_combined_objects,
        has_IsoForest,
        has_ExtIsoForest,
        has_Imputer,
        has_metadata,
        size_metadata
    );
}

void inspect_serialized_object
(
    const std::string &serialized_bytes,
    bool &is_isotree_model,
    bool &is_compatible,
    bool &has_combined_objects,
    bool &has_IsoForest,
    bool &has_ExtIsoForest,
    bool &has_Imputer,
    bool &has_metadata,
    size_t &size_metadata
)
{
    if (!serialized_bytes.size()) {
        is_isotree_model = false;
        is_compatible = false;
        has_IsoForest = false;
        has_ExtIsoForest = false;
        has_Imputer = false;
        return;
    }
    const char *in = &serialized_bytes[0];
    inspect_serialized_object<const char*>(
        in,
        is_isotree_model,
        is_compatible,
        has_combined_objects,
        has_IsoForest,
        has_ExtIsoForest,
        has_Imputer,
        has_metadata,
        size_metadata
    );
}

void inspect_serialized_object
(
    FILE *serialized_bytes,
    bool &is_isotree_model,
    bool &is_compatible,
    bool &has_combined_objects,
    bool &has_IsoForest,
    bool &has_ExtIsoForest,
    bool &has_Imputer,
    bool &has_metadata,
    size_t &size_metadata
)
{
    FILE *in = serialized_bytes;
    inspect_serialized_object<FILE*>(
        in,
        is_isotree_model,
        is_compatible,
        has_combined_objects,
        has_IsoForest,
        has_ExtIsoForest,
        has_Imputer,
        has_metadata,
        size_metadata
    );
}

void inspect_serialized_object
(
    std::istream &serialized_bytes,
    bool &is_isotree_model,
    bool &is_compatible,
    bool &has_combined_objects,
    bool &has_IsoForest,
    bool &has_ExtIsoForest,
    bool &has_Imputer,
    bool &has_metadata,
    size_t &size_metadata
)
{
    inspect_serialized_object<std::istream>(
        serialized_bytes,
        is_isotree_model,
        is_compatible,
        has_combined_objects,
        has_IsoForest,
        has_ExtIsoForest,
        has_Imputer,
        has_metadata,
        size_metadata
    );
}

size_t get_ntrees(const IsoForest &model)
{
    return model.trees.size();
}

size_t get_ntrees(const ExtIsoForest &model)
{
    return model.hplanes.size();
}

size_t get_ntrees(const Imputer &model)
{
    return model.imputer_tree.size();
}

template <class Model>
bool prev_cols_match(const Model &model, const char *serialized_bytes)
{
    return true;
}

bool prev_cols_match(const Imputer &model, const char *serialized_bytes)
{
    size_t prev[6];
    read_bytes<size_t>((void*)prev, (size_t)6, serialized_bytes);
    if (prev[0] != model.ncols_numeric ||
        prev[1] != model.ncols_categ ||
        prev[2] != model.ncat.size() ||
        prev[4] != model.col_means.size() ||
        prev[5] != model.col_modes.size())
    {
        return false;
    }

    return true;
}

template <class Model>
bool check_can_undergo_incremental_serialization(const Model &model, const char *serialized_bytes)
{
    const char *start = serialized_bytes;
    size_t curr_ntrees = get_ntrees(model);

    bool is_isotree_model;
    bool is_compatible;
    bool has_combined_objects;
    bool has_IsoForest;
    bool has_ExtIsoForest;
    bool has_Imputer;
    bool has_metadata;
    size_t size_metadata;
    bool has_same_int_size;
    bool has_same_size_t_size;
    bool has_same_endianness;

    inspect_serialized_object(
        serialized_bytes,
        is_isotree_model,
        is_compatible,
        has_combined_objects,
        has_IsoForest,
        has_ExtIsoForest,
        has_Imputer,
        has_metadata,
        size_metadata,
        has_same_int_size,
        has_same_size_t_size,
        has_same_endianness
    );

    if (!is_isotree_model || !is_compatible || has_combined_objects ||
        !has_same_int_size || !has_same_size_t_size || !has_same_endianness)
        return false;

    if (std::is_same<Model, IsoForest>::value) {
        if (!has_IsoForest || has_ExtIsoForest || has_Imputer)
            return false;
    }

    else if (std::is_same<Model, ExtIsoForest>::value) {
        if (has_IsoForest || !has_ExtIsoForest || has_Imputer)
            return false;
    }

    else if (std::is_same<Model, Imputer>::value) {
        if (has_IsoForest || has_ExtIsoForest || !has_Imputer)
            return false;
    }

    else {
        assert(0);
    }

    start += get_size_setup_info();
    start += sizeof(uint8_t);
    start += sizeof(size_t);

    if (std::is_same<Model, IsoForest>::value) {
        start += sizeof(uint8_t) * 3;
        start += sizeof(double) * 2;
        start += sizeof(size_t);
    }

    else if (std::is_same<Model, ExtIsoForest>::value) {
        start += sizeof(uint8_t) * 3;
        start += sizeof(double) * 2;
        start += sizeof(size_t);
    }

    else if (std::is_same<Model, Imputer>::value) {
        if (!prev_cols_match(model, start))
            return false;
        start += sizeof(size_t) * 3;
    }

    else {
        assert(0);
    }

    size_t old_ntrees;
    memcpy(&old_ntrees, start, sizeof(size_t));
    if (old_ntrees > curr_ntrees)
        return false;

    return true;
}

bool check_can_undergo_incremental_serialization(const IsoForest &model, const char *serialized_bytes)
{
    return check_can_undergo_incremental_serialization<IsoForest>(model, serialized_bytes);
}

bool check_can_undergo_incremental_serialization(const ExtIsoForest &model, const char *serialized_bytes)
{
    return check_can_undergo_incremental_serialization<ExtIsoForest>(model, serialized_bytes);
}

bool check_can_undergo_incremental_serialization(const Imputer &model, const char *serialized_bytes)
{
    return check_can_undergo_incremental_serialization<Imputer>(model, serialized_bytes);
}

size_t determine_serialized_size_combined
(
    const IsoForest *model,
    const ExtIsoForest *model_ext,
    const Imputer *imputer,
    const size_t size_optional_metadata
)
{
    size_t n_bytes = get_size_setup_info();
    n_bytes += 2 * sizeof(uint8_t);
    n_bytes += 3 * sizeof(size_t);

    if (model != NULL)
        n_bytes += get_size_model(*model);
    else
        n_bytes += get_size_model(*model_ext);
    if (imputer != NULL)
        n_bytes += get_size_model(*imputer);

    n_bytes += get_size_ending_metadata();
    return n_bytes;
}

template <class otype>
void serialize_combined
(
    const IsoForest *model,
    const ExtIsoForest *model_ext,
    const Imputer *imputer,
    const char *optional_metadata,
    const size_t size_optional_metadata,
    otype &out
)
{
    SignalSwitcher ss = SignalSwitcher();

    auto pos_watermark = set_return_position(out);

    add_setup_info(out, false);
    uint8_t model_type = AllObjectsCombined;
    write_bytes<uint8_t>((void*)&model_type, (size_t)1, out);

    if (model != NULL)
    {

        if (!size_optional_metadata)
        {
            if (imputer == NULL)
                model_type = HasSingleVarModelNext;
            else
                model_type = HasSingleVarModelPlusImputerNext;
        }

        else
        {
            if (imputer == NULL)
                model_type = HasSingleVarModelPlusMetadataNext;
            else
                model_type = HasSingleVarModelPlusImputerPlusMetadataNext;
        }

    }

    else if (model_ext != NULL)
    {

        if (!size_optional_metadata)
        {
            if (imputer == NULL)
                model_type = HasExtModelNext;
            else
                model_type = HasExtModelPlusImputerNext;
        }

        else
        {
            if (imputer == NULL)
                model_type = HasExtModelPlusMetadataNext;
            else
                model_type = HasExtModelPlusImputerPlusMetadataNext;
        }
    }

    else {
        throw std::runtime_error("Must pass one of 'model' or 'model_ext'.\n");
    }

    write_bytes<uint8_t>((void*)&model_type, (size_t)1, out);

    size_t size_model;
    if (model != NULL)
        size_model = get_size_model(*model);
    else
        size_model = get_size_model(*model_ext);
    write_bytes<size_t>((void*)&size_model, (size_t)1, out);

    if (imputer != NULL)
        size_model = get_size_model(*imputer);
    else
        size_model = 0;
    write_bytes<size_t>((void*)&size_model, (size_t)1, out);

    write_bytes<size_t>((void*)&size_optional_metadata, (size_t)1, out);


    check_interrupt_switch(ss);

    if (model != NULL)
        serialize_model(*model, out);
    else
        serialize_model(*model_ext, out);

    if (imputer != NULL)
        serialize_model(*imputer, out);

    if (size_optional_metadata)
        write_bytes<char>((void*)optional_metadata, size_optional_metadata, out);

    check_interrupt_switch(ss);

    uint8_t ending_type = (uint8_t)EndsHere;
    write_bytes<uint8_t>((void*)&ending_type, (size_t)1, out);
    size_t jump_ahead = 0;
    write_bytes<size_t>((void*)&jump_ahead, (size_t)1, out);

    auto end_pos = set_return_position(out);
    return_to_position(out, pos_watermark);
    add_full_watermark(out);
    return_to_position(out, end_pos);
}

void serialize_combined
(
    const IsoForest *model,
    const ExtIsoForest *model_ext,
    const Imputer *imputer,
    const char *optional_metadata,
    const size_t size_optional_metadata,
    char *out
)
{
    serialize_combined<char*>(
        model,
        model_ext,
        imputer,
        optional_metadata,
        size_optional_metadata,
        out
    );
}

void serialize_combined
(
    const IsoForest *model,
    const ExtIsoForest *model_ext,
    const Imputer *imputer,
    const char *optional_metadata,
    const size_t size_optional_metadata,
    FILE *out
)
{
    serialize_combined<FILE*>(
        model,
        model_ext,
        imputer,
        optional_metadata,
        size_optional_metadata,
        out
    );
}

void serialize_combined
(
    const IsoForest *model,
    const ExtIsoForest *model_ext,
    const Imputer *imputer,
    const char *optional_metadata,
    const size_t size_optional_metadata,
    std::ostream &out
)
{
    serialize_combined<std::ostream>(
        model,
        model_ext,
        imputer,
        optional_metadata,
        size_optional_metadata,
        out
    );
}

std::string serialize_combined
(
    const IsoForest *model,
    const ExtIsoForest *model_ext,
    const Imputer *imputer,
    const char *optional_metadata,
    const size_t size_optional_metadata
)
{
    std::string serialized;
    serialized.resize(determine_serialized_size_combined(model, model_ext, imputer, size_optional_metadata));
    char *ptr = &serialized[0];
    serialize_combined(model, model_ext, imputer, optional_metadata, size_optional_metadata, ptr);
    return serialized;
}

size_t determine_serialized_size_combined
(
    const char *serialized_model,
    const char *serialized_model_ext,
    const char *serialized_imputer,
    const size_t size_optional_metadata
)
{
    size_t n_bytes = get_size_setup_info();
    n_bytes += 2 * sizeof(uint8_t);
    n_bytes += 3 * sizeof(size_t);

    size_t model_size;

    if (serialized_model != NULL)
        memcpy(&model_size, serialized_model + get_size_setup_info() + sizeof(uint8_t), sizeof(size_t));
    else
        memcpy(&model_size, serialized_model_ext + get_size_setup_info() + sizeof(uint8_t), sizeof(size_t));
    n_bytes += model_size;
    if (serialized_imputer != NULL) {
        memcpy(&model_size, serialized_imputer + get_size_setup_info() + sizeof(uint8_t), sizeof(size_t));
        n_bytes += model_size;
    }

    n_bytes += size_optional_metadata;

    n_bytes += get_size_ending_metadata();
    return n_bytes;
}

template <class otype>
void serialize_combined
(
    const char *serialized_model,
    const char *serialized_model_ext,
    const char *serialized_imputer,
    const char *optional_metadata,
    const size_t size_optional_metadata,
    otype &out
)
{
    SignalSwitcher ss = SignalSwitcher();

    std::unique_ptr<char[]> curr_setup(new char[get_size_setup_info()]);
    char *ptr_curr_setup = curr_setup.get();
    add_setup_info(ptr_curr_setup, true);
    auto pos_watermark = set_return_position(out);
    add_setup_info(out, false);

    uint8_t model_type = AllObjectsCombined;
    write_bytes<uint8_t>((void*)&model_type, (size_t)1, out);

    if (serialized_model != NULL)
    {
        if (!size_optional_metadata)
        {
            if (serialized_imputer == NULL)
                model_type = HasSingleVarModelNext;
            else
                model_type = HasSingleVarModelPlusImputerNext;
        }

        else
        {
            if (serialized_imputer == NULL)
                model_type = HasSingleVarModelPlusMetadataNext;
            else
                model_type = HasSingleVarModelPlusImputerPlusMetadataNext;
        }
    }

    else
    {
        if (!size_optional_metadata)
        {
            if (serialized_imputer == NULL)
                model_type = HasExtModelNext;
            else
                model_type = HasExtModelPlusImputerNext;
        }

        else
        {
            if (serialized_imputer == NULL)
                model_type = HasExtModelPlusMetadataNext;
            else
                model_type = HasExtModelPlusImputerPlusMetadataNext;
        }
    }

    write_bytes<uint8_t>((void*)&model_type, (size_t)1, out);

    size_t model_size;
    size_t size_model1, size_model2, size_model3;

    std::unique_ptr<char[]> new_model;
    if (serialized_model != NULL)
    {
        if (memcmp(curr_setup.get(), serialized_model, get_size_setup_info()))
        {
            fprintf(stderr, "Warning: 'model' was serialized in a different setup, will need to convert.\n");
            IsoForest model;
            deserialization_pipeline(model, serialized_model);
            new_model = std::unique_ptr<char[]>(new char[get_size_model(model)]);
            char *ptr_new_model_ser = new_model.get();
            serialization_pipeline(model, ptr_new_model_ser);
            serialized_model = new_model.get();
        }
        serialized_model += get_size_setup_info() + sizeof(uint8_t);
        memcpy(&model_size, serialized_model, sizeof(size_t));
        serialized_model += sizeof(size_t);
        size_model1 = model_size;
    }

    else
    {
        if (memcmp(curr_setup.get(), serialized_model_ext, get_size_setup_info()))
        {
            fprintf(stderr, "Warning: 'model_ext' was serialized in a different setup, will need to convert.\n");
            ExtIsoForest model;
            deserialization_pipeline(model, serialized_model_ext);
            new_model = std::unique_ptr<char[]>(new char[get_size_model(model)]);
            char *ptr_new_model_ser = new_model.get();
            serialization_pipeline(model, ptr_new_model_ser);
            serialized_model_ext = new_model.get();
        }
        serialized_model_ext += get_size_setup_info() + sizeof(uint8_t);
        memcpy(&model_size, serialized_model_ext, sizeof(size_t));
        serialized_model_ext += sizeof(size_t);
        size_model2 = model_size;
    }

    check_interrupt_switch(ss);

    write_bytes<size_t>((void*)&model_size, (size_t)1, out);

    if (serialized_imputer != NULL)
    {
        if (memcmp(curr_setup.get(), serialized_imputer, get_size_setup_info()))
        {
            fprintf(stderr, "Warning: 'imputer' was serialized in a different setup, will need to convert.\n");
            Imputer model;
            deserialization_pipeline(model, serialized_imputer);
            new_model = std::unique_ptr<char[]>(new char[get_size_model(model)]);
            char *ptr_new_model_ser = new_model.get();
            serialization_pipeline(model, ptr_new_model_ser);
            serialized_imputer = new_model.get();
        }
        serialized_imputer += get_size_setup_info() + sizeof(uint8_t);
        memcpy(&model_size, serialized_imputer, sizeof(size_t));
        serialized_imputer += sizeof(size_t);
        size_model3 = model_size;
    }

    else {
        model_size = 0;
    }
    write_bytes<size_t>((void*)&model_size, (size_t)1, out);

    check_interrupt_switch(ss);

    write_bytes<size_t>((void*)&size_optional_metadata, (size_t)1, out);

    if (serialized_model != NULL)
        write_bytes<char>((void*)serialized_model, size_model1, out);
    else
        write_bytes<char>((void*)serialized_model_ext, size_model2, out);
    if (serialized_imputer != NULL)
        write_bytes<char>((void*)serialized_imputer, size_model3, out);

    if (size_optional_metadata)
        write_bytes<char>((void*)optional_metadata, size_optional_metadata, out);

    check_interrupt_switch(ss);

    uint8_t ending_type = (uint8_t)EndsHere;
    write_bytes<uint8_t>((void*)&ending_type, (size_t)1, out);
    size_t jump_ahead = 0;
    write_bytes<size_t>((void*)&jump_ahead, (size_t)1, out);

    auto end_pos = set_return_position(out);
    return_to_position(out, pos_watermark);
    add_full_watermark(out);
    return_to_position(out, end_pos);
}

void serialize_combined
(
    const char *serialized_model,
    const char *serialized_model_ext,
    const char *serialized_imputer,
    const char *optional_metadata,
    const size_t size_optional_metadata,
    FILE *out
)
{
    serialize_combined<FILE*>(
        serialized_model,
        serialized_model_ext,
        serialized_imputer,
        optional_metadata,
        size_optional_metadata,
        out
    );
}

void serialize_combined
(
    const char *serialized_model,
    const char *serialized_model_ext,
    const char *serialized_imputer,
    const char *optional_metadata,
    const size_t size_optional_metadata,
    std::ostream &out
)
{
    serialize_combined<std::ostream>(
        serialized_model,
        serialized_model_ext,
        serialized_imputer,
        optional_metadata,
        size_optional_metadata,
        out
    );
}

std::string serialize_combined
(
    const char *serialized_model,
    const char *serialized_model_ext,
    const char *serialized_imputer,
    const char *optional_metadata,
    const size_t size_optional_metadata
)
{
    std::string serialized;
    serialized.resize(
        determine_serialized_size_combined(
            serialized_model,
            serialized_model_ext,
            serialized_imputer,
            size_optional_metadata
        )
    );
    char *ptr = &serialized[0];
    serialize_combined(
        serialized_model,
        serialized_model_ext,
        serialized_imputer,
        optional_metadata,
        size_optional_metadata,
        ptr
    );
    return serialized;
}

template <class Model, class itype>
void deserialize_model
(
    Model &model,
    itype &in,
    const bool has_same_endianness,
    const bool has_same_int_size,
    const bool has_same_size_t_size,
    const PlatformSize saved_int_t,
    const PlatformSize saved_size_t
)
{
    if (has_same_endianness && has_same_int_size && has_same_size_t_size)
    {
        deserialize_model(model, in);
        return;
    }

    std::vector<char> buffer;

    if (saved_int_t == Is16Bit && saved_size_t == Is32Bit)
    {
        deserialize_model<itype, int16_t, uint32_t>(model, in, buffer, !has_same_endianness);
    }

    else if (saved_int_t == Is32Bit && saved_size_t == Is32Bit)
    {
        deserialize_model<itype, int32_t, uint32_t>(model, in, buffer, !has_same_endianness);
    }

    else if (saved_int_t == Is64Bit && saved_size_t == Is32Bit)
    {
        deserialize_model<itype, int64_t, uint32_t>(model, in, buffer, !has_same_endianness);
    }

    else if (saved_int_t == Is16Bit && saved_size_t == Is64Bit)
    {
        deserialize_model<itype, int16_t, uint64_t>(model, in, buffer, !has_same_endianness);
    }

    else if (saved_int_t == Is32Bit && saved_size_t == Is64Bit)
    {
        deserialize_model<itype, int32_t, uint64_t>(model, in, buffer, !has_same_endianness);
    }

    else if (saved_int_t == Is64Bit && saved_size_t == Is64Bit)
    {
        deserialize_model<itype, int16_t, uint64_t>(model, in, buffer, !has_same_endianness);
    }

    else
    {
        throw std::runtime_error("Unexpected error.\n");
    }
}

template <class itype>
void deserialize_combined
(
    itype &in,
    IsoForest *model,
    ExtIsoForest *model_ext,
    Imputer *imputer,
    char *optional_metadata
)
{
    SignalSwitcher ss = SignalSwitcher();

    bool has_same_int_size;
    bool has_same_size_t_size;
    bool has_same_endianness;
    PlatformSize saved_int_t;
    PlatformSize saved_size_t;
    PlatformEndianness saved_endian;

    check_setup_info(
        in,
        has_same_int_size,
        has_same_size_t_size,
        has_same_endianness,
        saved_int_t,
        saved_size_t,
        saved_endian
    );

    uint8_t model_in;
    read_bytes<uint8_t>((void*)&model_in, (size_t)1, in);
    if (model_in != AllObjectsCombined)
        throw std::runtime_error("Object to de-serialize was not created through 'serialize_combined'.\n");

    read_bytes<uint8_t>((void*)&model_in, (size_t)1, in);
    size_t size_model[3];
    read_bytes_size_t((void*)size_model, (size_t)3, in, saved_size_t, has_same_endianness);

    switch(model_in)
    {
        case HasSingleVarModelNext:
        {
            deserialize_model(*model, in, has_same_endianness, has_same_int_size, has_same_size_t_size, saved_int_t, saved_size_t);
            break;
        }
        case HasExtModelNext:
        {
            deserialize_model(*model_ext, in, has_same_endianness, has_same_int_size, has_same_size_t_size, saved_int_t, saved_size_t);
            break;
        }
        case HasSingleVarModelPlusImputerNext:
        {
            deserialize_model(*model, in, has_same_endianness, has_same_int_size, has_same_size_t_size, saved_int_t, saved_size_t);
            check_interrupt_switch(ss);
            deserialize_model(*imputer, in, has_same_endianness, has_same_int_size, has_same_size_t_size, saved_int_t, saved_size_t);
            break;
        }
        case HasExtModelPlusImputerNext:
        {
            deserialize_model(*model_ext, in, has_same_endianness, has_same_int_size, has_same_size_t_size, saved_int_t, saved_size_t);
            check_interrupt_switch(ss);
            deserialize_model(*imputer, in, has_same_endianness, has_same_int_size, has_same_size_t_size, saved_int_t, saved_size_t);
            break;
        }
        case HasSingleVarModelPlusMetadataNext:
        {
            deserialize_model(*model, in, has_same_endianness, has_same_int_size, has_same_size_t_size, saved_int_t, saved_size_t);
            check_interrupt_switch(ss);
            read_bytes<char>((void*)optional_metadata, size_model[2], in);
            break;
        }
        case HasExtModelPlusMetadataNext:
        {
            deserialize_model(*model_ext, in, has_same_endianness, has_same_int_size, has_same_size_t_size, saved_int_t, saved_size_t);
            check_interrupt_switch(ss);
            read_bytes<char>((void*)optional_metadata, size_model[2], in);
            break;
        }
        case HasSingleVarModelPlusImputerPlusMetadataNext:
        {
            deserialize_model(*model, in, has_same_endianness, has_same_int_size, has_same_size_t_size, saved_int_t, saved_size_t);
            check_interrupt_switch(ss);
            deserialize_model(*imputer, in, has_same_endianness, has_same_int_size, has_same_size_t_size, saved_int_t, saved_size_t);
            check_interrupt_switch(ss);
            read_bytes<char>((void*)optional_metadata, size_model[2], in);
            break;
        }
        case HasExtModelPlusImputerPlusMetadataNext:
        {
            deserialize_model(*model_ext, in, has_same_endianness, has_same_int_size, has_same_size_t_size, saved_int_t, saved_size_t);
            check_interrupt_switch(ss);
            deserialize_model(*imputer, in, has_same_endianness, has_same_int_size, has_same_size_t_size, saved_int_t, saved_size_t);
            check_interrupt_switch(ss);
            read_bytes<char>((void*)optional_metadata, size_model[2], in);
            break;
        }
        
        default:
        {
            throw std::runtime_error("Serialized format is incompatible.\n");
        }
    }
}

void deserialize_combined
(
    const char* in,
    IsoForest *model,
    ExtIsoForest *model_ext,
    Imputer *imputer,
    char *optional_metadata
)
{
    deserialize_combined<const char*>(
        in,
        model,
        model_ext,
        imputer,
        optional_metadata
    );
}

void deserialize_combined
(
    FILE* in,
    IsoForest *model,
    ExtIsoForest *model_ext,
    Imputer *imputer,
    char *optional_metadata
)
{
    deserialize_combined<FILE*>(
        in,
        model,
        model_ext,
        imputer,
        optional_metadata
    );
}

void deserialize_combined
(
    std::istream &in,
    IsoForest *model,
    ExtIsoForest *model_ext,
    Imputer *imputer,
    char *optional_metadata
)
{
    deserialize_combined<std::istream>(
        in,
        model,
        model_ext,
        imputer,
        optional_metadata
    );
}

void deserialize_combined
(
    const std::string &in,
    IsoForest *model,
    ExtIsoForest *model_ext,
    Imputer *imputer,
    char *optional_metadata
)
{
    const char *ptr = &in[0];
    deserialize_combined<const char*>(
        ptr,
        model,
        model_ext,
        imputer,
        optional_metadata
    );
}
