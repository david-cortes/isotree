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
*     Copyright (c) 2019, David Cortes
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

#ifdef _ENABLE_CEREAL


template <class T>
void serialize_obj(T &obj, std::ostream &output)
{
    cereal::BinaryOutputArchive archive(output);
    archive(obj);
}
template <class T>
std::string serialize_obj(T &obj)
{
    std::stringstream ss;
    {
        cereal::BinaryOutputArchive archive(ss);
        archive(obj);
    }
    return ss.str();
}
template <class T, class I>
void deserialize_obj(T &output, I &serialized)
{
    cereal::BinaryInputArchive archive(serialized);
    archive(output);
}
template <class T>
void deserialize_obj(T &output, std::string &serialized, bool move_str)
{
    std::stringstream ss;
    if (move_str)
        ss.str(std::move(serialized));
    else
        /* Bug with GCC4 not implementing the move method for stringsreams
           https://stackoverflow.com/questions/50926506/deleted-function-std-basic-stringstream-in-linux-with-g
           https://github.com/david-cortes/isotree/issues/7 */
        // ss = std::stringstream(serialized); /* <- fails with GCC4, CRAN complains */
        {
            std::string str_copy = serialized;
            ss.str(str_copy);
        }
    deserialize_obj(output, ss);
}


/* Serialization and de-serialization functions using Cereal
* 
* Parameters
* ==========
* - model (in)
*       A model object to serialize, after being fitted through function 'fit_iforest'.
* - imputer (in)
*       An imputer object to serialize, after being fitted through function 'fit_iforest'
*       with 'build_imputer=true'.
* - output_obj (out)
*       An already-allocated object into which a serialized object of the same class will
*       be de-serialized. The contents of this object will be overwritten. Should be initialized
*       through the default constructor (e.g. 'new ExtIsoForest' or 'ExtIsoForest()').
* - output (out)
*       An output stream (any type will do) in which to save/persist/serialize the
*       model or imputer object using the cereal library. In the functions that do not
*       take this parameter, it will be returned as a string containing the raw bytes.
* - serialized (in)
*       The input stream which contains the serialized/saved/persisted model or imputer object,
*       which will be de-serialized into 'output'.
* - output_file_path
*       File name into which to write the serialized model or imputer object as raw bytes.
*       Note that, on Windows, passing non-ASCII characters will fail, and in such case,
*       you might instead want to use instead the versions that take 'wchar_t', which are
*       only available in the MSVC compiler (it uses 'std::ofstream' internally, which as
*       of C++20, is not required by the standard to accept 'wchar_t' in its constructor).
*       Be aware that it will only write raw bytes, thus metadata such as CPU endianness
*       will be lost. If you need to transfer files berween e.g. an x86 computer and a SPARC
*       server, you'll have to use other methods.
*       This  functionality is intended for being easily wrapper into scripting languages
*       without having to copy the contents to to some intermediate language.
* - input_file_path
*       File name from which to read a serialized model or imputer object as raw bytes.
*       See the description for 'output_file_path' for more details.
* - move_str
*       Whether to move ('std::move') the contents of the string passed as input in order
*       to speed things up and avoid making a redundant copy of the raw bytes. If passing
*       'true', the input string will be rendered empty afterwards.
*/
void serialize_isoforest(IsoForest &model, std::ostream &output)
{
    serialize_obj(model, output);
}
void serialize_isoforest(IsoForest &model, const char *output_file_path)
{
    std::ofstream output(output_file_path);
    serialize_obj(model, output);
}
std::string serialize_isoforest(IsoForest &model)
{
    return serialize_obj(model);
}
void deserialize_isoforest(IsoForest &output_obj, std::istream &serialized)
{
    deserialize_obj(output_obj, serialized);
}
void deserialize_isoforest(IsoForest &output_obj, const char *input_file_path)
{
    std::ifstream serialized(input_file_path);
    deserialize_obj(output_obj, serialized);
}
void deserialize_isoforest(IsoForest &output_obj, std::string &serialized, bool move_str)
{
    deserialize_obj(output_obj, serialized, move_str);
}



void serialize_ext_isoforest(ExtIsoForest &model, std::ostream &output)
{
    serialize_obj(model, output);
}
void serialize_ext_isoforest(ExtIsoForest &model, const char *output_file_path)
{
    std::ofstream output(output_file_path);
    serialize_obj(model, output);
}
std::string serialize_ext_isoforest(ExtIsoForest &model)
{
    return serialize_obj(model);
}
void deserialize_ext_isoforest(ExtIsoForest &output_obj, std::istream &serialized)
{
    deserialize_obj(output_obj, serialized);
}
void deserialize_ext_isoforest(ExtIsoForest &output_obj, const char *input_file_path)
{
    std::ifstream serialized(input_file_path);
    deserialize_obj(output_obj, serialized);
}
void deserialize_ext_isoforest(ExtIsoForest &output_obj, std::string &serialized, bool move_str)
{
    deserialize_obj(output_obj, serialized, move_str);
}




void serialize_imputer(Imputer &imputer, std::ostream &output)
{
    serialize_obj(imputer, output);
}
void serialize_imputer(Imputer &imputer, const char *output_file_path)
{
    std::ofstream output(output_file_path);
    serialize_obj(imputer, output);
}
std::string serialize_imputer(Imputer &imputer)
{
    return serialize_obj(imputer);
}
void deserialize_imputer(Imputer &output_obj, std::istream &serialized)
{
    deserialize_obj(output_obj, serialized);
}
void deserialize_imputer(Imputer &output_obj, const char *input_file_path)
{
    std::ifstream serialized(input_file_path);
    deserialize_obj(output_obj, serialized);
}
void deserialize_imputer(Imputer &output_obj, std::string &serialized, bool move_str)
{
    deserialize_obj(output_obj, serialized, move_str);
}


#ifdef _MSC_VER
void serialize_isoforest(IsoForest &model, const wchar_t *output_file_path)
{
    std::ofstream output(output_file_path);
    serialize_obj(model, output);
}
void deserialize_isoforest(IsoForest &output_obj, const wchar_t *input_file_path)
{
    std::ifstream serialized(input_file_path);
    deserialize_obj(output_obj, serialized);
}
void serialize_ext_isoforest(ExtIsoForest &model, const wchar_t *output_file_path)
{
    std::ofstream output(output_file_path);
    serialize_obj(model, output);
}
void deserialize_ext_isoforest(ExtIsoForest &output_obj, const wchar_t *input_file_path)
{
    std::ifstream serialized(input_file_path);
    deserialize_obj(output_obj, serialized);
}
void serialize_imputer(Imputer &imputer, const wchar_t *output_file_path)
{
    std::ofstream output(output_file_path);
    serialize_obj(imputer, output);
}
void deserialize_imputer(Imputer &output_obj, const wchar_t *input_file_path)
{
    std::ifstream serialized(input_file_path);
    deserialize_obj(output_obj, serialized);
}
bool has_msvc()
{
    return true;
}

#else
bool has_msvc()
{
    return false;
}

#endif /* ifdef _MSC_VER */


#endif /* _ENABLE_CEREAL */
