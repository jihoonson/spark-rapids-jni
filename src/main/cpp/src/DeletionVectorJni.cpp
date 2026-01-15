/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cudf_jni_apis.hpp"
#include "jni_utils.hpp"
#include "multi_host_buffer_source.hpp"

#include <cudf/io/experimental/deletion_vectors.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>

#include <cuda/std/cstddef>
#include <vector>
#include <iostream>

extern "C" {

/**
 * @brief Read a Parquet file with deletion vector support
 *
 * This JNI function wraps cudf::io::parquet::experimental::read_parquet to read a Parquet file,
 * prepend an index column, and apply a deletion vector filter.
 *
 * @param env JNI environment
 * @param j_options_handle Handle to the parquet_reader_options object
 * @param j_serialized_roaring64 Serialized 64-bit roaring bitmap (deletion vector)
 * @param j_row_group_offsets Row index offsets for each row group
 * @param j_row_group_num_rows Number of rows in each row group
 * @return Handle to the resulting table (as jlong)
 */
JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_DeletionVector_readParquetWithDeletionVector(
  JNIEnv* env,
  jclass,
  jobjectArray filter_col_names,
  jbooleanArray j_col_binary_read,
  jstring inputfilepath,
  jlongArray addrs_and_sizes,
  jint unit,
  jbyteArray j_serialized_roaring64,
  jlongArray j_row_group_offsets,
  jintArray j_row_group_num_rows)
{
  JNI_NULL_CHECK(env, j_col_binary_read, "null col_binary_read", NULL);
  bool read_buffer = true;
  if (addrs_and_sizes == nullptr) {
    JNI_NULL_CHECK(env, inputfilepath, "input file or buffer must be supplied", NULL);
    read_buffer = false;
  } else if (inputfilepath != NULL) {
    JNI_THROW_NEW(env,
                  cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                  "cannot pass in both a buffer and an inputfilepath",
                  NULL);
  }

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    cudf::jni::native_jstring filename(env, inputfilepath);
    if (!read_buffer && filename.is_empty()) {
      JNI_THROW_NEW(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "inputfilepath can't be empty", NULL);
    }

    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);
    cudf::jni::native_jbooleanArray n_col_binary_read(env, j_col_binary_read);
    cudf::jni::native_jlongArray n_addrs_sizes(env, addrs_and_sizes);

    std::unique_ptr<cudf::io::datasource> multi_buffer_source;
    cudf::io::source_info source;
    if (read_buffer) {
      multi_buffer_source.reset(new cudf::jni::multi_host_buffer_source(n_addrs_sizes));
      source = cudf::io::source_info(multi_buffer_source.get());
    } else {
      source = cudf::io::source_info(filename.get());
    }

    auto builder = cudf::io::parquet_reader_options::builder(source);
    if (n_filter_col_names.size() > 0) {
      builder = builder.columns(n_filter_col_names.as_cpp_vector());
    }

    cudf::io::parquet_reader_options opts =
      builder.convert_strings_to_categories(false)
        .timestamp_type(cudf::data_type(static_cast<cudf::type_id>(unit)))
        // Ignore any missing projected column(s) by default
        .ignore_missing_columns(true)
        .build();

    // Convert serialized roaring bitmap
    cudf::host_span<cuda::std::byte const> serialized_roaring64;
    cudf::jni::native_jbyteArray n_serialized_roaring64(env, j_serialized_roaring64);
    if (j_serialized_roaring64 != nullptr && n_serialized_roaring64.size() > 0) {
      serialized_roaring64 = cudf::host_span<cuda::std::byte const>(
        reinterpret_cast<cuda::std::byte const*>(n_serialized_roaring64.data()),
        n_serialized_roaring64.size());
    }

    // Convert row group offsets
    cudf::host_span<size_t const> row_group_offsets;
    cudf::jni::native_jlongArray n_row_group_offsets(env, j_row_group_offsets);
    if (j_row_group_offsets != nullptr && n_row_group_offsets.size() > 0) {
      row_group_offsets = cudf::host_span<size_t const>(
        reinterpret_cast<size_t const*>(n_row_group_offsets.data()), n_row_group_offsets.size());
    }

    // Convert row group num_rows
    cudf::host_span<cudf::size_type const> row_group_num_rows;
    cudf::jni::native_jintArray n_row_group_num_rows(env, j_row_group_num_rows);
    if (j_row_group_num_rows != nullptr && n_row_group_num_rows.size() > 0) {
      row_group_num_rows = cudf::host_span<cudf::size_type const>(n_row_group_num_rows.data(),
                                                                   n_row_group_num_rows.size());
    }

    // Call the cuDF function
    auto tbl = cudf::io::parquet::experimental::read_parquet(
      opts, serialized_roaring64, row_group_offsets, row_group_num_rows).tbl;

    n_col_binary_read.cancel();
    n_addrs_sizes.cancel();

    return cudf::jni::convert_table_for_return(env, tbl);
  }
  JNI_CATCH(env, NULL);
}

/**
 * @brief Read a Parquet file with multiple deletion vectors support
 *
 * This JNI function wraps cudf::io::parquet::experimental::read_parquet to read a Parquet file,
 * prepend an index column, and apply multiple deletion vector filters.
 *
 * @param env JNI environment
 * @param j_options_handle Handle to the parquet_reader_options object
 * @param j_serialized_roaring_bitmaps Array of serialized 64-bit roaring bitmaps (deletion vectors)
 * @param j_deletion_vector_row_counts Number of rows in each deletion vector
 * @param j_row_group_offsets Row index offsets for each row group
 * @param j_row_group_num_rows Number of rows in each row group
 * @return Handle to the resulting table (as jlong)
 */
JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_DeletionVector_readParquetWithMultipleDeletionVectors(
  JNIEnv* env,
  jclass,
  jlong j_options_handle,
  jobjectArray j_serialized_roaring_bitmaps,
  jintArray j_deletion_vector_row_counts,
  jlongArray j_row_group_offsets,
  jintArray j_row_group_num_rows)
{
  JNI_NULL_CHECK(env, j_options_handle, "options handle is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    // Get the parquet_reader_options from the handle
    auto const& options =
      *reinterpret_cast<cudf::io::parquet_reader_options const*>(j_options_handle);

    // Convert array of serialized roaring bitmaps
    std::vector<std::vector<cuda::std::byte>> bitmap_data_storage;
    std::vector<cudf::host_span<cuda::std::byte const>> bitmap_spans;
    
    if (j_serialized_roaring_bitmaps != nullptr) {
      jsize num_bitmaps = env->GetArrayLength(j_serialized_roaring_bitmaps);
      bitmap_data_storage.reserve(num_bitmaps);
      bitmap_spans.reserve(num_bitmaps);
      
      for (jsize i = 0; i < num_bitmaps; i++) {
        auto j_bitmap = static_cast<jbyteArray>(env->GetObjectArrayElement(j_serialized_roaring_bitmaps, i));
        if (j_bitmap != nullptr) {
          cudf::jni::native_jbyteArray n_bitmap(env, j_bitmap);
          if (n_bitmap.size() > 0) {
            // Store the data in a vector
            std::vector<cuda::std::byte> bitmap_data(n_bitmap.size());
            std::memcpy(bitmap_data.data(), 
                       reinterpret_cast<cuda::std::byte const*>(n_bitmap.data()),
                       n_bitmap.size());
            bitmap_data_storage.push_back(std::move(bitmap_data));
            bitmap_spans.push_back(cudf::host_span<cuda::std::byte const>(
              bitmap_data_storage.back().data(), bitmap_data_storage.back().size()));
          } else {
            bitmap_spans.push_back(cudf::host_span<cuda::std::byte const>());
          }
        } else {
          bitmap_spans.push_back(cudf::host_span<cuda::std::byte const>());
        }
        env->DeleteLocalRef(j_bitmap);
      }
    }

    cudf::host_span<cudf::host_span<cuda::std::byte const> const> serialized_roaring_bitmaps(
      bitmap_spans.data(), bitmap_spans.size());

    // Convert deletion vector row counts
    cudf::host_span<cudf::size_type const> deletion_vector_row_counts;
    cudf::jni::native_jintArray n_deletion_vector_row_counts(env, j_deletion_vector_row_counts);
    if (j_deletion_vector_row_counts != nullptr && n_deletion_vector_row_counts.size() > 0) {
      deletion_vector_row_counts = cudf::host_span<cudf::size_type const>(
        n_deletion_vector_row_counts.data(), n_deletion_vector_row_counts.size());
    }

    // Convert row group offsets
    cudf::host_span<size_t const> row_group_offsets;
    cudf::jni::native_jlongArray n_row_group_offsets(env, j_row_group_offsets);
    if (j_row_group_offsets != nullptr && n_row_group_offsets.size() > 0) {
      row_group_offsets = cudf::host_span<size_t const>(
        reinterpret_cast<size_t const*>(n_row_group_offsets.data()), n_row_group_offsets.size());
    }

    // Convert row group num_rows
    cudf::host_span<cudf::size_type const> row_group_num_rows;
    cudf::jni::native_jintArray n_row_group_num_rows(env, j_row_group_num_rows);
    if (j_row_group_num_rows != nullptr && n_row_group_num_rows.size() > 0) {
      row_group_num_rows = cudf::host_span<cudf::size_type const>(n_row_group_num_rows.data(),
                                                                   n_row_group_num_rows.size());
    }

    // Call the cuDF function with multiple deletion vectors
    auto result = cudf::io::parquet::experimental::read_parquet(
      options, serialized_roaring_bitmaps, deletion_vector_row_counts, 
      row_group_offsets, row_group_num_rows);

    return cudf::jni::release_as_jlong(result.tbl);
  }
  JNI_CATCH(env, 0);
}

/**
 * @brief Create a chunked Parquet reader with deletion vector support and pass read limit
 *
 * This JNI function creates a chunked_parquet_reader with both chunk and pass read limits.
 *
 * @param env JNI environment
 * @param j_chunk_read_limit Byte limit on returned table chunk size, 0 if no limit
 * @param j_pass_read_limit Byte limit on decompression memory, 0 if no limit
 * @param j_options_handle Handle to the parquet_reader_options object
 * @param j_serialized_roaring64 Serialized 64-bit roaring bitmap (deletion vector)
 * @param j_row_group_offsets Row index offsets for each row group
 * @param j_row_group_num_rows Number of rows in each row group
 * @return Handle to the chunked_parquet_reader (as jlong)
 */
JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_DeletionVector_createChunkedParquetReaderWithPassLimit(
  JNIEnv* env,
  jclass,
  jlong j_chunk_read_limit,
  jlong j_pass_read_limit,
  jobjectArray filter_col_names,
  jbooleanArray j_col_binary_read,
  jstring inp_file_path,
  jlongArray addrs_sizes,
  jint unit,
  jbyteArray j_serialized_roaring64,
  jlongArray j_row_group_offsets,
  jintArray j_row_group_num_rows)
{
  JNI_NULL_CHECK(env, j_col_binary_read, "Null col_binary_read", nullptr);
  bool read_buffer = true;
  if (addrs_sizes == nullptr) {
    JNI_NULL_CHECK(env, inp_file_path, "Input file or buffer must be supplied", nullptr);
    read_buffer = false;
  } else if (inp_file_path != nullptr) {
    JNI_THROW_NEW(env,
                  cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                  "Cannot pass in both buffers and an inp_file_path",
                  nullptr);
  }

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstring filename(env, inp_file_path);
    if (!read_buffer && filename.is_empty()) {
      JNI_THROW_NEW(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "inp_file_path cannot be empty", nullptr);
    }

    cudf::jni::native_jstringArray n_filter_col_names(env, filter_col_names);

    // TODO: This variable is unused now, but we still don't know what to do with it yet.
    // As such, it needs to stay here for a little more time before we decide to use it again,
    // or remove it completely.
    cudf::jni::native_jbooleanArray n_col_binary_read(env, j_col_binary_read);
    (void)n_col_binary_read;

    cudf::jni::native_jlongArray n_addrs_sizes(env, addrs_sizes);
    std::unique_ptr<cudf::io::datasource> multi_buffer_source;
    cudf::io::source_info source;
    if (read_buffer) {
      multi_buffer_source.reset(new cudf::jni::multi_host_buffer_source(n_addrs_sizes));
      source = cudf::io::source_info(multi_buffer_source.get());
    } else {
      source = cudf::io::source_info(filename.get());
    }

    auto opts_builder = cudf::io::parquet_reader_options::builder(source);
    if (n_filter_col_names.size() > 0) {
      opts_builder = opts_builder.columns(n_filter_col_names.as_cpp_vector());
    }
    auto const read_opts = opts_builder.convert_strings_to_categories(false)
                             .timestamp_type(cudf::data_type(static_cast<cudf::type_id>(unit)))
                             // Ignore any missing projected column(s) by default
                             .ignore_missing_columns(true)
                             .build();

    // Convert serialized roaring bitmap
    cudf::host_span<cuda::std::byte const> serialized_roaring64;
    cudf::jni::native_jbyteArray n_serialized_roaring64(env, j_serialized_roaring64);
    if (j_serialized_roaring64 != nullptr && n_serialized_roaring64.size() > 0) {
      serialized_roaring64 = cudf::host_span<cuda::std::byte const>(
        reinterpret_cast<cuda::std::byte const*>(n_serialized_roaring64.data()),
        n_serialized_roaring64.size());
    }

    printf("creating serialized_roaring64_bitmaps span\n");
    std::vector<cudf::host_span<cuda::std::byte const>> serialized_roaring_bitmaps_vec;
    serialized_roaring_bitmaps_vec.push_back(serialized_roaring64);
    cudf::host_span<cudf::host_span<cuda::std::byte const> const> serialized_roaring64_bitmaps(
      serialized_roaring_bitmaps_vec);
    printf("created serialized_roaring64_bitmaps span\n");

    // Convert row group offsets
    cudf::host_span<size_t const> row_group_offsets;
    cudf::jni::native_jlongArray n_row_group_offsets(env, j_row_group_offsets);
    if (j_row_group_offsets != nullptr && n_row_group_offsets.size() > 0) {
      row_group_offsets = cudf::host_span<size_t const>(
        reinterpret_cast<size_t const*>(n_row_group_offsets.data()), n_row_group_offsets.size());
    }

    // Convert row group num_rows
    cudf::host_span<cudf::size_type const> row_group_num_rows;
    cudf::jni::native_jintArray n_row_group_num_rows(env, j_row_group_num_rows);
    if (j_row_group_num_rows != nullptr && n_row_group_num_rows.size() > 0) {
      row_group_num_rows = cudf::host_span<cudf::size_type const>(n_row_group_num_rows.data(),
                                                                   n_row_group_num_rows.size());
    }

    n_addrs_sizes.cancel();
    n_col_binary_read.cancel();

    printf("creating deletion_vector_row_counts span\n");
    std::vector<cudf::size_type> deletion_vector_row_counts;
    deletion_vector_row_counts.emplace_back(std::numeric_limits<cudf::size_type>::max());
    cudf::host_span<cudf::size_type const> deletion_vector_row_counts_span(deletion_vector_row_counts);
    printf("created deletion_vector_row_counts span\n");

    // Create the chunked reader with pass read limit
    auto reader = new cudf::io::parquet::experimental::chunked_parquet_reader(
      static_cast<std::size_t>(j_chunk_read_limit),
      static_cast<std::size_t>(j_pass_read_limit),
      read_opts,
      serialized_roaring64_bitmaps,
      deletion_vector_row_counts_span,
      row_group_offsets,
      row_group_num_rows);

    auto reader_handle = reinterpret_cast<jlong>(reader);
    cudf::jni::native_jlongArray result(env, 2);
    result[0] = reader_handle;
    result[1] = cudf::jni::release_as_jlong(multi_buffer_source);
    return result.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

/**
 * @brief Create a chunked Parquet reader with multiple deletion vectors support
 *
 * This JNI function creates a chunked_parquet_reader with multiple deletion vectors.
 *
 * @param env JNI environment
 * @param j_chunk_read_limit Byte limit on returned table chunk size, 0 if no limit
 * @param j_options_handle Handle to the parquet_reader_options object
 * @param j_serialized_roaring_bitmaps Array of serialized 64-bit roaring bitmaps
 * @param j_deletion_vector_row_counts Number of rows in each deletion vector
 * @param j_row_group_offsets Row index offsets for each row group
 * @param j_row_group_num_rows Number of rows in each row group
 * @return Handle to the chunked_parquet_reader (as jlong)
 */
JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_DeletionVector_createChunkedReaderWithMultipleDeletionVectors(
  JNIEnv* env,
  jclass,
  jlong j_chunk_read_limit,
  jlong j_options_handle,
  jobjectArray j_serialized_roaring_bitmaps,
  jintArray j_deletion_vector_row_counts,
  jlongArray j_row_group_offsets,
  jintArray j_row_group_num_rows)
{
  JNI_NULL_CHECK(env, j_options_handle, "options handle is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    // Get the parquet_reader_options from the handle
    auto const& options =
      *reinterpret_cast<cudf::io::parquet_reader_options const*>(j_options_handle);

    // Convert array of serialized roaring bitmaps
    std::vector<std::vector<cuda::std::byte>> bitmap_data_storage;
    std::vector<cudf::host_span<cuda::std::byte const>> bitmap_spans;
    
    if (j_serialized_roaring_bitmaps != nullptr) {
      jsize num_bitmaps = env->GetArrayLength(j_serialized_roaring_bitmaps);
      bitmap_data_storage.reserve(num_bitmaps);
      bitmap_spans.reserve(num_bitmaps);
      
      for (jsize i = 0; i < num_bitmaps; i++) {
        auto j_bitmap = static_cast<jbyteArray>(env->GetObjectArrayElement(j_serialized_roaring_bitmaps, i));
        if (j_bitmap != nullptr) {
          cudf::jni::native_jbyteArray n_bitmap(env, j_bitmap);
          if (n_bitmap.size() > 0) {
            std::vector<cuda::std::byte> bitmap_data(n_bitmap.size());
            std::memcpy(bitmap_data.data(), 
                       reinterpret_cast<cuda::std::byte const*>(n_bitmap.data()),
                       n_bitmap.size());
            bitmap_data_storage.push_back(std::move(bitmap_data));
            bitmap_spans.push_back(cudf::host_span<cuda::std::byte const>(
              bitmap_data_storage.back().data(), bitmap_data_storage.back().size()));
          } else {
            bitmap_spans.push_back(cudf::host_span<cuda::std::byte const>());
          }
        } else {
          bitmap_spans.push_back(cudf::host_span<cuda::std::byte const>());
        }
        env->DeleteLocalRef(j_bitmap);
      }
    }

    cudf::host_span<cudf::host_span<cuda::std::byte const> const> serialized_roaring_bitmaps(
      bitmap_spans.data(), bitmap_spans.size());

    // Convert deletion vector row counts
    cudf::host_span<cudf::size_type const> deletion_vector_row_counts;
    cudf::jni::native_jintArray n_deletion_vector_row_counts(env, j_deletion_vector_row_counts);
    if (j_deletion_vector_row_counts != nullptr && n_deletion_vector_row_counts.size() > 0) {
      deletion_vector_row_counts = cudf::host_span<cudf::size_type const>(
        n_deletion_vector_row_counts.data(), n_deletion_vector_row_counts.size());
    }

    // Convert row group offsets
    cudf::host_span<size_t const> row_group_offsets;
    cudf::jni::native_jlongArray n_row_group_offsets(env, j_row_group_offsets);
    if (j_row_group_offsets != nullptr && n_row_group_offsets.size() > 0) {
      row_group_offsets = cudf::host_span<size_t const>(
        reinterpret_cast<size_t const*>(n_row_group_offsets.data()), n_row_group_offsets.size());
    }

    // Convert row group num_rows
    cudf::host_span<cudf::size_type const> row_group_num_rows;
    cudf::jni::native_jintArray n_row_group_num_rows(env, j_row_group_num_rows);
    if (j_row_group_num_rows != nullptr && n_row_group_num_rows.size() > 0) {
      row_group_num_rows = cudf::host_span<cudf::size_type const>(n_row_group_num_rows.data(),
                                                                   n_row_group_num_rows.size());
    }

    // Create the chunked reader with multiple deletion vectors
    auto reader = new cudf::io::parquet::experimental::chunked_parquet_reader(
      static_cast<std::size_t>(j_chunk_read_limit),
      options,
      serialized_roaring_bitmaps,
      deletion_vector_row_counts,
      row_group_offsets,
      row_group_num_rows);

    return reinterpret_cast<jlong>(reader);
  }
  JNI_CATCH(env, 0);
}

/**
 * @brief Create a chunked Parquet reader with multiple deletion vectors and pass read limit
 *
 * This JNI function creates a chunked_parquet_reader with both chunk and pass read limits
 * and multiple deletion vectors.
 *
 * @param env JNI environment
 * @param j_chunk_read_limit Byte limit on returned table chunk size, 0 if no limit
 * @param j_pass_read_limit Byte limit on decompression memory, 0 if no limit
 * @param j_options_handle Handle to the parquet_reader_options object
 * @param j_serialized_roaring_bitmaps Array of serialized 64-bit roaring bitmaps
 * @param j_deletion_vector_row_counts Number of rows in each deletion vector
 * @param j_row_group_offsets Row index offsets for each row group
 * @param j_row_group_num_rows Number of rows in each row group
 * @return Handle to the chunked_parquet_reader (as jlong)
 */
JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_DeletionVector_createChunkedReaderWithMultipleDeletionVectorsAndPassLimit(
  JNIEnv* env,
  jclass,
  jlong j_chunk_read_limit,
  jlong j_pass_read_limit,
  jlong j_options_handle,
  jobjectArray j_serialized_roaring_bitmaps,
  jintArray j_deletion_vector_row_counts,
  jlongArray j_row_group_offsets,
  jintArray j_row_group_num_rows)
{
  JNI_NULL_CHECK(env, j_options_handle, "options handle is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    // Get the parquet_reader_options from the handle
    auto const& options =
      *reinterpret_cast<cudf::io::parquet_reader_options const*>(j_options_handle);

    // Convert array of serialized roaring bitmaps
    std::vector<std::vector<cuda::std::byte>> bitmap_data_storage;
    std::vector<cudf::host_span<cuda::std::byte const>> bitmap_spans;
    
    if (j_serialized_roaring_bitmaps != nullptr) {
      jsize num_bitmaps = env->GetArrayLength(j_serialized_roaring_bitmaps);
      bitmap_data_storage.reserve(num_bitmaps);
      bitmap_spans.reserve(num_bitmaps);
      
      for (jsize i = 0; i < num_bitmaps; i++) {
        auto j_bitmap = static_cast<jbyteArray>(env->GetObjectArrayElement(j_serialized_roaring_bitmaps, i));
        if (j_bitmap != nullptr) {
          cudf::jni::native_jbyteArray n_bitmap(env, j_bitmap);
          if (n_bitmap.size() > 0) {
            std::vector<cuda::std::byte> bitmap_data(n_bitmap.size());
            std::memcpy(bitmap_data.data(), 
                       reinterpret_cast<cuda::std::byte const*>(n_bitmap.data()),
                       n_bitmap.size());
            bitmap_data_storage.push_back(std::move(bitmap_data));
            bitmap_spans.push_back(cudf::host_span<cuda::std::byte const>(
              bitmap_data_storage.back().data(), bitmap_data_storage.back().size()));
          } else {
            bitmap_spans.push_back(cudf::host_span<cuda::std::byte const>());
          }
        } else {
          bitmap_spans.push_back(cudf::host_span<cuda::std::byte const>());
        }
        env->DeleteLocalRef(j_bitmap);
      }
    }

    cudf::host_span<cudf::host_span<cuda::std::byte const> const> serialized_roaring_bitmaps(
      bitmap_spans.data(), bitmap_spans.size());

    // Convert deletion vector row counts
    cudf::host_span<cudf::size_type const> deletion_vector_row_counts;
    cudf::jni::native_jintArray n_deletion_vector_row_counts(env, j_deletion_vector_row_counts);
    if (j_deletion_vector_row_counts != nullptr && n_deletion_vector_row_counts.size() > 0) {
      deletion_vector_row_counts = cudf::host_span<cudf::size_type const>(
        n_deletion_vector_row_counts.data(), n_deletion_vector_row_counts.size());
    }

    // Convert row group offsets
    cudf::host_span<size_t const> row_group_offsets;
    cudf::jni::native_jlongArray n_row_group_offsets(env, j_row_group_offsets);
    if (j_row_group_offsets != nullptr && n_row_group_offsets.size() > 0) {
      row_group_offsets = cudf::host_span<size_t const>(
        reinterpret_cast<size_t const*>(n_row_group_offsets.data()), n_row_group_offsets.size());
    }

    // Convert row group num_rows
    cudf::host_span<cudf::size_type const> row_group_num_rows;
    cudf::jni::native_jintArray n_row_group_num_rows(env, j_row_group_num_rows);
    if (j_row_group_num_rows != nullptr && n_row_group_num_rows.size() > 0) {
      row_group_num_rows = cudf::host_span<cudf::size_type const>(n_row_group_num_rows.data(),
                                                                   n_row_group_num_rows.size());
    }

    // Create the chunked reader with pass read limit and multiple deletion vectors
    auto reader = new cudf::io::parquet::experimental::chunked_parquet_reader(
      static_cast<std::size_t>(j_chunk_read_limit),
      static_cast<std::size_t>(j_pass_read_limit),
      options,
      serialized_roaring_bitmaps,
      deletion_vector_row_counts,
      row_group_offsets,
      row_group_num_rows);

    return reinterpret_cast<jlong>(reader);
  }
  JNI_CATCH(env, 0);
}

/**
 * @brief Check if the chunked reader has more data to read
 *
 * @param env JNI environment
 * @param j_reader_handle Handle to the chunked_parquet_reader
 * @return true if there is more data to read, false otherwise
 */
JNIEXPORT jboolean JNICALL
Java_com_nvidia_spark_rapids_jni_DeletionVector_chunkedReaderHasNext(JNIEnv* env,
                                                                     jclass,
                                                                     jlong j_reader_handle)
{
  JNI_NULL_CHECK(env, j_reader_handle, "reader handle is null", false);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const reader =
      reinterpret_cast<cudf::io::parquet::experimental::chunked_parquet_reader* const>(j_reader_handle);
    return reader->has_next();
  }
  JNI_CATCH(env, false);
}

/**
 * @brief Read the next chunk from the chunked reader
 *
 * @param env JNI environment
 * @param j_reader_handle Handle to the chunked_parquet_reader
 * @return Handle to the resulting table (as jlongArray)
 */
JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_DeletionVector_chunkedReaderReadChunk(JNIEnv* env,
                                                                       jclass,
                                                                       jlong j_reader_handle)
{
  JNI_NULL_CHECK(env, j_reader_handle, "reader handle is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const reader =
      reinterpret_cast<cudf::io::parquet::experimental::chunked_parquet_reader* const>(j_reader_handle);
    auto chunk = reader->read_chunk();
    return chunk.tbl ? cudf::jni::convert_table_for_return(env, chunk.tbl) : nullptr;
  }
  JNI_CATCH(env, nullptr);
}

/**
 * @brief Close and destroy the chunked reader
 *
 * @param env JNI environment
 * @param j_reader_handle Handle to the chunked_parquet_reader
 */
JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_DeletionVector_closeChunkedReader(
  JNIEnv* env, jclass, jlong j_reader_handle)
{
  JNI_NULL_CHECK(env, j_reader_handle, "reader handle is null", );

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete reinterpret_cast<cudf::io::parquet::experimental::chunked_parquet_reader*>(
      j_reader_handle);
  }
  JNI_CATCH(env, );
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_DeletionVector_destroyMultiHostBufferSource(
  JNIEnv* env, jclass, jlong handle)
{
  JNI_NULL_CHECK(env, handle, "handle is null", );

  JNI_TRY { delete reinterpret_cast<cudf::jni::multi_host_buffer_source*>(handle); }
  JNI_CATCH(env, );
}

}  // extern "C"
