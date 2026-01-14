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

package com.nvidia.spark.rapids.jni;

import java.io.File;

import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.ParquetOptions;
import ai.rapids.cudf.Table;

/**
 * Provides JNI wrappers for reading Parquet files with deletion vector support.
 * 
 * Deletion vectors are used in Delta Lake and other table formats to track deleted rows
 * without physically rewriting data files. This class provides APIs to read Parquet files
 * while applying deletion vectors using 64-bit roaring bitmap serialization format.
 */
public class DeletionVector {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private static long[] computeAddrsAndSizes(HostMemoryBuffer... buffers) {
    assert buffers.length > 0;
    long[] addrsSizes = new long[buffers.length * 2];
    for (int i = 0; i < buffers.length; i++) {
      addrsSizes[i * 2] = buffers[i].getAddress();
      addrsSizes[(i * 2) + 1] = buffers[i].getLength();
    }
    return addrsSizes;
  }

  /**
   * Read a Parquet file with deletion vector support.
   * 
   * Reads a Parquet file, prepends an index column to the table, and applies the deletion vector
   * filter. If row group metadata is not provided, the index column will be a simple sequence
   * from 0 to the number of rows. If the deletion vector is null or empty, the table with the
   * prepended index column is returned as-is without filtering.
   * 
   * @param opts ParquetOptions
   * @param serializedRoaring64 Serialized 64-bit roaring bitmap in portable format representing
   *                            the deletion vector. Can be null or empty if no filtering is needed.
   * @param rowGroupOffsets Row index offsets for each row group. Used to build the index column.
   *                        Can be null or empty to use a simple 0-based sequence.
   * @param rowGroupNumRows Number of rows in each row group. Must match the length of
   *                        rowGroupOffsets if both are provided.
   * @return A Table containing the filtered data with a prepended UINT64 index column.
   * @throws CudfException if an error occurs during reading
   */
  public static Table readParquet(ParquetOptions opts,
                                  byte[] serializedRoaring64,
                                  long[] rowGroupOffsets,
                                  int[] rowGroupNumRows,
                                  HostMemoryBuffer... buffers) {
    long[] addrsSizes = computeAddrsAndSizes(buffers);
    long[] columnHandles = readParquetWithDeletionVector(opts.getIncludeColumnNames(),
                                                         opts.getReadBinaryAsString(),
                                                         null,
                                                         addrsSizes,
                                                         opts.timeUnit().typeId.getNativeId(),
                                                         serializedRoaring64,
                                                         rowGroupOffsets,
                                                         rowGroupNumRows);
    return columnHandles == null ? null : new Table(columnHandles);
  }

  /**
   * Read a Parquet file with multiple deletion vectors support.
   * 
   * Reads a Parquet file with multiple deletion vectors that can be applied to different
   * ranges of data as specified by the deletion vector row counts.
   * 
   * @param optionsHandle Native handle to a parquet_reader_options object
   * @param serializedRoaringBitmaps Array of serialized 64-bit roaring bitmaps
   * @param deletionVectorRowCounts Number of rows in each deletion vector
   * @param rowGroupOffsets Row index offsets for each row group
   * @param rowGroupNumRows Number of rows in each row group
   * @return A Table containing the filtered data with a prepended UINT64 index column
   * @throws CudfException if an error occurs during reading
   */
  public static Table readParquetWithDeletionVectors(long optionsHandle,
                                                             byte[][] serializedRoaringBitmaps,
                                                             int[] deletionVectorRowCounts,
                                                             long[] rowGroupOffsets,
                                                             int[] rowGroupNumRows) {
    long[] columnHandles = readParquetWithMultipleDeletionVectors(optionsHandle,
                                                                  serializedRoaringBitmaps,
                                                                  deletionVectorRowCounts,
                                                                  rowGroupOffsets,
                                                                  rowGroupNumRows);
    return columnHandles == null ? null : new Table(columnHandles);
  }

  /**
   * Provide an interface for reading a Parquet file in an iterative manner.
   */
  public static class ParquetChunkedReader implements AutoCloseable {

    /**
     * Auxiliary variable to help {@link #hasNext()} returning true at least once.
     */
    private boolean firstCall = true;

    /**
     * Handle for memory address of the native Parquet chunked reader class.
     */
    private long handle;

    private long dataSourceHandle = 0;

    private long multiHostBufferSourceHandle = 0;

    /**
     * Construct the reader instance from a read limit and a file path.
     *
     * @param chunkSizeByteLimit Limit on total number of bytes to be returned per read,
     *                           or 0 if there is no limit.
     * @param filePath Full path of the input Parquet file to read.
     */
    public ParquetChunkedReader(long chunkSizeByteLimit, File filePath,
                                byte[] serializedRoaring64,
                                long[] rowGroupOffsets,
                                int[] rowGroupNumRows) {
      this(chunkSizeByteLimit, ParquetOptions.DEFAULT, filePath, serializedRoaring64,
          rowGroupOffsets, rowGroupNumRows);
    }

    /**
     * Construct the reader instance from a read limit, a ParquetOptions object, and a file path.
     *
     * @param chunkSizeByteLimit Limit on total number of bytes to be returned per read,
     *                           or 0 if there is no limit.
     * @param opts The options for Parquet reading.
     * @param filePath Full path of the input Parquet file to read.
     */
    public ParquetChunkedReader(long chunkSizeByteLimit, ParquetOptions opts, File filePath,
                                byte[] serializedRoaring64,
                                long[] rowGroupOffsets,
                                int[] rowGroupNumRows) {
      this(chunkSizeByteLimit, 0, opts, filePath, serializedRoaring64,
          rowGroupOffsets, rowGroupNumRows);
    }

    /**
     * Construct the reader instance from a read limit, a ParquetOptions object, and a file path.
     *
     * @param chunkSizeByteLimit Limit on total number of bytes to be returned per read,
     *                           or 0 if there is no limit.
     * @param passReadLimit Limit on the amount of memory used for reading and decompressing data or
     *                      0 if there is no limit
     * @param opts The options for Parquet reading.
     * @param filePath Full path of the input Parquet file to read.
     */
    public ParquetChunkedReader(long chunkSizeByteLimit,
                                long passReadLimit, 
                                ParquetOptions opts, 
                                File filePath,
                                byte[] serializedRoaring64,
                                long[] rowGroupOffsets,
                                int[] rowGroupNumRows) {
      long[] handles = createChunkedParquetReaderWithPassLimit(chunkSizeByteLimit, passReadLimit,
        opts.getIncludeColumnNames(), opts.getReadBinaryAsString(),
        filePath.getAbsolutePath(), null, opts.timeUnit().typeId.getNativeId(),
      serializedRoaring64, rowGroupOffsets, rowGroupNumRows);
      handle = handles[0];
      if (handle == 0) {
        throw new IllegalStateException("Cannot create native chunked Parquet reader object.");
      }
      multiHostBufferSourceHandle = handles[1];
    }

    /**
     * Construct the reader instance from a read limit and data in host memory buffers.
     *
     * @param chunkSizeByteLimit Limit on total number of bytes to be returned per read,
     *                           or 0 if there is no limit.
     * @param passReadLimit Limit on the amount of memory used for reading and decompressing data or
     *                      0 if there is no limit
     * @param opts The options for Parquet reading.
     * @param buffers Array of buffers containing the file data. The buffers are logically
     *                concatenated to construct the file being read.
     */
    public ParquetChunkedReader(long chunkSizeByteLimit, long passReadLimit,
                                ParquetOptions opts,
                                byte[] serializedRoaring64,
                                long[] rowGroupOffsets,
                                int[] rowGroupNumRows, 
                                HostMemoryBuffer... buffers) {
      long[] addrsSizes = new long[buffers.length * 2];
      for (int i = 0; i < buffers.length; i++) {
        addrsSizes[i * 2] = buffers[i].getAddress();
        addrsSizes[(i * 2) + 1] = buffers[i].getLength();
      }

      long[] handles = createChunkedParquetReaderWithPassLimit(chunkSizeByteLimit, passReadLimit,
        opts.getIncludeColumnNames(), opts.getReadBinaryAsString(), null,
          addrsSizes, opts.timeUnit().typeId.getNativeId(),
          serializedRoaring64, rowGroupOffsets, rowGroupNumRows);
      handle = handles[0];
      if (handle == 0) {
        throw new IllegalStateException("Cannot create native chunked Parquet reader object.");
      }
      multiHostBufferSourceHandle = handles[1];
    }

    /**
     * Check if the given file has anything left to read.
     *
     * @return A boolean value indicating if there is more data to read from file.
     */
    public boolean hasNext() {
      if (handle == 0) {
        throw new IllegalStateException("Native chunked Parquet reader object may have been closed.");
      }

      if (firstCall) {
        // This function needs to return true at least once, so an empty table
        // (but having empty columns instead of no column) can be returned by readChunk()
        // if the input file has no row.
        firstCall = false;
        return true;
      }
      return chunkedReaderHasNext(handle);
    }

    /**
     * Read a chunk of rows in the given Parquet file such that the returning data has total size
     * does not exceed the given read limit. If the given file has no data, or all data has been read
     * before by previous calls to this function, a null Table will be returned.
     *
     * @return A table of new rows reading from the given file.
     */
    public Table readChunk() {
      if (handle == 0) {
        throw new IllegalStateException("Native chunked Parquet reader object may have been closed.");
      }

      long[] columnPtrs = chunkedReaderReadChunk(handle);
      return columnPtrs != null ? new Table(columnPtrs) : null;
    }

    @Override
    public void close() {
      if (handle != 0) {
        closeChunkedReader(handle);
        handle = 0;
      }
      // if (dataSourceHandle != 0) {
      //   DataSourceHelper.destroyWrapperDataSource(dataSourceHandle);
      //   dataSourceHandle = 0;
      // }
      if (multiHostBufferSourceHandle != 0) {
        destroyMultiHostBufferSource(multiHostBufferSourceHandle);
        multiHostBufferSourceHandle = 0;
      }
    }
  }

  // Native methods

  /**
   * Native method to read Parquet with deletion vector support.
   */
  private static native long[] readParquetWithDeletionVector(String[] filterColumnNames,
                                                             boolean[] binaryToString,
                                                             String filePath,
                                                             long[] addrsAndSizes,
                                                             int timeUnit,
                                                             byte[] serializedRoaring64,
                                                             long[] rowGroupOffsets,
                                                             int[] rowGroupNumRows)
    throws CudfException;

  /**
   * Native method to read Parquet with multiple deletion vectors support.
   */
  private static native long[] readParquetWithMultipleDeletionVectors(long optionsHandle,
                                                                      byte[][] serializedRoaringBitmaps,
                                                                      int[] deletionVectorRowCounts,
                                                                      long[] rowGroupOffsets,
                                                                      int[] rowGroupNumRows)
    throws CudfException;

  /**
   * Native method to create a chunked Parquet reader.
   */
  private static native long createChunkedParquetReader(String[] filterColumnNames,
                                                       boolean[] binaryToString,
                                                       String filePath,
                                                       long[] addrsAndSizes,
                                                       int timeUnit,
                                                       long chunkReadLimit,
                                                       byte[] serializedRoaring64,
                                                       long[] rowGroupOffsets,
                                                       int[] rowGroupNumRows)
    throws CudfException;

  /**
   * Native method to create a chunked Parquet reader with pass limit.
   */
  private static native long[] createChunkedParquetReaderWithPassLimit(long chunkReadLimit,
                                                                     long passReadLimit,
                                                                     String[] filterColumnNames,
                                                                     boolean[] binaryToString,
                                                                     String filePath,
                                                                     long[] addrsAndSizes,
                                                                     int timeUnit,
                                                                     byte[] serializedRoaring64,
                                                                     long[] rowGroupOffsets,
                                                                     int[] rowGroupNumRows)
    throws CudfException;

  /**
   * Native method to create a chunked Parquet reader with multiple deletion vectors.
   */
  private static native long createChunkedReaderWithMultipleDeletionVectors(long chunkReadLimit,
                                                                            long optionsHandle,
                                                                            byte[][] serializedRoaringBitmaps,
                                                                            int[] deletionVectorRowCounts,
                                                                            long[] rowGroupOffsets,
                                                                            int[] rowGroupNumRows)
    throws CudfException;

  /**
   * Native method to create a chunked Parquet reader with multiple deletion vectors and pass limit.
   */
  private static native long createChunkedReaderWithMultipleDeletionVectorsAndPassLimit(
                                                                     long chunkReadLimit,
                                                                     long passReadLimit,
                                                                     long optionsHandle,
                                                                     byte[][] serializedRoaringBitmaps,
                                                                     int[] deletionVectorRowCounts,
                                                                     long[] rowGroupOffsets,
                                                                     int[] rowGroupNumRows)
    throws CudfException;

  /**
   * Native method to check if chunked reader has more data.
   */
  private static native boolean chunkedReaderHasNext(long readerHandle) throws CudfException;

  /**
   * Native method to read next chunk from chunked reader.
   */
  private static native long[] chunkedReaderReadChunk(long readerHandle) throws CudfException;

  /**
   * Native method to close and destroy chunked reader.
   */
  private static native void closeChunkedReader(long readerHandle) throws CudfException;

  private static native void destroyMultiHostBufferSource(long handle);
}
