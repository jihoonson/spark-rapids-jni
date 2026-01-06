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
    assert buffers.length > 0;
    long[] addrsSizes = new long[buffers.length * 2];
    for (int i = 0; i < buffers.length; i++) {
      addrsSizes[i * 2] = buffers[i].getAddress();
      addrsSizes[(i * 2) + 1] = buffers[i].getLength();
    }

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
   * Chunked Parquet reader with deletion vector support.
   * 
   * This class allows reading large Parquet files in chunks while applying deletion vectors.
   * Each chunk is guaranteed to stay within the specified size limits. The reader maintains
   * state across chunk reads and applies the deletion vector consistently.
   */
  public static class ChunkedReader implements AutoCloseable {
    private long handle = 0;

    /**
     * Create a chunked Parquet reader with deletion vector support.
     * 
     * @param chunkReadLimit Byte limit on the returned table chunk size, or 0 if no limit
     * @param optionsHandle Native handle to a parquet_reader_options object
     * @param serializedRoaring64 Serialized 64-bit roaring bitmap (deletion vector)
     * @param rowGroupOffsets Row index offsets for each row group
     * @param rowGroupNumRows Number of rows in each row group
     */
    public ChunkedReader(long chunkReadLimit,
                        long optionsHandle,
                        byte[] serializedRoaring64,
                        long[] rowGroupOffsets,
                        int[] rowGroupNumRows) {
      this.handle = createChunkedParquetReader(chunkReadLimit,
                                               optionsHandle,
                                               serializedRoaring64,
                                               rowGroupOffsets,
                                               rowGroupNumRows);
      if (this.handle == 0) {
        throw new IllegalStateException("Failed to create native chunked reader");
      }
    }

    /**
     * Create a chunked Parquet reader with both chunk and pass read limits.
     * 
     * The pass read limit controls memory usage for decompression, allowing more fine-grained
     * control over resource usage during reading.
     * 
     * @param chunkReadLimit Byte limit on the returned table chunk size, or 0 if no limit
     * @param passReadLimit Byte limit on decompression memory, or 0 if no limit
     * @param optionsHandle Native handle to a parquet_reader_options object
     * @param serializedRoaring64 Serialized 64-bit roaring bitmap (deletion vector)
     * @param rowGroupOffsets Row index offsets for each row group
     * @param rowGroupNumRows Number of rows in each row group
     */
    public ChunkedReader(long chunkReadLimit,
                        long passReadLimit,
                        long optionsHandle,
                        byte[] serializedRoaring64,
                        long[] rowGroupOffsets,
                        int[] rowGroupNumRows) {
      this.handle = createChunkedParquetReaderWithPassLimit(chunkReadLimit,
                                                            passReadLimit,
                                                            optionsHandle,
                                                            serializedRoaring64,
                                                            rowGroupOffsets,
                                                            rowGroupNumRows);
      if (this.handle == 0) {
        throw new IllegalStateException("Failed to create native chunked reader");
      }
    }

    /**
     * Create a chunked Parquet reader with multiple deletion vectors support.
     * 
     * @param chunkReadLimit Byte limit on the returned table chunk size, or 0 if no limit
     * @param optionsHandle Native handle to a parquet_reader_options object
     * @param serializedRoaringBitmaps Array of serialized 64-bit roaring bitmaps (deletion vectors)
     * @param deletionVectorRowCounts Number of rows in each deletion vector
     * @param rowGroupOffsets Row index offsets for each row group
     * @param rowGroupNumRows Number of rows in each row group
     */
    public ChunkedReader(long chunkReadLimit,
                        long optionsHandle,
                        byte[][] serializedRoaringBitmaps,
                        int[] deletionVectorRowCounts,
                        long[] rowGroupOffsets,
                        int[] rowGroupNumRows) {
      this.handle = createChunkedReaderWithMultipleDeletionVectors(chunkReadLimit,
                                                                   optionsHandle,
                                                                   serializedRoaringBitmaps,
                                                                   deletionVectorRowCounts,
                                                                   rowGroupOffsets,
                                                                   rowGroupNumRows);
      if (this.handle == 0) {
        throw new IllegalStateException("Failed to create native chunked reader");
      }
    }

    /**
     * Create a chunked Parquet reader with multiple deletion vectors and pass read limit.
     * 
     * The pass read limit controls memory usage for decompression, allowing more fine-grained
     * control over resource usage during reading.
     * 
     * @param chunkReadLimit Byte limit on the returned table chunk size, or 0 if no limit
     * @param passReadLimit Byte limit on decompression memory, or 0 if no limit
     * @param optionsHandle Native handle to a parquet_reader_options object
     * @param serializedRoaringBitmaps Array of serialized 64-bit roaring bitmaps (deletion vectors)
     * @param deletionVectorRowCounts Number of rows in each deletion vector
     * @param rowGroupOffsets Row index offsets for each row group
     * @param rowGroupNumRows Number of rows in each row group
     */
    public ChunkedReader(long chunkReadLimit,
                        long passReadLimit,
                        long optionsHandle,
                        byte[][] serializedRoaringBitmaps,
                        int[] deletionVectorRowCounts,
                        long[] rowGroupOffsets,
                        int[] rowGroupNumRows) {
      this.handle = createChunkedReaderWithMultipleDeletionVectorsAndPassLimit(
                                                            chunkReadLimit,
                                                            passReadLimit,
                                                            optionsHandle,
                                                            serializedRoaringBitmaps,
                                                            deletionVectorRowCounts,
                                                            rowGroupOffsets,
                                                            rowGroupNumRows);
      if (this.handle == 0) {
        throw new IllegalStateException("Failed to create native chunked reader");
      }
    }

    /**
     * Check if there is more data to read.
     * 
     * @return true if there are more chunks to read, false otherwise
     */
    public boolean hasNext() {
      if (handle == 0) {
        throw new IllegalStateException("ChunkedReader has been closed");
      }
      return chunkedReaderHasNext(handle);
    }

    /**
     * Read the next chunk from the Parquet file.
     * 
     * Returns a Table containing the next chunk of data with the deletion vector applied
     * and index column prepended. The chunk size will not exceed the limit specified during
     * reader creation (unless a single row group exceeds the limit).
     * 
     * @return A Table containing the next chunk, or null if no more data
     */
    public Table readChunk() {
      if (handle == 0) {
        throw new IllegalStateException("ChunkedReader has been closed");
      }
      long[] columnPtrs = chunkedReaderReadChunk(handle);
      return columnPtrs == null ? null : new Table(columnPtrs);
    }

    @Override
    public void close() {
      if (handle != 0) {
        closeChunkedReader(handle);
        handle = 0;
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
  private static native long createChunkedParquetReader(long chunkReadLimit,
                                                       long optionsHandle,
                                                       byte[] serializedRoaring64,
                                                       long[] rowGroupOffsets,
                                                       int[] rowGroupNumRows)
    throws CudfException;

  /**
   * Native method to create a chunked Parquet reader with pass limit.
   */
  private static native long createChunkedParquetReaderWithPassLimit(long chunkReadLimit,
                                                                     long passReadLimit,
                                                                     long optionsHandle,
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
}
