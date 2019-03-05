#ifndef PSRDADA_CPP_EFFELSBERG_EDD_GATEDSPECTROMETER_HPP
#define PSRDADA_CPP_EFFELSBERG_EDD_GATEDSPECTROMETER_HPP

#include "psrdada_cpp/effelsberg/edd/Unpacker.cuh"
#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/double_device_buffer.cuh"
#include "psrdada_cpp/double_host_buffer.cuh"
#include "psrdada_cpp/effelsberg/edd/DetectorAccumulator.cuh"

#include "thrust/device_vector.h"
#include "cufft.h"

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {


#define BIT_MASK(bit) (1L << (bit))
#define SET_BIT(value, bit) ((value) |= BIT_MASK(bit))
#define CLEAR_BIT(value, bit) ((value) &= ~BIT_MASK(bit))
#define TEST_BIT(value, bit) (((value)&BIT_MASK(bit)) ? 1 : 0)


/**
 @class GatedSpectrometer
 @brief Split data into two streams and create integrated spectra depending on
 bit set in side channel data.

 */
template <class HandlerType> class GatedSpectrometer {
public:
  typedef uint64_t RawVoltageType;
  typedef float UnpackedVoltageType;
  typedef float2 ChannelisedVoltageType;
  typedef int8_t IntegratedPowerType;

public:
  /**
   * @brief      Constructor
   *
   * @param      buffer_bytes A RawBytes object wrapping a DADA header buffer
   * @param      nSideChannels Number of side channel items in the data stream,
   * @param      selectedSideChannel Side channel item used for gating
   * @param      selectedBit bit of side channel item used for gating
   */
  GatedSpectrometer(std::size_t buffer_bytes, std::size_t nSideChannels,
                    std::size_t selectedSideChannel, std::size_t selectedBit,
                    std::size_t speadHeapSize, std::size_t fft_length,
                    std::size_t naccumulate, std::size_t nbits,
                    float input_level, float output_level,
                    HandlerType &handler);
  ~GatedSpectrometer();

  /**
   * @brief      A callback to be called on connection
   *             to a ring buffer.
   *
   * @detail     The first available header block in the
   *             in the ring buffer is provided as an argument.
   *             It is here that header parameters could be read
   *             if desired.
   *
   * @param      block  A RawBytes object wrapping a DADA header buffer
   */
  void init(RawBytes &block);

  /**
   * @brief      A callback to be called on acqusition of a new
   *             data block.
   *
   * @param      block  A RawBytes object wrapping a DADA data buffer output
   * are the integrated specttra with/without bit set.
   */
  bool operator()(RawBytes &block);

private:
  void process(thrust::device_vector<RawVoltageType> const &digitiser_raw,
               thrust::device_vector<RawVoltageType> const &sideChannelData,
               thrust::device_vector<IntegratedPowerType> &detected_G0,
               thrust::device_vector<IntegratedPowerType> &detected_G1);

private:
  std::size_t _buffer_bytes;
  std::size_t _fft_length;
  std::size_t _naccumulate;
  std::size_t _nbits;
  std::size_t _nSideChannels;
  std::size_t _selectedSideChannel;
  std::size_t _selectedBit;
  std::size_t _speadHeapSize;
  std::size_t _sideChannelSize;
  std::size_t _totalHeapSize;
  std::size_t _nHeaps;
  std::size_t _gapSize;
  std::size_t _dataBlockBytes;

  HandlerType &_handler;
  cufftHandle _fft_plan;
  int _nchans;
  int _call_count;
  std::unique_ptr<Unpacker> _unpacker;
  std::unique_ptr<DetectorAccumulator> _detector;

  DoubleDeviceBuffer<RawVoltageType> _raw_voltage_db;
  DoubleDeviceBuffer<IntegratedPowerType> _power_db_G0;
  DoubleDeviceBuffer<IntegratedPowerType> _power_db_G1;
  DoubleDeviceBuffer<RawVoltageType> _sideChannelData_db;

  thrust::device_vector<UnpackedVoltageType> _unpacked_voltage_G0;
  thrust::device_vector<UnpackedVoltageType> _unpacked_voltage_G1;
  thrust::device_vector<ChannelisedVoltageType> _channelised_voltage;

  DoublePinnedHostBuffer<IntegratedPowerType> _host_power_db;

  cudaStream_t _h2d_stream;
  cudaStream_t _proc_stream;
  cudaStream_t _d2h_stream;
};


/// Route the data in G0 to G1 if corresponding sideChannelData bit at bitpos is
/// set to 1.
/// The data in the other stream is set to 0.
__global__ void gating(float *G0, float *G1, const int64_t *sideChannelData,
                       size_t N, size_t heapSize, int64_t bitpos,
                       int64_t noOfSideChannels, int64_t selectedSideChannel);


} // edd
} // effelsberg
} // psrdada_cpp

#include "psrdada_cpp/effelsberg/edd/detail/GatedSpectrometer.cu"
#endif //PSRDADA_CPP_EFFELSBERG_EDD_GATEDSPECTROMETER_HPP
