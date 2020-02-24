#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_VOLTAGESCALINGTESTER_CUH
#define PSRDADA_CPP_MEERKAT_FBFUSE_VOLTAGESCALINGTESTER_CUH

#include "psrdada_cpp/meerkat/fbfuse/VoltageScaling.cuh"
#include "thrust/host_vector.h"
#include <gtest/gtest.h>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

class VoltageScalingTester: public ::testing::Test
{
public:
    typedef thrust::host_vector<char2> HostVoltageVectorType;
    typedef thrust::device_vector<char2> DeviceVoltageVectorType;

    typedef thrust::host_vector<float2> HostGainsVectorType;
    typedef thrust::device_vector<float2> DeviceGainsVectorType;

    typedef thrust::host_vector<float> HostChannelScalesVectorType;
    typedef thrust::device_vector<float> DeviceChannelScalesVectorType;

protected:
    void SetUp() override;
    void TearDown() override;

public:
    VoltageScalingTester();
    ~VoltageScalingTester();

protected:
    void voltage_scaling_reference(
        HostVoltageVectorType& taftp_voltages_out,
        HostVoltageVectorType const& taftp_voltages_in,
        HostGainsVectorType const& afp_gains,
        HostChannelScalesVectorType const& f_channel_scalings);

    void compare_against_host(
        DeviceVoltageVectorType const& taftp_voltages_out,
        DeviceVoltageVectorType const& taftp_voltages_out,
        DeviceGainsVectorType const& afp_gains,
        DeviceChannelScalesVectorType const& f_channel_scalings,
        std::size_t nsamples);

protected:
    cudaStream_t _stream;
};

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_FBFUSE_VOLTAGESCALINGTESTER_CUH
