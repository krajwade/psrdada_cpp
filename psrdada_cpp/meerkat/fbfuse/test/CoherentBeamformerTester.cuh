#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_COHERENTBEAMFORMERTESTER_CUH
#define PSRDADA_CPP_MEERKAT_FBFUSE_COHERENTBEAMFORMERTESTER_CUH

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/meerkat/fbfuse/CoherentBeamformer.cuh"
#include "thrust/host_vector.h"
#include <gtest/gtest.h>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

class CoherentBeamformerTester: public ::testing::Test
{
public:
    typedef CoherentBeamformer::VoltageVectorType DeviceVoltageVectorType;
    typedef thrust::host_vector<char2> HostVoltageVectorType;
    typedef CoherentBeamformer::PowerVectorType DevicePowerVectorType;
    typedef thrust::host_vector<char> HostPowerVectorType;
    typedef CoherentBeamformer::WeightsVectorType DeviceWeightsVectorType;
    typedef thrust::host_vector<char2> HostWeightsVectorType;
    typedef CoherentBeamformer::ScalingVectorType DeviceScalingVectorType;
    typedef thrust::host_vector<float> HostScalingVectorType;


protected:
    void SetUp() override;
    void TearDown() override;

public:
    CoherentBeamformerTester();
    ~CoherentBeamformerTester();

protected:
    void beamformer_c_reference(
        HostVoltageVectorType const& ftpa_voltages,
        HostWeightsVectorType const& fbpa_weights,
        HostPowerVectorType& btf_powers,
        int nchannels,
        int tscrunch,
        int fscrunch,
        int nsamples,
        int nbeams,
        int nantennas,
        int npol,
        float const* scales,
        float const* offsets);

    void compare_against_host(
        DeviceVoltageVectorType const& ftpa_voltages_gpu,
        DeviceWeightsVectorType const& fbpa_weights_gpu,
        DeviceScalingVectorType const& scales_gpu,
        DeviceScalingVectorType const& offsets_gpu,
        DevicePowerVectorType& btf_powers_gpu,
        int nsamples);

protected:
    PipelineConfig _config;
    cudaStream_t _stream;
};

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_FBFUSE_COHERENTBEAMFORMERTESTER_CUH
