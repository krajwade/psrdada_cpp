#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_CHANNELSCALINGMANAGERTEST_CUH
#define PSRDADA_CPP_MEERKAT_FBFUSE_CHANNELSCALINGMANAGERTEST_CUH

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/meerkat/fbfuse/ChannelScalingManager.cuh"
#include <gtest/gtest.h>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

class ChannelScalingTrigger
{
    public:
        ChannelScalingTrigger(PipelineConfig const& config);
        ~ChannelScalingTrigger();

        void request_statistics();

    private:
        sem_t* _count_sem;
        std::string _channel_scaling_sem;
};

class ChannelScalingManagerTester: public ::testing::Test
{
protected:
    void SetUp() override;
    void TearDown() override;

public:
    ChannelScalingManagerTester();
    ~ChannelScalingManagerTester();

protected:
    void compare_against_host(ChannelScalingManager::ScalingVectorType const&);
    void calculate_std_cpu(int nsamples, thrust::host_vector<char2> const& taftp_voltages, thrust::host_vector<float>& input_levels);

protected:
    PipelineConfig _config;
    cudaStream_t _stream;
};

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_FBFUSE_CHANNELSCALINGMANAGERTEST_CUH
