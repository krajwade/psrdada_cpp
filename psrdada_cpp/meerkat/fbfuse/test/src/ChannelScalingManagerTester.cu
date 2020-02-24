#include "psrdada_cpp/meerkat/fbfuse/test/ChannelScalingManagerTester.cuh"
#include "psrdada_cpp/meerkat/fbfuse/fbfuse_constants.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "thrust/host_vector.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>
#include <sys/mman.h>
#include <sstream>
#include <random>
#include <cmath>
#include <complex>




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

ChannelScalingTrigger::ChannelScalingTrigger(PipelineConfig const& config)
: _channel_scaling_sem(config.channel_scaling_sem())
{
    _count_sem = sem_open(_channel_scaling_sem.c_str(), O_CREAT, 0666, 0);
    if (_count_sem == SEM_FAILED)
    {
        std::stringstream msg;
        msg << "Failed to open count semaphore "
        << _count_sem << " with error: "
        << std::strerror(errno);
        throw std::runtime_error(msg.str());
    }
}

ChannelScalingTrigger::~ChannelScalingTrigger()
{
    if (sem_close(_count_sem) == -1)
    {
        std::stringstream msg;
        msg << "Failed to close semaphore "
        << _count_sem << " with error: "
        << std::strerror(errno);
        BOOST_LOG_TRIVIAL(error) << msg.str();
    }
}

void ChannelScalingTrigger::request_statistics()
{
    sem_post(_count_sem);
}

ChannelScalingManagerTester::ChannelScalingManagerTester()
    : ::testing::Test()
    , _stream(0)
{
    _config.channel_scaling_sem("test_channel_scaling_sem");
}

ChannelScalingManagerTester::~ChannelScalingManagerTester()
{
}

void ChannelScalingManagerTester::SetUp()
{
    CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
}

void ChannelScalingManagerTester::TearDown()
{
    CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
}

void ChannelScalingManagerTester::calculate_std_cpu(int nsamples, thrust::host_vector<char2> const& taftp_voltages, thrust::host_vector<float>& input_levels)
{

    double sum;
    double sum_sq;
    std::vector<float> std_estimates(FBFUSE_NSAMPLES_PER_HEAP);

    for (std::uint64_t ll=0; ll < FBFUSE_NCHANS; ++ll)
    {
        for (std::uint64_t mm = 0; mm < FBFUSE_NSAMPLES_PER_HEAP; ++mm)
        {
            sum = 0.0;
            sum_sq = 0.0;
            for (std::uint64_t ii=0; ii < nsamples; ++ii)
            {
                for (std::uint64_t jj = 0; jj < FBFUSE_TOTAL_NANTENNAS; ++jj)
                {
                    for (std::uint64_t kk=0; kk < FBFUSE_NPOL; ++kk)
                    {
                        auto idx = FBFUSE_NPOL*mm +
                            (FBFUSE_NSAMPLES_PER_HEAP*FBFUSE_NPOL)*ll +
                            ii *(FBFUSE_NSAMPLES_PER_HEAP*FBFUSE_NPOL) * FBFUSE_TOTAL_NANTENNAS * FBFUSE_NCHANS +
                            jj*(FBFUSE_NSAMPLES_PER_HEAP*FBFUSE_NPOL*FBFUSE_NCHANS) + kk;

                        sum += (taftp_voltages[idx].x + taftp_voltages[idx].y);
                        sum_sq += pow(taftp_voltages[idx].x,2) + pow(taftp_voltages[idx].y, 2);
                    }
                }
            }
            std_estimates[mm] = std::sqrt((sum_sq/(nsamples*FBFUSE_TOTAL_NANTENNAS*FBFUSE_NPOL*2) - pow(sum/(nsamples*FBFUSE_TOTAL_NANTENNAS*FBFUSE_NPOL*2),2.0f)));
        }
        input_levels[ll] = std::accumulate(std_estimates.begin(), std_estimates.end(),0.0f)/(FBFUSE_NSAMPLES_PER_HEAP); 
    }

}

void ChannelScalingManagerTester::compare_against_host(ChannelScalingManager::ScalingVectorType const& input_levels)
{
    // Implicit sync copy back to host
    thrust::host_vector<float> host_input_levels = input_levels;
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    for (int ii=0; ii < FBFUSE_NCHANS; ++ii)
    {
        ASSERT_EQ(input_levels[ii], host_input_levels[ii]);
    }
}

TEST_F(ChannelScalingManagerTester, test_gaussian_noise)
{
    std::default_random_engine generator;
    std::normal_distribution<float> normal_dist(0.0, 32.0f);
    ChannelScalingTrigger trigger(_config);
    int nsamples = 16;
    //simulate noise
    thrust::host_vector<char2> taftp_voltages_host(nsamples*FBFUSE_TOTAL_NANTENNAS*FBFUSE_NCHANS*FBFUSE_NSAMPLES_PER_HEAP*FBFUSE_NPOL);
    thrust::host_vector<float> input_level(FBFUSE_NCHANS);
    for (int ii = 0; ii < taftp_voltages_host.size(); ++ii)
    {
        taftp_voltages_host[ii].x = static_cast<int8_t>(std::lround(normal_dist(generator)));
        taftp_voltages_host[ii].y = static_cast<int8_t>(std::lround(normal_dist(generator)));
    }
    // sync copy to the device
    thrust::device_vector<char2> taftp_voltages_gpu = taftp_voltages_host;
    // Run the kernel
    ChannelScalingManager level_manager(_config, _stream);
    // trigger a request
    trigger.request_statistics();
    // Get levels from the GPU
    BOOST_LOG_TRIVIAL(debug) << "Running Statistics kernel";
    level_manager.channel_statistics(taftp_voltages_gpu);
    // Get levels from the CPU
    calculate_std_cpu(nsamples, taftp_voltages_host, input_level);
    //check if they are the same
    for (std::size_t ii=0; ii < level_manager.channel_input_levels().size(); ++ii)
    {
       // BOOST_LOG_TRIVIAL(debug) << "cpu level:" << input_level[ii] << " gpu level: " << level_manager.channel_input_levels()[ii]; 
        ASSERT_NEAR((float)input_level[ii], (float)level_manager.channel_input_levels()[ii],input_level[ii]*0.0001);
    }
}


TEST_F(ChannelScalingManagerTester, test_bad_keys)
{
    ChannelScalingTrigger trigger(_config);
    _config.channel_scaling_sem("bad_test_delay_buffer_sem");
    ASSERT_ANY_THROW(ChannelScalingManager(_config, _stream));
}

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

