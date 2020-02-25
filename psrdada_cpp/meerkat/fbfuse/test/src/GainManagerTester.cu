#include "psrdada_cpp/meerkat/fbfuse/test/GainManagerTester.cuh"
#include "psrdada_cpp/meerkat/fbfuse/GainEngineSimulator.cuh"
#include "psrdada_cpp/meerkat/fbfuse/fbfuse_constants.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "thrust/host_vector.h"

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

GainManagerTester::GainManagerTester()
    : ::testing::Test()
    , _stream(0)
{
    _config.gain_buffer_shm("test_gain_buffer_shm");
    _config.gain_buffer_sem("test_gain_buffer_sem");
    _config.gain_buffer_mutex("test_gain_buffer_mutex");
}

GainManagerTester::~GainManagerTester()
{
}

void GainManagerTester::SetUp()
{
    CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
}

void GainManagerTester::TearDown()
{
    CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
}

void GainManagerTester::compare_against_host(GainManager::ComplexGainVectorType const& gains, GainManager::ComplexGainType const* expected_gains)
{
    // Implicit sync copy back to host
    thrust::host_vector<GainManager::GainType> host_gains = gains;
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    for (int ii=0; ii < FBFUSE_CB_NBEAMS * FBFUSE_CB_NANTENNAS; ++ii)
    {
        ASSERT_EQ(expected_gains[ii].x, host_gains[ii].x);
        ASSERT_EQ(expected_gains[ii].y, host_gains[ii].y);
    }
}

TEST_F(GainManagerTester, test_updates)
{
    GainEngineSimulator simulator(_config);
    GainManager gain_manager(_config, _stream);
    simulator.update_gains();
    auto const& gain_vector = gain_manager.gains();
    compare_against_host(gain_vector, simulator.gains());
    std::size_t expected_bytes = _config.nantennas() * _config.nchans() * _config.npol() * sizeof(GainManager::ComplexGainType);
    std::memset(static_cast<void*>(simulator.gains()), 1, expected_bytes);
    simulator.update_gains();
    auto const& gain_vector_2 = gain_manager.gains();
    compare_against_host(gain_vector_2, simulator.gains());
}

TEST_F(GainManagerTester, test_bad_keys)
{
    GainEngineSimulator simulator(_config);
    _config.gain_buffer_shm("bad_test_gain_buffer_shm");
    _config.gain_buffer_sem("bda_test_gain_buffer_sem");
    _config.gain_buffer_mutex("bad_test_gain_buffer_mutex");
    ASSERT_ANY_THROW(GainManager(_config, _stream));
}

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

