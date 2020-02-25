#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_GAINMANAGERTEST_CUH
#define PSRDADA_CPP_MEERKAT_FBFUSE_GAINMANAGERTEST_CUH

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/meerkat/fbfuse/GainManager.cuh"
#include <gtest/gtest.h>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

class GainManagerTester: public ::testing::Test
{
protected:
    void SetUp() override;
    void TearDown() override;

public:
    GainManagerTester();
    ~GainManagerTester();

protected:
    void compare_against_host(GainManager::ComplexGainVectorType const&, ComplexGainType const*);

protected:
    PipelineConfig _config;
    cudaStream_t _stream;
};

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif //PSRDADA_CPP_MEERKAT_FBFUSE_GAINMANAGERTEST_CUH
