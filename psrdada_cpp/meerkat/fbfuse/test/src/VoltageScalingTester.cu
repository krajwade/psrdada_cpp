#include "psrdada_cpp/meerkat/fbfuse/test/VoltageScalingTester.cuh"
#include "psrdada_cpp/meerkat/fbfuse/fbfuse_constants.hpp"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include <random>
#include <cmath>
#include <complex>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

VoltageScalingTester::VoltageScalingTester()
    : ::testing::Test()
    , _stream(0)
{

}

VoltageScalingTester::~VoltageScalingTester()
{

}

void VoltageScalingTester::SetUp()
{
    CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
}

void VoltageScalingTester::TearDown()
{
    CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
}

void VoltageScalingTester::voltage_scaling_reference(
    HostVoltageVectorType& taftp_voltages_out,
    HostVoltageVectorType const& taftp_voltages_in,
    HostGainsVectorType const& afp_gains,
    HostChannelScalesVectorType const& f_channel_scalings,
    std::size_t n_outer_t,
    std::size_t nantennas,
    std::size_t nchans)
{
    char4* output = reinterpret_cast<char4*>(thrust::raw_pointer_cast(taftp_voltages_out.data()));
    char4 const* input = reinterpret_cast<char4 const*>(thrust::raw_pointer_cast(taftp_voltages_in.data()));
    float4 const* f4gains = reinterpret_cast<float4 const*>(thrust::raw_pointer_cast(afp_gains.data()));
    std::size_t const t = FBFUSE_NSAMPLES_PER_HEAP;
    std::size_t const ft = nchans * t;
    std::size_t const aft = nantennas * ft;

    for (std::size_t outer_t_idx = 0; outer_t_idx < n_outer_t; ++outer_t_idx)
    {
        for (std::size_t antenna_idx = 0; antenna_idx < nantennas; ++antenna_idx)
        {
            for (std::size_t channel_idx = 0; channel_idx < nchans; ++channel_idx)
            {
                for (std::size_t inner_t_idx = 0; inner_t_idx < t; ++inner_t_idx)
                {
                    std::size_t idx = outer_t_idx * aft + antenna_idx * ft + channel_idx * t + inner_t_idx;
                    std::size_t gain_idx = antenna_idx * nchans + channel_idx;
                    char4 data = input[idx];
                    float4 gain = f4gains[gain_idx];
                    float scaling = f_channel_scalings[channel_idx];
                    float xx = data.x * gain.x;
                    float yy = data.y * gain.y;
                    float xy = data.x * gain.y;
                    float yx = data.y * gain.x;
                    float zz = data.z * gain.z;
                    float ww = data.w * gain.w;
                    float zw = data.z * gain.w;
                    float wz = data.w * gain.z;
                    data.x = (int8_t) (fmaxf(-127.0f, fminf(127.0f, (scaling * (xx - yy)))) + 0.5f);
                    data.y = (int8_t) (fmaxf(-127.0f, fminf(127.0f, (scaling * (xy + yx)))) + 0.5f);
                    data.z = (int8_t) (fmaxf(-127.0f, fminf(127.0f, (scaling * (zz - ww)))) + 0.5f);
                    data.w = (int8_t) (fmaxf(-127.0f, fminf(127.0f, (scaling * (zw + wz)))) + 0.5f);
                    output[idx] = data;
                }
            }
        }
    }
}

void VoltageScalingTester::compare_against_host(
    DeviceVoltageVectorType & taftp_voltages_out,
    DeviceVoltageVectorType const& taftp_voltages_in,
    DeviceGainsVectorType const& afp_gains,
    DeviceChannelScalesVectorType const& f_channel_scalings,
    std::size_t nsamples)
{
    HostVoltageVectorType h_taftp_voltages_out(taftp_voltages_out.size());
    HostVoltageVectorType h_taftp_voltages_in = taftp_voltages_in;
    HostGainsVectorType h_afp_gains = afp_gains;
    HostChannelScalesVectorType h_f_channel_scalings = f_channel_scalings;

    voltage_scaling_reference(
        h_taftp_voltages_out,
        h_taftp_voltages_in,
        h_afp_gains,
        h_f_channel_scalings,
        nsamples,
        FBFUSE_TOTAL_NANTENNAS,
        FBFUSE_NCHANS);
    HostVoltageVectorType h_taftp_voltages_out_orig = taftp_voltages_out;
    for (int ii = 0; ii < taftp_voltages_out.size(); ++ii)
    {
        ASSERT_EQ( h_taftp_voltages_out_orig[ii].x, h_taftp_voltages_out[ii].x);
        ASSERT_EQ( h_taftp_voltages_out_orig[ii].y, h_taftp_voltages_out[ii].y);
    }
}

TEST_F(VoltageScalingTester, representative_noise_test)
{
    const float input_level = 32.0f;
    const double pi = std::acos(-1);
    std::default_random_engine generator;
    std::normal_distribution<float> normal_dist(0.0, input_level);
    std::uniform_real_distribution<float> uniform_dist(0.0, 2*pi);

    std::size_t n_outer_t = 10;
    std::size_t input_size = n_outer_t * FBFUSE_TOTAL_NANTENNAS * FBFUSE_NCHANS * FBFUSE_NSAMPLES_PER_HEAP * FBFUSE_NPOL;

    HostVoltageVectorType h_input(input_size);
    HostGainsVectorType h_gains(FBFUSE_TOTAL_NANTENNAS * FBFUSE_NCHANS * FBFUSE_NPOL);
    HostChannelScalesVectorType h_scales(FBFUSE_NCHANS, 1.0f);

    for (int ii = 0; ii < h_input.size(); ++ii)
    {
        h_input[ii].x = static_cast<int8_t>(std::lround(normal_dist(generator)));
        h_input[ii].y = static_cast<int8_t>(std::lround(normal_dist(generator)));
    }

    for (int ii = 0; ii < h_gains.size(); ++ii)
    {
        // Build complex weight as C * exp(i * theta).
        std::complex<float> val = std::exp(std::complex<float>(0.0f, uniform_dist(generator)));
        h_gains[ii].x = val.real();
        h_gains[ii].y = val.imag();
    }
    DeviceVoltageVectorType d_input = h_input;
    DeviceVoltageVectorType d_output(d_input.size());
    DeviceGainsVectorType d_gains = h_gains;
    DeviceChannelScalesVectorType d_scales = h_scales;
    voltage_scaling(d_output, d_input, d_gains, d_scales, _stream);
    compare_against_host(d_output, d_input, d_gains, d_scales, n_outer_t);
}

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

