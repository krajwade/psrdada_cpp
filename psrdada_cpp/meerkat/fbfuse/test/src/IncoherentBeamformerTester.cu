#include "psrdada_cpp/meerkat/fbfuse/test/IncoherentBeamformerTester.cuh"
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

IncoherentBeamformerTester::IncoherentBeamformerTester()
    : ::testing::Test()
    , _stream(0)
{

}

IncoherentBeamformerTester::~IncoherentBeamformerTester()
{

}

void IncoherentBeamformerTester::SetUp()
{
    CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
}

void IncoherentBeamformerTester::TearDown()
{
    CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
}

void IncoherentBeamformerTester::beamformer_c_reference(
    HostVoltageVectorType const& taftp_voltages,
    HostPowerVectorType& tf_powers,
    int nchannels,
    int tscrunch,
    int fscrunch,
    int ntimestamps,
    int nantennas,
    int npol,
    int nsamples_per_timestamp,
    HostScalingVectorType const& scale,
    HostScalingVectorType const& offset)
{
    const int tp = nsamples_per_timestamp * npol;
    const int ftp = nchannels * tp;
    const int aftp = nantennas * ftp;
    double power_sum = 0.0;
    double power_sq_sum = 0.0;
    std::size_t count = 0;

    for (int timestamp_idx = 0; timestamp_idx < ntimestamps; ++timestamp_idx)
    {
        for (int subint_idx = 0; subint_idx < nsamples_per_timestamp/tscrunch; ++subint_idx)
        {
            int subint_start = subint_idx * tscrunch;
            for (int subband_idx = 0; subband_idx < nchannels/fscrunch; ++subband_idx)
            {
                int subband_start = subband_idx * fscrunch;
                {
                    float xx = 0.0f, yy = 0.0f;
                    for (int antenna_idx = 0; antenna_idx < nantennas; ++antenna_idx)
                    {
                        for (int channel_idx = subband_start; channel_idx < subband_start + fscrunch;  ++channel_idx)
                        {
                            for (int sample_idx = subint_start; sample_idx < subint_start + tscrunch; ++sample_idx)
                            {
                                for (int pol_idx = 0; pol_idx < npol; ++pol_idx)
                                {
                                    int input_index = timestamp_idx * aftp + antenna_idx * ftp + channel_idx * tp + sample_idx * npol + pol_idx;
                                    char2 ant = taftp_voltages[input_index];
                                    xx += ((float) ant.x) * ant.x;
                                    yy += ((float) ant.y) * ant.y;
                                }
                            }
                        }
                    }
                    int time_idx = timestamp_idx * nsamples_per_timestamp/tscrunch + subint_idx;
                    int output_idx = time_idx * nchannels/fscrunch + subband_idx;
                    float power = (xx + yy);
                    power_sum += power;
                    power_sq_sum += power * power;
                    ++count;
                    float scaled_power = ((power - offset[subband_idx]) / scale[subband_idx]);
		    tf_powers[output_idx] = (int8_t) fmaxf(-127.0f, fminf(127.0f, scaled_power));
                }
            }
        }
    }
    double power_mean = power_sum / count;
    BOOST_LOG_TRIVIAL(debug) << "Average power level: " << power_mean;
    BOOST_LOG_TRIVIAL(debug) << "Power variance: " << power_sq_sum / count - power_mean * power_mean;
}

void IncoherentBeamformerTester::compare_against_host(
    DeviceVoltageVectorType const& taftp_voltages_gpu,
    DevicePowerVectorType& tf_powers_gpu,
    DeviceScalingVectorType const& scaling_vector,
    DeviceScalingVectorType const& offset_vector,
    int ntimestamps)
{
    HostVoltageVectorType taftp_voltages_host = taftp_voltages_gpu;
    HostPowerVectorType tf_powers_cuda = tf_powers_gpu;
    HostPowerVectorType tf_powers_host(tf_powers_gpu.size());

    HostScalingVectorType h_scaling_vector = scaling_vector;
    HostScalingVectorType h_offset_vector = offset_vector;

    beamformer_c_reference(taftp_voltages_host,
        tf_powers_host,
        _config.nchans(),
        _config.ib_tscrunch(),
        _config.ib_fscrunch(),
        ntimestamps,
        _config.ib_nantennas(),
        _config.npol(),
        _config.nsamples_per_heap(),
        h_scaling_vector,
        h_offset_vector);
    for (int ii = 0; ii < tf_powers_host.size(); ++ii)
    {
        ASSERT_TRUE(std::abs(static_cast<int>(tf_powers_host[ii]) - tf_powers_cuda[ii]) <= 1);
    }
}

TEST_F(IncoherentBeamformerTester, ib_representative_noise_test)
{
    float input_level = 32.0f;
    _config.output_level(32.0f);
    std::default_random_engine generator;
    std::normal_distribution<float> normal_dist(0.0, 32.0f);
    IncoherentBeamformer incoherent_beamformer(_config);
    std::size_t ntimestamps = 32;
    std::size_t input_size = (ntimestamps * _config.ib_nantennas()
        * _config.nchans() * _config.nsamples_per_heap() * _config.npol());
    HostVoltageVectorType taftp_voltages_host(input_size);
    for (int ii = 0; ii < taftp_voltages_host.size(); ++ii)
    {
        taftp_voltages_host[ii].x = static_cast<int8_t>(std::lround(normal_dist(generator)));
        taftp_voltages_host[ii].y = static_cast<int8_t>(std::lround(normal_dist(generator)));
    }

    float ib_scale = std::pow(input_level, 2);
    float ib_dof = 2 * _config.ib_tscrunch() * _config.ib_fscrunch() * _config.ib_nantennas() * _config.npol();
    float ib_power_offset  = ib_scale * ib_dof;
    float ib_power_scaling = ib_scale * std::sqrt(2 * ib_dof) / _config.output_level();
    /*
    printf("Nantennas: %d, tscrunch: %d, fscrunch: %d, npol: %d, Output level: %f, Input level: %f, Scale val: %f, Offset val: %f\n",
           _config.ib_nantennas(), _config.ib_tscrunch(), _config.ib_fscrunch(), _config.npol(),
           _config.output_level(), input_level, ib_power_scaling, ib_power_offset);
    */
    DeviceScalingVectorType scales(_config.nchans() / _config.ib_fscrunch(), ib_power_scaling);
    DeviceScalingVectorType offset(_config.nchans() / _config.ib_fscrunch(), ib_power_offset);
    DeviceVoltageVectorType taftp_voltages_gpu = taftp_voltages_host;
    DevicePowerVectorType tf_powers_gpu;
    incoherent_beamformer.beamform(taftp_voltages_gpu, tf_powers_gpu, scales, offset, _stream);
    compare_against_host(taftp_voltages_gpu, tf_powers_gpu, scales, offset, ntimestamps);
}

} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

