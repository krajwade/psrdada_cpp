#include "psrdada_cpp/meerkat/fbfuse/ChannelScalingManager.cuh"
#include "psrdada_cpp/cuda_utils.hpp"
#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <errno.h>
#include <cstring>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace kernel{

__global__ void calculate_std(
        char const* __restrict__ taftp_voltages,
        float* __restrict__ input_level,
        int nsamples
        )
{
    // TAFTP format
    float __shared__ std_estimates[FBFUSE_NSAMPLES_PER_HEAP*2];
    int const freq_idx = blockIdx.x;
    double sum = 0.0;
    double sum_sq = 0.0;

    for (std::uint64_t ii=0; ii < nsamples; ++ii)
    {
        for (std::uint64_t jj = 0; jj < 64; ++jj)
        {
            for(std::uint64_t kk = 0; kk < FBFUSE_NPOL; ++kk)
            {
                auto idx = (threadIdx.x*FBFUSE_NPOL*2) + (FBFUSE_NSAMPLES_PER_HEAP*2*2)*freq_idx + ii * (FBFUSE_NSAMPLES_PER_HEAP*2*2) * 64 * FBFUSE_NCHANS/16 +     jj*(FBFUSE_NSAMPLES_PER_HEAP*FBFUSE_NPOL*2)*FBFUSE_NCHANS/16 + 2*kk;
                sum += (taftp_voltages[idx] + taftp_voltages[idx+1])/2;
                sum_sq += pow((taftp_voltages[idx] + taftp_voltages[idx+1])/2, 2);
            }
        }
    }

    // STD_deviation estimate
    std_estimates[threadIdx.x] = std::sqrt((sum_sq/(nsamples*64*FBFUSE_NPOL) - pow(sum/(nsamples*64*FBFUSE_NPOL),2.0f)));

    // Parallel reduction of the std_estimates
    for (std::uint32_t ii=0 ; ii < 8; ++ii)
    {
        std::size_t shift = pow(2,ii);
        if (shift < FBFUSE_NSAMPLES_PER_HEAP)
        {
            float val = (std_estimates[threadIdx.x] +
                std_estimates[threadIdx.x + shift])/2;
            __syncthreads();
            std_estimates[threadIdx.x] = val;
            __syncthreads();
        }
    }
    if (threadIdx.x == 0)
        input_level[freq_idx] = std_estimates[threadIdx.x];
}

} // kernel


ChannelScalingManager::ChannelScalingManager(PipelineConfig const& config, cudaStream_t stream)
    : _config(config)
    , _copy_stream(stream)
    , _last_sem_value(0)
{
    BOOST_LOG_TRIVIAL(debug) << "Constructing new ChannelScalingManager instance";
    BOOST_LOG_TRIVIAL(debug) << "Opening channel scaling counting semaphore";
    _channel_scaling_count_sem = sem_open(_config.channel_scaling_sem().c_str(), O_EXCL);
    if (_channel_scaling_count_sem == SEM_FAILED)
    {
        throw std::runtime_error(std::string(
            "Failed to open channel scaling counting semaphore: "
            ) + std::strerror(errno));
    }

    // Resize the GPU array for channel statistics
    _channel_input_levels.resize(FBFUSE_NCHANS/16);
    _cb_offsets.resize(FBFUSE_NCHANS/16);
    _cb_scaling.resize(FBFUSE_NCHANS/16);
    _ib_offsets.resize(FBFUSE_NCHANS/16);
    _ib_scaling.resize(FBFUSE_NCHANS/16);
}

ChannelScalingManager::~ChannelScalingManager()
{
    BOOST_LOG_TRIVIAL(debug) << "Destroying ChannelScalingManager instance";

    if (sem_close(_channel_scaling_count_sem) == -1)
    {
        throw std::runtime_error(std::string(
            "Failed to close counting semaphore with error: ")
            + std::strerror(errno));
    }
}

bool ChannelScalingManager::update_available()
{
    BOOST_LOG_TRIVIAL(debug) << "Checking for request for channel statistics update";
    int count;
    int retval = sem_getvalue(_channel_scaling_count_sem, &count);
    if (retval != 0)
    {
        throw std::runtime_error(std::string(
            "Unable to retrieve value of counting semaphore: ")
            + std::strerror(errno));
    }
    if (count == _last_sem_value)
    {
        BOOST_LOG_TRIVIAL(debug) << "No  update available";
        return false;
    }
    else
    {
        BOOST_LOG_TRIVIAL(debug) << "New channel input levels requested";
        if (_last_sem_value - count > 1)
        {
            // This implies that there has been an update since the function was last called and
            // we need to trigger a memcpy between the host and the device. This should acquire
            // the mutex during the copy.
            // We also check if we have somehow skipped and update.
            BOOST_LOG_TRIVIAL(warning) << "Semaphore value increased by " << (_last_sem_value - count)
            << " between checks (exepcted increase of 1)";
        }
        _last_sem_value = count;
        return true;
    }
}

void ChannelScalingManager::channel_statistics(int nsamples, thrust::device_vector<char2>& taftp_voltages)
{
    // This function should return the delays in GPU memory
    // First check if we need to update GPU memory
    if (update_available())
    {
        // define host arrays
        thrust::host_vector<float> h_input_levels(FBFUSE_NCHANS/16);
        thrust::host_vector<float> h_cb_offsets(FBFUSE_NCHANS/16);
        thrust::host_vector<float> h_ib_offsets(FBFUSE_NCHANS/16);
        thrust::host_vector<float> h_cb_scaling(FBFUSE_NCHANS/16);
        thrust::host_vector<float> h_ib_scaling(FBFUSE_NCHANS/16);

        // call GPU kernel
        char const* taftp_voltages_ptr = (char const*) thrust::raw_pointer_cast(taftp_voltages.data()); 
        float* input = thrust::raw_pointer_cast(_channel_input_levels.data());
        kernel::calculate_std<<<FBFUSE_NCHANS/16, FBFUSE_NSAMPLES_PER_HEAP, 0, _copy_stream>>>(
                taftp_voltages_ptr,
                input,
                nsamples);

        BOOST_LOG_TRIVIAL(debug) << "Finished running input levels kernel";
       // Copy input levels to host and calculate statistics
        thrust::copy(_channel_input_levels.begin(), _channel_input_levels.end(), h_input_levels.begin());
        BOOST_LOG_TRIVIAL(debug) << "Copied input levels to host";
        const float weights_amp = 127.0f;
        for (std::uint32_t ii = 0; ii < FBFUSE_NCHANS/16; ++ii )
        {
            float cb_scale = std::pow(weights_amp * h_input_levels[ii]
                    * std::sqrt(static_cast<float>(_config.cb_nantennas())), 2);
            float cb_dof = 2 * _config.cb_tscrunch() * _config.cb_fscrunch() * _config.npol();
            h_cb_offsets[ii] = cb_scale * cb_dof;
            h_cb_scaling[ii] = cb_scale * std::sqrt(2 * cb_dof) / _config.output_level();
            BOOST_LOG_TRIVIAL(debug) << "Coherent beam power offset: " << h_cb_offsets[ii];
            BOOST_LOG_TRIVIAL(debug) << "Coherent beam power scaling: " << h_cb_scaling[ii];
            // scaling for incoherent beamformer
            float ib_scale = std::pow(h_input_levels[ii], 2);
            float ib_dof = 2 * _config.ib_tscrunch() * _config.ib_fscrunch() * _config.ib_nantennas() * _config.npol();
            h_ib_offsets[ii]  = ib_scale * ib_dof;
            h_ib_scaling[ii] = ib_scale * std::sqrt(2 * ib_dof) / _config.output_level();
            BOOST_LOG_TRIVIAL(debug) << "Incoherent beam power offset: " << h_ib_offsets[ii];
            BOOST_LOG_TRIVIAL(debug) << "Incoherent beam power scaling: " << h_ib_scaling[ii];
        }

        // Copying these back to the device
        h_cb_offsets = _cb_offsets;
        h_cb_scaling = _cb_scaling;
        h_ib_offsets = _ib_offsets;
        h_ib_scaling = _ib_scaling;
    }
}

ChannelScalingManager::ScalingVectorType ChannelScalingManager::channel_input_levels() const
{
    return _channel_input_levels;
}

ChannelScalingManager::ScalingVectorType ChannelScalingManager::cb_offsets() const
{
    return _cb_offsets;
}

ChannelScalingManager::ScalingVectorType ChannelScalingManager::cb_scaling() const
{
    return _cb_scaling;
}

ChannelScalingManager::ScalingVectorType ChannelScalingManager::ib_offsets() const
{
    return _ib_offsets;
}

ChannelScalingManager::ScalingVectorType ChannelScalingManager::ib_scaling() const
{
    return _ib_scaling;
}

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp
