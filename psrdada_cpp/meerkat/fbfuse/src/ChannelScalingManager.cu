#include "psrdada_cpp/meerkat/fbfuse/ChannelScalingManager.cuh"
#include "psrdada_cpp/cuda_utils.hpp"
#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <errno.h>
#include <cstring>

#define LOG2_FBFUSE_NSAMPLES_PER_HEAP 8

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace kernel{

__global__ void calculate_std(
        char4 const* __restrict__ taftp_voltages,
        float* __restrict__ input_level,
        std::size_t nsamples
        )
{
    // TAFTP format
    float __shared__ std_estimates[FBFUSE_NSAMPLES_PER_HEAP];
    unsigned const freq_idx = blockIdx.x;
    double sum = 0.0;
    double sum_sq = 0.0;

    for (std::size_t ii=0; ii < nsamples; ++ii)
    {
        for (std::size_t jj = 0; jj < FBFUSE_TOTAL_NANTENNAS; ++jj)
        {
            std::size_t idx = threadIdx.x + (FBFUSE_NSAMPLES_PER_HEAP)*freq_idx + ii * FBFUSE_NSAMPLES_PER_HEAP * FBFUSE_TOTAL_NANTENNAS * FBFUSE_NCHANS +     jj*FBFUSE_NSAMPLES_PER_HEAP*FBFUSE_NCHANS;
            char4 temp = taftp_voltages[idx];
            sum += (temp.x + temp.y + temp.w + temp.z);
            sum_sq += (temp.x*temp.x + temp.y*temp.y + temp.w*temp.w + temp.z*temp.z);
        }
    }

    // STD_deviation estimate
    std_estimates[threadIdx.x] = std::sqrt((sum_sq/(nsamples*FBFUSE_TOTAL_NANTENNAS*FBFUSE_NPOL*2) - (sum/(nsamples*FBFUSE_TOTAL_NANTENNAS*FBFUSE_NPOL*2) * sum/(nsamples*FBFUSE_TOTAL_NANTENNAS*FBFUSE_NPOL*2))));

    // Parallel reduction of the std_estimates
    for (std::uint32_t ii = 0; ii < LOG2_FBFUSE_NSAMPLES_PER_HEAP; ++ii)
    {
        std::size_t shift = 1 << ii;
        if ((shift + threadIdx.x) < FBFUSE_NSAMPLES_PER_HEAP)
        {
            float val = (std_estimates[threadIdx.x] +
                std_estimates[threadIdx.x + shift]);
            __syncthreads();
            std_estimates[threadIdx.x] = val;
            __syncthreads();
        }
    }
    if (threadIdx.x == 0)
        input_level[freq_idx] = std_estimates[threadIdx.x]/FBFUSE_NSAMPLES_PER_HEAP;
}

} // kernel


ChannelScalingManager::ChannelScalingManager(PipelineConfig const& config, cudaStream_t stream)
    : _config(config)
    , _stream(stream)
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
    _channel_input_levels.resize(FBFUSE_NCHANS);
    _cb_offsets.resize(FBFUSE_NCHANS/FBFUSE_CB_FSCRUNCH);
    _cb_scaling.resize(FBFUSE_NCHANS/FBFUSE_CB_FSCRUNCH);
    _ib_offsets.resize(FBFUSE_NCHANS/FBFUSE_IB_FSCRUNCH);
    _ib_scaling.resize(FBFUSE_NCHANS/FBFUSE_IB_FSCRUNCH);
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

void ChannelScalingManager::channel_statistics(thrust::device_vector<char2> const& taftp_voltages)
{
    // This function should return the delays in GPU memory
    // First check if we need to update GPU memory
    if (update_available())
    {
        // define host arrays
        thrust::host_vector<float> h_input_levels(FBFUSE_NCHANS);
        thrust::host_vector<float> h_cb_offsets(FBFUSE_NCHANS/FBFUSE_CB_FSCRUNCH);
        thrust::host_vector<float> h_ib_offsets(FBFUSE_NCHANS/FBFUSE_IB_FSCRUNCH);
        thrust::host_vector<float> h_cb_scaling(FBFUSE_NCHANS/FBFUSE_CB_FSCRUNCH);
        thrust::host_vector<float> h_ib_scaling(FBFUSE_NCHANS/FBFUSE_IB_FSCRUNCH);
        std::size_t n_per_timestamp = FBFUSE_NCHANS * FBFUSE_TOTAL_NANTENNAS * FBFUSE_NPOL * FBFUSE_NSAMPLES_PER_HEAP;
        assert(taftp_voltages.size() % n_per_timestamp == 0 /* TAFTP voltages is not a multiple of AFTP size*/);
        std::size_t nsamples = taftp_voltages.size() / n_per_timestamp;
        // call GPU kernel
        char4 const* taftp_voltages_ptr = reinterpret_cast<char4 const*>(thrust::raw_pointer_cast(taftp_voltages.data()));
        float* input = thrust::raw_pointer_cast(_channel_input_levels.data());
        kernel::calculate_std<<<FBFUSE_NCHANS, FBFUSE_NSAMPLES_PER_HEAP, 0, _stream>>>(
                taftp_voltages_ptr,
                input,
                nsamples);
        CUDA_ERROR_CHECK(cudaStreamSynchronize(_stream));
        BOOST_LOG_TRIVIAL(debug) << "Finished running input levels kernel";
        // Copy input levels to host and calculate statistics
        thrust::copy(_channel_input_levels.begin(), _channel_input_levels.end(), h_input_levels.begin());
        BOOST_LOG_TRIVIAL(debug) << "Copied input levels to host";
        const float weights_amp = 127.0f;
        std::size_t reduced_nchans_ib = FBFUSE_NCHANS/FBFUSE_IB_FSCRUNCH;
        std::size_t reduced_nchans_cb = FBFUSE_NCHANS/FBFUSE_CB_FSCRUNCH;

        // define function
        auto get_offset_cb = [&](float x, float y)
        {
            float scale = std::pow(weights_amp * y * std::sqrt(static_cast<float>(_config.cb_nantennas())), 2);
            float dof = 2 * _config.cb_tscrunch() * _config.cb_fscrunch() * _config.npol();
            return x +  (scale * dof);
        };

        auto get_scale_cb = [&](float x, float y)
        {
            float scale = std::pow(weights_amp * y * std::sqrt(static_cast<float>(_config.cb_nantennas())), 2);
            float  dof = 2 * _config.cb_tscrunch() * _config.cb_fscrunch() * _config.npol();
            return x + (scale * std::sqrt(2 * dof) / _config.output_level());
        };

        auto get_offset_ib = [&](float x, float y)
        {
            float scale = std::pow(y, 2);
            float dof = 2 * _config.ib_tscrunch() * _config.ib_fscrunch() * _config.ib_nantennas() * _config.npol();
            return x + (scale * dof);
        };

        auto get_scale_ib = [&](float x, float y)
        {
            float scale = std::pow(y, 2);
            float  dof = 2 * _config.ib_tscrunch() * _config.ib_fscrunch() * _config.ib_nantennas() * _config.npol();
            return x + (scale * std::sqrt(2 * dof) / _config.output_level());
        };

        // CB scaling and  offsets
        for (std::uint32_t ii = 0; ii < reduced_nchans_cb; ++ii )
        {
            h_cb_offsets[ii] = std::accumulate(
                    &h_input_levels[FBFUSE_CB_FSCRUNCH*ii],
                    &h_input_levels[FBFUSE_CB_FSCRUNCH*ii + FBFUSE_CB_FSCRUNCH],
                    0.0f,
                    get_offset_cb
                    )/FBFUSE_CB_FSCRUNCH;

            h_cb_scaling[ii] = std::accumulate(
                    &h_input_levels[FBFUSE_CB_FSCRUNCH*ii],
                    &h_input_levels[FBFUSE_CB_FSCRUNCH*ii + FBFUSE_CB_FSCRUNCH],
                    0.0f,
                    get_scale_cb
                    )/FBFUSE_CB_FSCRUNCH;

            BOOST_LOG_TRIVIAL(debug) << "Coherent beam power offset: " << h_cb_offsets[ii];
            BOOST_LOG_TRIVIAL(debug) << "Coherent beam power scaling: " << h_cb_scaling[ii];
        }

            // scaling for incoherent beamformer
        for (std::uint32_t ii = 0; ii < reduced_nchans_ib; ++ii )
        {
            h_ib_offsets[ii] = std::accumulate(
                    &h_input_levels[FBFUSE_IB_FSCRUNCH*ii],
                    &h_input_levels[FBFUSE_IB_FSCRUNCH*ii + FBFUSE_IB_FSCRUNCH],
                    0.0f,
                    get_offset_ib
                    )/FBFUSE_IB_FSCRUNCH;

            h_ib_scaling[ii] = std::accumulate(
                    &h_input_levels[FBFUSE_IB_FSCRUNCH*ii],
                    &h_input_levels[FBFUSE_IB_FSCRUNCH*ii + FBFUSE_IB_FSCRUNCH],
                    0.0f,
                    get_scale_ib
                    )/FBFUSE_IB_FSCRUNCH;

            BOOST_LOG_TRIVIAL(debug) << "Incoherent beam power offset: " << h_ib_offsets[ii];
            BOOST_LOG_TRIVIAL(debug) << "Incoherent beam power scaling: " << h_ib_scaling[ii];
        }

        // Copying these back to the device
        _cb_offsets = h_cb_offsets;
        _cb_scaling = h_cb_scaling;
        _ib_offsets = h_ib_offsets;
        _ib_scaling = h_ib_scaling;

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
