#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_GAINMANAGER_HPP
#define PSRDADA_CPP_MEERKAT_FBFUSE_GAINMANAGER_HPP

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include <thrust/device_vector.h>
#include <semaphore.h>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace kernels {

    __global__ void apply_gains(
        char4* __restrict__ taftp_voltages,
        float2 const* __restrict__ af_gains);

}

/**
 * @brief      Class for managing the POSIX shared memory buffers
 *             and semaphores wrapping the complex gain corrections
 *             for telescope phasing.
 */
class GainManager
{
public:
    typedef float2 ComplexGainType;
    typedef thrust::device_vector<DelayType> ComplexGainVectorType;
    typedef double TimeType;

public:
    /**
     * @brief      Create a new GainManager object
     *
     * @param      config  The pipeline configuration.
     *
     * @detail     The passed pipeline configuration contains the names
     *             of the POSIX shm and sem to connect to for the delay
     *             models.
     */
    GainManager(PipelineConfig const& config, cudaStream_t stream);
    ~GainManager();
    GainManager(GainManager const&) = delete;

    /**
     * @brief      Get the current complex gain corrections
     *
     * @detail     On a call to this function, a check is made on the
     *             delays counting semaphore to see if a delay model
     *             update is available. If so, the values are retrieved
     *             from shared memory and copied to the GPU. This function
     *             is not thread-safe!!!
     *
     * @return     A device vector containing the current delays
     */
    ComplexGainVectorType const& gains();

private:
    bool update_available();

private:
    PipelineConfig const& _config;
    cudaStream_t _copy_stream;
    int _gain_buffer_fd;
    sem_t* _gain_mutex_sem;
    sem_t* _gain_count_sem;
    int _last_sem_value;
    std::size_t _buffer_size;
    ComplexGainType* _gains_h;
    ComplexGainVectorType _gains;
};

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif // PSRDADA_CPP_MEERKAT_FBFUSE_GAINMANAGER_HPP



