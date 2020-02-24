#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_CHANNELSCALINGMANAGER_HPP
#define PSRDADA_CPP_MEERKAT_FBFUSE_CHANNELSCALINGMANAGER_HPP

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include <thrust/device_vector.h>
#include <semaphore.h>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {

/**
 * @brief      Class for managing channel scaling calculations
 *             Checks for semaphore increments and then triggers
 *             calculation of statistics and saves them on device.
 */
class ChannelScalingManager
{
public:
    typedef float ScalingType;
    typedef thrust::device_vector<ScalingType> ScalingVectorType;
    typedef double TimeType;

public:
    /**
     * @brief      Create a new ChannelScalingManager object
     *
     * @param      config  The pipeline configuration.
     *
     * @detail     The passed pipeline configuration contains the names
     *             of the sem to connect to for the channel statistics
     */
    ChannelScalingManager(PipelineConfig const& config, cudaStream_t stream);
    ~ChannelScalingManager();
    ChannelScalingManager(ChannelScalingManager const&) = delete;

    /**
     * @brief      Get the current channel scalings
     *
     * @detail     On a call to this function, a check is made on the
     *             channel scaling counting semaphore to see if new channel statistics
     *             have been requested. If so, the values are retrieved
     *             from shared memory and copied to the GPU. This function
     *             is not thread-safe!!!
     */
    void channel_statistics(int nsamples, thrust::device_vector<char2>& taftp_voltages);

    /**
     * @brief      Return the current channel input levels
     */
    ScalingVectorType channel_input_levels() const;


    /**
     * @brief      Return the current coherent beam offsets
     */
    ScalingVectorType cb_offsets() const;

    /**
     * @brief      Return the current coherent beam scaling
     */
    ScalingVectorType cb_scaling() const;

    /**
     * @brief      Return the current incoherent beam offsets
     */
    ScalingVectorType ib_offsets() const;

    /**
     * @brief      Return the current incoherent beam scaling
     */
    ScalingVectorType ib_scaling() const;


private:
    bool update_available();

private:
    PipelineConfig const& _config;
    cudaStream_t _copy_stream;
    sem_t* _channel_scaling_mutex_sem;
    sem_t* _channel_scaling_count_sem;
    int _last_sem_value;
    ScalingVectorType _channel_input_levels;
    ScalingVectorType _cb_offsets;
    ScalingVectorType _cb_scaling;
    ScalingVectorType _ib_offsets;
    ScalingVectorType _ib_scaling;
    
};

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif // PSRDADA_CPP_MEERKAT_FBFUSE_CHANNELSCALINGMANAGER_HPP



