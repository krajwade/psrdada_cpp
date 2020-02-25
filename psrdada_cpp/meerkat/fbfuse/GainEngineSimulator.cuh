#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_GAINENGINESIMULATOR_HPP
#define PSRDADA_CPP_MEERKAT_FBFUSE_GAINENGINESIMULATOR_HPP

#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/meerkat/fbfuse/GainManager.cuh"
#include <semaphore.h>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {

class GainEngineSimulator
{
public:
    /**
     * @brief      Create a new GainEngineSimulator object
     *
     * @param      config  The pipeline configuration.
     *
     */
    explicit GainEngineSimulator(PipelineConfig const& config);
    ~GainEngineSimulator();
    GainEngineSimulator(GainEngineSimulator const&) = delete;

    /**
     * @brief      Simulate an update to the gain model by the control system
     */
    void update_gains();

    /**
     * @brief      Return a pointer to the gain model
     */
    GainManager::ComplexGainType* gains();

private:
    std::string const _gain_buffer_shm;
    std::string const _gain_buffer_sem;
    std::string const _gain_buffer_mutex;
    GainManager::ComplexGainType* _gain_model;
    int _shm_fd;
    void* _shm_ptr;
    sem_t* _sem_id;
    sem_t* _mutex_id;
};

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif // PSRDADA_CPP_MEERKAT_FBFUSE_GAINENGINESIMULATOR_HPP



