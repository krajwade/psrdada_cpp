#include "psrdada_cpp/meerkat/fbfuse/GainManager.cuh"
#include "psrdada_cpp/cuda_utils.hpp"
#include <sys/mman.h>
#include <sys/stat.h>        /* For mode constants */
#include <fcntl.h>           /* For O_* constants */
#include <errno.h>
#include <cstring>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {

GainManager::GainManager(PipelineConfig const& config, cudaStream_t stream)
    : _config(config)
    , _copy_stream(stream)
    , _last_sem_value(0)
    , _buffer_size(sizeof(ComplexGainType) * FBFUSE_NPOL * FBFUSE_TOTAL_NANTENNAS)
{
    BOOST_LOG_TRIVIAL(debug) << "Constructing new GainManager instance";
    BOOST_LOG_TRIVIAL(debug) << "Opening gain buffer shared memory segement";
    // First we open file descriptors to all shared memory segments and semaphores
    _gain_buffer_fd = shm_open(_config.gain_buffer_shm().c_str(), O_RDONLY, 0);
    if (_gain_buffer_fd == -1)
    {
        throw std::runtime_error(std::string(
            "Failed to open gain buffer shared memory (")
            + _config.gain_buffer_shm() + "): "
            + std::strerror(errno));
    }
    BOOST_LOG_TRIVIAL(debug) << "Opening gain buffer mutex semaphore";
    _gain_mutex_sem = sem_open(_config.gain_buffer_mutex().c_str(), O_EXCL);
    if (_gain_mutex_sem == SEM_FAILED)
    {
        throw std::runtime_error(std::string(
            "Failed to open gain buffer mutex semaphore (")
            + _config.gain_buffer_mutex() + "): "
            + std::strerror(errno));
    }
    BOOST_LOG_TRIVIAL(debug) << "Opening gain buffer counting semaphore";
    _gain_count_sem = sem_open(_config.gain_buffer_sem().c_str(), O_EXCL);
    if (_gain_count_sem == SEM_FAILED)
    {
        throw std::runtime_error(std::string(
            "Failed to open gain buffer counting semaphore (")
            + _config.gain_buffer_sem() + "): "
            + std::strerror(errno));
    }

    // Here we run fstat on the shared memory buffer to check that it is the right dimensions
    BOOST_LOG_TRIVIAL(debug) << "Verifying shared memory segment dimensions";
    struct stat mem_info;
    int retval = fstat(_gain_buffer_fd, &mem_info);
    if (retval == -1)
    {
        throw std::runtime_error(std::string(
            "Could not fstat the gain buffer shared memory: ")
            + std::strerror(errno));
    }
    if (mem_info.st_size != _buffer_size)
    {
        throw std::runtime_error(std::string(
            "Shared memory buffer had unexpected size: ")
            + std::to_string(mem_info.st_size));
    }

    // Here we memory map the buffer and cast to the expected format (GainModel POD struct)
    BOOST_LOG_TRIVIAL(debug) << "Memory mapping shared memory segment";
    _gains_h = static_cast<float2*>(mmap(NULL, _buffer_size, PROT_READ,
        MAP_SHARED, _gain_buffer_fd, 0));
    if (_gains_h == NULL)
    {
        throw std::runtime_error(std::string(
            "MMAP on gain model buffer returned a null pointer: ")
            + std::strerror(errno));
    }
    // Note: cudaHostRegister is not working below for a couple of reasons:
    //     1. The size is not a multiple of the page size on this machine
    //     2. The memory is not page aligned properly
    // This issue is solvable by changing the GainModel struct, but I will
    // only fix this if there is a strong performance need.
    //
    // To maximise the copy throughput for the gains we here register the host memory
    // BOOST_LOG_TRIVIAL(debug) << "Registering shared memory segement with CUDA driver";
    // CUDA_ERROR_CHECK(cudaHostRegister(static_cast<void*>(_gains_h->gains),
    //    sizeof(_gains_h->gains), cudaHostRegisterMapped));
    // Resize the GPU array for the gains
    _gains.resize(FBFUSE_NPOL * FBFUSE_TOTAL_NANTENNAS, float2{1.0f, 0.0f});
}

GainManager::~GainManager()
{
    BOOST_LOG_TRIVIAL(debug) << "Destroying GainManager instance";
    //CUDA_ERROR_CHECK(cudaHostUnregister(static_cast<void*>(_gains_h->gains)));
    if (munmap(_gains_h, _buffer_size) == -1)
    {
        BOOST_LOG_TRIVIAL(error) << (std::string(
            "Failed to unmap shared memory with error: ")
            + std::strerror(errno));
    }
    if (close(_gain_buffer_fd) == -1)
    {
        BOOST_LOG_TRIVIAL(error) << (std::string(
            "Failed to close shared memory file descriptor with error: ")
            + std::strerror(errno));
    }
    if (sem_close(_gain_count_sem) == -1)
    {
        BOOST_LOG_TRIVIAL(error) << (std::string(
            "Failed to close counting semaphore with error: ")
            + std::strerror(errno));
    }
    if (sem_close(_gain_mutex_sem) == -1)
    {
        BOOST_LOG_TRIVIAL(error) << (std::string(
            "Failed to close mutex semaphore with error: ")
            + std::strerror(errno));
    }
}

bool GainManager::update_available()
{
    BOOST_LOG_TRIVIAL(debug) << "Checking for gain model update";
    int count;
    int retval = sem_getvalue(_gain_count_sem, &count);
    if (retval != 0)
    {
        throw std::runtime_error(std::string(
            "Unable to retrieve value of counting semaphore: ")
            + std::strerror(errno));
    }
    if (count == _last_sem_value)
    {
        BOOST_LOG_TRIVIAL(debug) << "No gain model update available";
        return false;
    }
    else
    {
        BOOST_LOG_TRIVIAL(debug) << "New gain model avialable";
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

GainManager::ComplexGainVectorType const& GainManager::gains()
{
    // This function should return the gains in GPU memory
    // First check if we need to update GPU memory
    if (update_available())
    {
        // Block on mutex semaphore
        // Technically this should *never* block as an increment to the
        // counting semaphore implies that the gain model has been updated
        // already. This is merely here for safety but may be removed in future.
        BOOST_LOG_TRIVIAL(debug) << "Acquiring shared memory mutex";
        int retval = sem_wait(_gain_mutex_sem);
        if (retval != 0)
        {
            throw std::runtime_error(std::string(
                "Unable to wait on mutex semaphore: ")
            + std::strerror(errno));
        }
        // Although this is intended as a blocking copy, it should only block on the host, not the GPU,
        // as such we use an async memcpy in a dedicated stream.
        void* dst = static_cast<void*>(thrust::raw_pointer_cast(_gains.data()));
        BOOST_LOG_TRIVIAL(debug) << "Copying gains to GPU";
        CUDA_ERROR_CHECK(cudaMemcpyAsync(dst, (void*) _gains_h, _buffer_size,
            cudaMemcpyHostToDevice, _copy_stream));
        CUDA_ERROR_CHECK(cudaStreamSynchronize(_copy_stream));

        BOOST_LOG_TRIVIAL(debug) << "Releasing shared memory mutex";
        retval = sem_post(_gain_mutex_sem);
        if (retval != 0)
        {
            throw std::runtime_error(std::string(
                "Unable to release mutex semaphore: ")
            + std::strerror(errno));
        }
    }
    return _gains;
}

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp
