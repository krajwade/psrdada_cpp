#include "psrdada_cpp/meerkat/fbfuse/GainEngineSimulator.cuh"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>
#include <sys/mman.h>
#include <sstream>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {

GainEngineSimulator::GainEngineSimulator(PipelineConfig const& config)
    : _gain_buffer_shm(config.gain_buffer_shm())
    , _gain_buffer_sem(config.gain_buffer_sem())
    , _gain_buffer_mutex(config.gain_buffer_mutex())
{

    std::size_t expected_bytes = config.nantennas() * config.nchans() * config.npol() * sizeof(GainManager::ComplexGainType);

    _shm_fd = shm_open(_gain_buffer_shm.c_str(), O_CREAT | O_RDWR, 0666);
    if (_shm_fd == -1)
    {
        std::stringstream msg;
        msg << "Failed to open shared memory named "
        << _gain_buffer_shm << " with error: "
        << std::strerror(errno);
        throw std::runtime_error(msg.str());
    }
    if (ftruncate(_shm_fd, expected_bytes) == -1)
    {
        std::stringstream msg;
        msg << "Failed to ftruncate shared memory named "
        << _gain_buffer_shm << " with error: "
        << std::strerror(errno);
        throw std::runtime_error(msg.str());
    }
    _shm_ptr = mmap(0, expected_bytes, PROT_WRITE, MAP_SHARED, _shm_fd, 0);
    if (_shm_ptr == NULL)
    {
        std::stringstream msg;
        msg << "Failed to mmap shared memory named "
        << _gain_buffer_shm << " with error: "
        << std::strerror(errno);
        throw std::runtime_error(msg.str());
    }
    _gain_model = static_cast<GainManager::ComplexGainType*>(_shm_ptr);
    _sem_id = sem_open(_gain_buffer_sem.c_str(), O_CREAT, 0666, 0);
    if (_sem_id == SEM_FAILED)
    {
        std::stringstream msg;
        msg << "Failed to open gain buffer semaphore "
        << _gain_buffer_sem << " with error: "
        << std::strerror(errno);
        throw std::runtime_error(msg.str());
    }
    _mutex_id = sem_open(_gain_buffer_mutex.c_str(), O_CREAT, 0666, 0);
    if (_mutex_id == SEM_FAILED)
    {
        std::stringstream msg;
        msg << "Failed to open gain buffer mutex "
        << _gain_buffer_mutex << " with error: "
        << std::strerror(errno);
        throw std::runtime_error(msg.str());
    }
    // Here we post once so that the mutex has a value of 1
    // and can so be safely acquired
    sem_post(_mutex_id);
}

GainEngineSimulator::~GainEngineSimulator()
{
    if (munmap(_shm_ptr, sizeof(GainModel)) == -1)
    {
        std::stringstream msg;
        msg << "Failed to unmap shared memory "
        << _gain_buffer_shm << " with error: "
        << std::strerror(errno);
        //throw std::runtime_error(msg.str());
        BOOST_LOG_TRIVIAL(error) << msg.str();
    }

    if (close(_shm_fd) == -1)
    {
        std::stringstream msg;
        msg << "Failed to close shared memory file descriptor "
        << _shm_fd << " with error: "
        << std::strerror(errno);
        //throw std::runtime_error(msg.str());
	BOOST_LOG_TRIVIAL(error) << msg.str();
    }

    if (shm_unlink(_gain_buffer_shm.c_str()) == -1)
    {
        std::stringstream msg;
        msg << "Failed to unlink shared memory "
        << _gain_buffer_shm << " with error: "
        << std::strerror(errno);
        //throw std::runtime_error(msg.str());
        BOOST_LOG_TRIVIAL(error) << msg.str();
    }

    if (sem_close(_sem_id) == -1)
    {
        std::stringstream msg;
        msg << "Failed to close semaphore "
        << _gain_buffer_sem << " with error: "
        << std::strerror(errno);
        //throw std::runtime_error(msg.str());
        BOOST_LOG_TRIVIAL(error) << msg.str();
    }

    if (sem_close(_mutex_id) == -1)
    {
        std::stringstream msg;
        msg << "Failed to close mutex "
        << _gain_buffer_mutex << " with error: "
        << std::strerror(errno);
        //throw std::runtime_error(msg.str());
        BOOST_LOG_TRIVIAL(error) << msg.str();
    }
}

void GainEngineSimulator::update_gains()
{
    sem_post(_sem_id);
}

GainManager::ComplexGainType* GainEngineSimulator::gains()
{
    return _gain_model;
}


} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp
