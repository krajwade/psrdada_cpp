#include "psrdada_cpp/dada_io_loop.hpp"

namespace psrdada_cpp
{

    DadaIoLoop::DadaIoLoop(key_t key, MultiLog& log)
    : _key(key)
    , _log(log)
    , _stop(false)
    , _running(false)
    {
    }

    DadaIoLoop::~DadaIoLoop()
    {
    }

    void DadaIoLoop::stop()
    {
        _stop = true;
    }

} //namespace psrdada_cpp