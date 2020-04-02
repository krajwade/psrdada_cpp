#include "psrdada_cpp/meerkat/tuse/transpose_to_dada.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include <ctime>
#include <thread>
namespace psrdada_cpp {
namespace meerkat {
namespace tuse {

	template <class HandlerType>
	TransposeToDada<HandlerType>::TransposeToDada(std::size_t numbeams, std::vector<std::shared_ptr<HandlerType>> handler)
	: _numbeams(numbeams)
	, _handler(std::move(handler))
	, _nchans(128)
	, _nsamples(64)
	, _nfreq(32)
	, _ngroups(10)
    , _tscrunch(1)
    , _fscrunch(1)
	{
	}

	template <class HandlerType>
	TransposeToDada<HandlerType>::~TransposeToDada()
	{
	}

	template <class HandlerType>
	void TransposeToDada<HandlerType>::init(RawBytes& block)
	{
		std::uint32_t ii;
		for (ii = 0; ii < _numbeams; ii++ )
		{
			(*_handler[ii]).init(block);
		}
	}

	template <class HandlerType>
	bool TransposeToDada<HandlerType>::operator()(RawBytes& block)
	{
		std::uint32_t ii;
		std::vector<std::thread> threads;
		auto transpose_size = (_nchans *_nfreq/ _fscrunch) * (_nsamples * _ngroups/_tscrunch);
        _transpose_buffers.resize(_numbeams);
        for (auto& buffer: _transpose_buffers)
        {
            buffer.resize(transpose_size);
        }

        bool thread_error = false;

        for(ii=0; ii< _numbeams; ii++)
        {
            threads.emplace_back(std::thread([&, ii]()
            {
                try
                {
                    char* o_data = _transpose_buffers[ii].data();
                    RawBytes transpose(o_data,std::size_t(transpose_size),std::size_t(0));
                    transpose::do_transpose(transpose, block, _nchans, _nsamples, _nfreq, ii, _numbeams, _ngroups, _tscrunch, _fscrunch);
                    transpose.used_bytes(transpose.total_bytes());
                    (*_handler[ii])(transpose);
                }
                catch(...)
                {
                    BOOST_LOG_TRIVIAL(error) << "Error in transpose";
                    thread_error = true;
                }
            }
            ));
        }

        for (ii=0; ii< _numbeams; ii++)
        {
            threads[ii].join();
        }

		if (thread_error)
		{
			throw std::runtime_error("Unknonw error in transpose thread");
		}
		return false;
	}

	template <class HandlerType>
	void TransposeToDada<HandlerType>::set_nchans(const int nchans)
	{
		_nchans = nchans;
	}

	template <class HandlerType>
	void TransposeToDada<HandlerType>::set_nbeams(const int nbeams)
	{
		_numbeams = nbeams;
	}

	template <class HandlerType>
	void TransposeToDada<HandlerType>::set_ngroups(const int ngroups)
	{
		_ngroups = ngroups;
	}

	template <class HandlerType>
	void TransposeToDada<HandlerType>::set_nsamples(const int nsamples)
	{
		_nsamples = nsamples;
	}

	template <class HandlerType>
	void TransposeToDada<HandlerType>::set_nfreq(const int nfreq)
	{
		_nfreq = nfreq;
	}

	template <class HandlerType>
	void TransposeToDada<HandlerType>::set_tscrunch(const std::size_t tscrunch)
	{
		_tscrunch = tscrunch;
	}

	template <class HandlerType>
	void TransposeToDada<HandlerType>::set_fscrunch(const std::size_t fscrunch)
	{
		_fscrunch = fscrunch;
	}

	template <class HandlerType>
	std::uint32_t TransposeToDada<HandlerType>::nchans()
	{
		return _nchans;
	}

	template <class HandlerType>
	std::uint32_t TransposeToDada<HandlerType>::nsamples()
	{
		return _nsamples;
	}

	template <class HandlerType>
	std::uint32_t TransposeToDada<HandlerType>::nfreq()
	{
		return _nfreq;
	}

	template <class HandlerType>
	std::uint32_t TransposeToDada<HandlerType>::nbeams()
	{
		return _numbeams;
	}

	template <class HandlerType>
	std::uint32_t TransposeToDada<HandlerType>::ngroups()
	{
		return _ngroups;
	}

	template <class HandlerType>
	std::size_t TransposeToDada<HandlerType>::tscrunch()
	{
		return _tscrunch;
	}

	template <class HandlerType>
	std::size_t TransposeToDada<HandlerType>::fscrunch()
	{
		return _fscrunch;
	}
} //tuse
} //meerkat
} //psrdada_cpp
