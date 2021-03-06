#include "psrdada_cpp/effelsberg/edd/Channeliser.cuh"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "thrust/functional.h"
#include "thrust/transform.h"
#include <cuda.h>

namespace psrdada_cpp {
namespace effelsberg {
namespace edd {

Channeliser::Channeliser(
    std::size_t buffer_bytes,
    std::size_t fft_length,
    std::size_t nbits,
    float input_level,
    float output_level,
    DadaWriteClient& client)
    : _buffer_bytes(buffer_bytes)
    , _fft_length(fft_length)
    , _nbits(nbits)
    , _client(client)
    , _fft_plan(0)
    , _call_count(0)
{

    assert(((_nbits == 12) || (_nbits == 8)));
    BOOST_LOG_TRIVIAL(debug)
    << "Creating new Channeliser instance with parameters: \n"
    << "fft_length = " << _fft_length << "\n"
    << "nbits = " << _nbits;
    std::size_t nsamps_per_buffer = buffer_bytes * 8 / nbits;
    assert(nsamps_per_buffer % _fft_length == 0 /*Number of samples is not multiple of FFT size*/);
    std::size_t n64bit_words = buffer_bytes / sizeof(uint64_t);
    _nchans = _fft_length / 2 + 1;
    int batch = nsamps_per_buffer/_fft_length;
    std::size_t packed_channelised_voltage_bytes = _nchans * batch * sizeof(PackedChannelisedVoltageType);
    BOOST_LOG_TRIVIAL(debug) << "Output buffer bytes: " << packed_channelised_voltage_bytes;
    assert(_client.data_buffer_size() == packed_channelised_voltage_bytes /* Incorrect output DADA buffer size */);
    BOOST_LOG_TRIVIAL(debug) << "Calculating scales and offsets";
    float scale = std::sqrt(_nchans) * input_level;
    BOOST_LOG_TRIVIAL(debug) << "Correction factors for 8-bit conversion:  scaling = " << scale;
    BOOST_LOG_TRIVIAL(debug) << "Generating FFT plan";
    int n[] = {static_cast<int>(_fft_length)};
    //Not we put this into transposed output order, so the inner dimension will be time.
    CUFFT_ERROR_CHECK(cufftPlanMany(&_fft_plan, 1, n, NULL, 1, _fft_length,
        NULL, 1, _nchans, CUFFT_R2C, batch));
    cufftSetStream(_fft_plan, _proc_stream);
    BOOST_LOG_TRIVIAL(debug) << "Allocating memory";
    _raw_voltage_db.resize(n64bit_words);
    BOOST_LOG_TRIVIAL(debug) << "Input voltages size (in 64-bit words): " << _raw_voltage_db.size();
    _unpacked_voltage.resize(nsamps_per_buffer);
    BOOST_LOG_TRIVIAL(debug) << "Unpacked voltages size (in samples): " << _unpacked_voltage.size();
    _channelised_voltage.resize(_nchans * batch);
    BOOST_LOG_TRIVIAL(debug) << "Channelised voltages size: " << _channelised_voltage.size();
    _packed_channelised_voltage.resize(_channelised_voltage.size());
    BOOST_LOG_TRIVIAL(debug) << "Packed channelised voltages size: " << _packed_channelised_voltage.size();
    CUDA_ERROR_CHECK(cudaStreamCreate(&_h2d_stream));
    CUDA_ERROR_CHECK(cudaStreamCreate(&_proc_stream));
    CUDA_ERROR_CHECK(cudaStreamCreate(&_d2h_stream));
    CUFFT_ERROR_CHECK(cufftSetStream(_fft_plan, _proc_stream));
    _unpacker.reset(new Unpacker(_proc_stream));
    _transposer.reset(new ScaledTransposeTFtoTFT(_nchans, 8192, scale, 0.0, _proc_stream));
}

Channeliser::~Channeliser()
{
    BOOST_LOG_TRIVIAL(debug) << "Destroying Channeliser";
    if (!_fft_plan)
        cufftDestroy(_fft_plan);
    cudaStreamDestroy(_h2d_stream);
    cudaStreamDestroy(_proc_stream);
    cudaStreamDestroy(_d2h_stream);
}

void Channeliser::init(RawBytes& block)
{
    BOOST_LOG_TRIVIAL(debug) << "Channeliser init called";
    auto& header_block = _client.header_stream().next();
    /* Populate new header */
    std::memcpy(header_block.ptr(), block.ptr(), block.total_bytes());
    header_block.used_bytes(header_block.total_bytes());
    _client.header_stream().release();
}

void Channeliser::process(
    thrust::device_vector<RawVoltageType> const& digitiser_raw,
    thrust::device_vector<PackedChannelisedVoltageType>& packed_channelised)
{
    BOOST_LOG_TRIVIAL(debug) << "Unpacking raw voltages";
    switch (_nbits)
    {
        case 8:  _unpacker->unpack<8>(digitiser_raw, _unpacked_voltage); break;
        case 12: _unpacker->unpack<12>(digitiser_raw, _unpacked_voltage); break;
        default: throw std::runtime_error("Unsupported number of bits");
    }
    BOOST_LOG_TRIVIAL(debug) << "Performing FFT";
    UnpackedVoltageType* _unpacked_voltage_ptr = thrust::raw_pointer_cast(_unpacked_voltage.data());
    ChannelisedVoltageType* _channelised_voltage_ptr = thrust::raw_pointer_cast(_channelised_voltage.data());
    CUFFT_ERROR_CHECK(cufftExecR2C(_fft_plan,
        (cufftReal*) _unpacked_voltage_ptr,
        (cufftComplex*) _channelised_voltage_ptr));
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_proc_stream));
    _transposer->transpose(_channelised_voltage, packed_channelised);
}

bool Channeliser::operator()(RawBytes& block)
{
    ++_call_count;
    BOOST_LOG_TRIVIAL(debug) << "Channeliser operator() called (count = " << _call_count << ")";
    assert(block.used_bytes() == _buffer_bytes /* Unexpected buffer size */);

    CUDA_ERROR_CHECK(cudaStreamSynchronize(_h2d_stream));
    _raw_voltage_db.swap();

    CUDA_ERROR_CHECK(cudaMemcpyAsync(static_cast<void*>(_raw_voltage_db.a_ptr()),
        static_cast<void*>(block.ptr()), block.used_bytes(),
        cudaMemcpyHostToDevice, _h2d_stream));

    if (_call_count == 1)
    {
        return false;
    }

    // Synchronize all streams
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_proc_stream));
    _packed_channelised_voltage.swap();
    process(_raw_voltage_db.b(), _packed_channelised_voltage.a());

    if (_call_count == 2)
    {
        return false;
    }
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_d2h_stream));
    
    if (_call_count > 3)
    {
        _client.data_stream().release();
    }
    auto& data_block = _client.data_stream().next();
    CUDA_ERROR_CHECK(cudaMemcpyAsync(
        static_cast<void*>(data_block.ptr()),
        static_cast<void*>(_packed_channelised_voltage.b_ptr()),
        _packed_channelised_voltage.size() * sizeof(PackedChannelisedVoltageType),
        cudaMemcpyDeviceToHost,
        _d2h_stream));
    data_block.used_bytes(data_block.total_bytes());
    return false;
}

} //edd
} //effelsberg
} //psrdada_cpp


