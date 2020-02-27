#include "psrdada_cpp/meerkat/fbfuse/test/PipelineTester.cuh"
#include "psrdada_cpp/meerkat/fbfuse/fbfuse_constants.hpp"
#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include "psrdada_cpp/meerkat/fbfuse/test/ChannelScalingManagerTester.cuh"
#include "psrdada_cpp/meerkat/fbfuse/BeamBandpassGenerator.hpp"
#include "psrdada_cpp/meerkat/fbfuse/DelayEngineSimulator.cuh"
#include "psrdada_cpp/meerkat/fbfuse/GainEngineSimulator.cuh"
#include "psrdada_cpp/Header.hpp"
#include "psrdada_cpp/dada_null_sink.hpp"
#include "psrdada_cpp/dada_read_client.hpp"
#include "psrdada_cpp/dada_input_stream.hpp"
#include "psrdada_cpp/dada_db.hpp"
#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include <random>
#include <cmath>
#include <complex>
#include <thread>
#include <chrono>
#include <vector>
#include <exception>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {

class StatisticsChecker
{
    public:
        StatisticsChecker( float, float, float );
        ~StatisticsChecker();

        void init(RawBytes& data);

        void operator()(RawBytes& data);

        float max_mean_diff();

        float max_std_diff();

        bool valid();

    private:
        float _exp_std;
        float _exp_mean;
        float _exp_dev;
        float _max_mean_diff;
        float _max_std_diff;
        bool _called;
};

StatisticsChecker::StatisticsChecker(float exp_mean, float exp_std, float exp_dev)
    : _exp_std(exp_std),
      _exp_mean(exp_mean),
      _exp_dev(exp_dev),
      _called(false)
{
}

StatisticsChecker::~StatisticsChecker()
{
}

void StatisticsChecker::init(RawBytes& data)
{

}

void  StatisticsChecker::operator()(RawBytes& data)
{
    // Populating the Channel Statistics Vector
    _called = true;
    std::size_t nchans = data.total_bytes()/sizeof(ChannelStatistics);
    ChannelStatistics* data_ptr = reinterpret_cast<ChannelStatistics*>(data.ptr());

    _max_mean_diff = std::numeric_limits<float>::min();
    _max_std_diff = std::numeric_limits<float>::min();

    for (std::size_t ii = 0; ii < nchans; ++ii)
    {
        auto dev_mean =  abs(data_ptr->mean - _exp_mean);
        if (dev_mean > _max_mean_diff)
            _max_mean_diff = dev_mean;

        auto dev_std = abs(std::sqrt(data_ptr->variance) - _exp_std)/_exp_std;
        if (dev_std > _max_std_diff)
            _max_std_diff = dev_std;

        ++data_ptr;
    }

}

float StatisticsChecker::max_mean_diff()
{
    return _max_mean_diff;
}

float StatisticsChecker::max_std_diff()
{
    return _max_std_diff;
}

bool StatisticsChecker::valid()
{
    return _called;
}

PipelineTester::PipelineTester()
    : ::testing::Test()
{
}

PipelineTester::~PipelineTester()
{
}

void PipelineTester::SetUp()
{
    _config.centre_frequency(1.4e9);
    _config.bandwidth(56.0e6);
    _config.channel_scaling_sem("test_chan_scaling_sem");
    _config.delay_buffer_shm("test_delay_buffer_shm");
    _config.delay_buffer_sem("test_delay_buffer_sem");
    _config.delay_buffer_mutex("test_delay_buffer_mutex");
    _config.gain_buffer_shm("test_gain_buffer_shm");
    _config.gain_buffer_sem("test_gain_buffer_sem");
    _config.gain_buffer_mutex("test_gain_buffer_mutex");
}

void PipelineTester::TearDown()
{
}

TEST_F(PipelineTester, simple_run_test)
{

    DelayEngineSimulator simulator(_config);
    GainEngineSimulator gsimulator(_config);
    ChannelScalingTrigger trigger(_config);
    trigger.request_statistics();

    int const ntimestamps_per_block = 64;
    int const taftp_block_size = (ntimestamps_per_block * _config.total_nantennas()
        * _config.nchans() * _config.nsamples_per_heap() * _config.npol());
    int const taftp_block_bytes = taftp_block_size * sizeof(char2);

    //Create output buffer for coherent beams
    int const cb_output_nsamps = _config.nsamples_per_heap() * ntimestamps_per_block / _config.cb_tscrunch();
    int const cb_output_nchans = _config.nchans() / _config.cb_fscrunch();
    int const cb_block_size = _config.cb_nbeams() * cb_output_nsamps * cb_output_nchans;
    DadaDB cb_buffer(8, cb_block_size, 4, 4096);
    cb_buffer.create();
    _config.cb_dada_key(cb_buffer.key());

    //Create output buffer for incoherent beams
    int const ib_output_nsamps = _config.nsamples_per_heap() * ntimestamps_per_block / _config.ib_tscrunch();
    int const ib_output_nchans = _config.nchans() / _config.ib_fscrunch();
    int const ib_block_size = _config.ib_nbeams() * ib_output_nsamps * ib_output_nchans;
    DadaDB ib_buffer(8, ib_block_size, 4, 4096);
    ib_buffer.create();
    _config.ib_dada_key(ib_buffer.key());

    //Setup write clients
    MultiLog log("PipelineTester");
    DadaWriteClient cb_write_client(_config.cb_dada_key(), log);
    DadaWriteClient ib_write_client(_config.ib_dada_key(), log);
    Pipeline pipeline(_config, cb_write_client, ib_write_client, taftp_block_bytes);

    //Set up null sinks on all buffers
    NullSink null_sink;

    DadaInputStream<NullSink> cb_consumer(_config.cb_dada_key(), log, null_sink);
    DadaInputStream<NullSink> ib_consumer(_config.ib_dada_key(), log, null_sink);

    std::thread cb_consumer_thread( [&](){
        try {
            cb_consumer.start();
        } catch (std::exception& e) {
            BOOST_LOG_TRIVIAL(error) << e.what();
        }
	});
    std::thread ib_consumer_thread( [&](){
        try {
            ib_consumer.start();
        } catch (std::exception& e) {
            BOOST_LOG_TRIVIAL(error) << e.what();
        }
	});

    //Create and input header buffer
    std::vector<char> input_header_buffer(4096, 0);
    RawBytes input_header_rb(input_header_buffer.data(), 4096, 4096);
    Header header(input_header_rb);
    header.set<long double>("SAMPLE_CLOCK", 856000000.0);
    header.set<long double>("SYNC_TIME", 0.0);
    header.set<std::size_t>("SAMPLE_CLOCK_START", 0);

    //Create and input data buffer
    char* input_data_buffer;
    CUDA_ERROR_CHECK(cudaMallocHost((void**)&input_data_buffer, taftp_block_bytes));
    RawBytes input_data_rb(input_data_buffer, taftp_block_bytes, taftp_block_bytes);

    float input_level = 32.0f;
    _config.output_level(32.0f);
    std::default_random_engine generator;
    std::normal_distribution<float> normal_dist(0.0, input_level);
    for (std::size_t idx = 0; idx < taftp_block_bytes; ++idx)
    {
        input_data_buffer[idx] = static_cast<int8_t>(std::lround(normal_dist(generator)));
    }
    //Run the init
    pipeline.init(input_header_rb);
    //Loop over N data blocks and push them through the system
    for (int ii = 0; ii < 10; ++ii)
    {
        pipeline(input_data_rb);
    }
    cb_consumer.stop();
    ib_consumer.stop();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    pipeline(input_data_rb);
    cb_consumer_thread.join();
    ib_consumer_thread.join();
    CUDA_ERROR_CHECK(cudaFreeHost((void*)input_data_buffer));
}

TEST_F(PipelineTester, check_stats_test)
{

    DelayEngineSimulator simulator(_config);
    GainEngineSimulator gsimulator(_config);
    ChannelScalingTrigger trigger(_config);
    trigger.request_statistics();

    int const ntimestamps_per_block = 64;
    int const taftp_block_size = (ntimestamps_per_block * _config.total_nantennas()
            * _config.nchans() * _config.nsamples_per_heap() * _config.npol());
    int const taftp_block_bytes = taftp_block_size * sizeof(char2);

    //Create output buffer for coherent beams
    int const cb_output_nsamps = _config.nsamples_per_heap() * ntimestamps_per_block / _config.cb_tscrunch();
    int const cb_output_nchans = _config.nchans() / _config.cb_fscrunch();
    int const cb_block_size = _config.cb_nbeams() * cb_output_nsamps * cb_output_nchans;
    DadaDB cb_buffer(8, cb_block_size, 4, 4096);
    cb_buffer.create();
    _config.cb_dada_key(cb_buffer.key());

    //Create output buffer for incoherent beams
    int const ib_output_nsamps = _config.nsamples_per_heap() * ntimestamps_per_block / _config.ib_tscrunch();
    int const ib_output_nchans = _config.nchans() / _config.ib_fscrunch();
    int const ib_block_size = _config.ib_nbeams() * ib_output_nsamps * ib_output_nchans;
    DadaDB ib_buffer(8, ib_block_size, 4, 4096);
    ib_buffer.create();
    _config.ib_dada_key(ib_buffer.key());

    //Setup write clients
    MultiLog log("PipelineTester");
    DadaWriteClient cb_write_client(_config.cb_dada_key(), log);
    DadaWriteClient ib_write_client(_config.ib_dada_key(), log);
    Pipeline pipeline(_config, cb_write_client, ib_write_client, taftp_block_bytes);

    //Set up null sinks on all buffers
    StatisticsChecker checker(0.0f, 32.0f, 0.05);
    BeamBandpassGenerator<decltype(checker)> cb_bpmon(_config.cb_nbeams(), 64, 1, 8192, 1, checker);
    BeamBandpassGenerator<decltype(checker)> ib_bpmon(1, 64, 1, 8192, 1, checker);

    DadaInputStream<decltype(cb_bpmon)> cb_consumer(_config.cb_dada_key(), log, cb_bpmon);
    DadaInputStream<decltype(ib_bpmon)> ib_consumer(_config.ib_dada_key(), log, ib_bpmon);

    std::thread cb_consumer_thread( [&](){
        try {
            cb_consumer.start();
        } catch (std::exception& e) {
            BOOST_LOG_TRIVIAL(error) << e.what();
        }
	});
    std::thread ib_consumer_thread( [&](){
        try {
            ib_consumer.start();
        } catch (std::exception& e) {
            BOOST_LOG_TRIVIAL(error) << e.what();
        }
	});

    //Create and input header buffer
    std::vector<char> input_header_buffer(4096, 0);
    RawBytes input_header_rb(input_header_buffer.data(), 4096, 4096);
    Header header(input_header_rb);
    header.set<long double>("SAMPLE_CLOCK", 856000000.0);
    header.set<long double>("SYNC_TIME", 0.0);
    header.set<std::size_t>("SAMPLE_CLOCK_START", 0);

    //Create and input data buffer
    char* input_data_buffer;
    CUDA_ERROR_CHECK(cudaMallocHost((void**)&input_data_buffer, taftp_block_bytes));
    RawBytes input_data_rb(input_data_buffer, taftp_block_bytes, taftp_block_bytes);
    float input_level = 32.0f;
    _config.output_level(32.0f);
    std::default_random_engine generator;
    std::normal_distribution<float> normal_dist(0.0, input_level);

    for (std::size_t ii = 0; ii < taftp_block_bytes; ++ii)
    {
        std::size_t chan_idx = ((ii/1024/_config.nchans()) % _config.nchans());
        float factor = chan_idx * (0.8/_config.nchans()) +  0.6;
        float val = std::lround(factor*normal_dist(generator));
        input_data_buffer[ii] = static_cast<int8_t>(std::fmaxf(-127.0f,std::fminf(127.0f,val)));
    }

    //Run the init
    pipeline.init(input_header_rb);
    //Loop over N data blocks and push them through the system
    for (int ii = 0; ii < 10; ++ii)
    {
        pipeline(input_data_rb);
        if (checker.valid())
        {
            EXPECT_GT(5.0, checker.max_mean_diff());
            EXPECT_GT(0.12, checker.max_std_diff());
        }
    }
    cb_consumer.stop();
    ib_consumer.stop();
    std::this_thread::sleep_for(std::chrono::seconds(10));
    pipeline(input_data_rb);
    pipeline(input_data_rb);
    cb_consumer_thread.join();
    ib_consumer_thread.join();
    CUDA_ERROR_CHECK(cudaFreeHost((void*)input_data_buffer));
}
} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

