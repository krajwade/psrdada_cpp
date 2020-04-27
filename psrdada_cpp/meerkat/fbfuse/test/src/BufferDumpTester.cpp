
#include "psrdada_cpp/meerkat/fbfuse/test/BufferDumpTester.hpp"
#include "psrdada_cpp/dada_db.hpp"
#include "psrdada_cpp/dada_null_sink.hpp"
#include "psrdada_cpp/dada_output_stream.hpp"
#include "psrdada_cpp/Header.hpp"
#include <iostream>
#include <boost/asio.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/optional.hpp>
#include <sstream>
#include <cstdlib>
#include <thread>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace test {


void send(boost::asio::local::stream_protocol::socket & socket, const std::string& message)
{
    try
    {
    	const std::string msg = message;
    	boost::asio::write( socket, boost::asio::buffer(message) );
    }
    catch (std::exception& e)
    {
	BOOST_LOG_TRIVIAL(error) << "Error in send";
	BOOST_LOG_TRIVIAL(error) << e.what();
    }
}

void receive(boost::asio::local::stream_protocol::socket& socket)
{
    try
    {
        char* receiver = new char[4096];
    	boost::asio::read( socket, boost::asio::buffer(receiver, 4096) );
    }
    catch (std::exception& e)
    {
	BOOST_LOG_TRIVIAL(error) << "Error in receive";
	BOOST_LOG_TRIVIAL(error) << e.what();
    }
}
void get_json(std::stringstream& ss, long double starttime, long double endtime, float dm, float ref_freq, std::string trig_id )
{
    boost::property_tree::ptree pt;
    pt.put<long double>("utc_start", starttime);
    pt.put<long double>("utc_end", endtime);
    pt.put<float>("dm", dm);
    pt.put<float>("reference_freq", ref_freq);
    pt.put<std::string>("trigger_id", trig_id);
    boost::property_tree::json_parser::write_json(ss, pt);
    return;
}

void send_json(long double starttime, long double endtime, float dm, float ref_freq, std::string trig_id, boost::asio::local::stream_protocol::socket& socket)
{
    try
    {
        boost::asio::local::stream_protocol::endpoint ep("/tmp/buffer_dump_test.sock");
        socket.connect(ep);
        //Send the message
        std::stringstream event_string;
        get_json(event_string, starttime, endtime, dm, ref_freq, trig_id);
        BOOST_LOG_TRIVIAL(debug) << "Sending Trigger...";
        send(socket, event_string.str());
    }
    catch(std::exception& e)
    {
        BOOST_LOG_TRIVIAL(error) << "Error in send_json";
        BOOST_LOG_TRIVIAL(error) << e.what();
        exit(1);
    }

    return;
}


BufferDumpTester::BufferDumpTester()
    : ::testing::Test()
{

}

BufferDumpTester::~BufferDumpTester()
{
}

void BufferDumpTester::SetUp()
{
}

void BufferDumpTester::TearDown()
{
}

TEST_F(BufferDumpTester, do_nothing)
{
    boost::asio::io_service io_service;
    boost::asio::local::stream_protocol::socket socket(io_service);

    std::size_t nchans = 64;
    std::size_t total_nchans = 4096;
    std::size_t nantennas = 64;
    std::size_t ngroups = 8;
    std::size_t nblocks = 64;
    std::size_t block_size = nchans * nantennas * ngroups * 256 * sizeof(unsigned);

    float cfreq = 856e6;
    float bw = 856e6 / (total_nchans / nchans);
    float max_fill_level = 0.8;

    DadaDB buffer(nblocks, block_size, 4, 4096);
    buffer.create();
    MultiLog log("log");
    DadaOutputStream ostream(buffer.key(), log);

    std::vector<char> input_header_buffer(4096, 0);
    RawBytes input_header_rb(input_header_buffer.data(), 4096, 4096);
    Header header(input_header_rb);
    header.set<long double>("SAMPLE_CLOCK", 856000000.0);
    header.set<long double>("SYNC_TIME", 0.0);
    header.set<std::size_t>("SAMPLE_CLOCK_START", 0);
    ostream.init(input_header_rb);
    std::vector<char> input_data_buffer(block_size, 0);
    RawBytes input_data_rb(&input_data_buffer[0], block_size, block_size);
    for (uint32_t ii=0; ii < nblocks-1; ++ii)
    {
        ostream(input_data_rb);
    }

    //DadaReadClient reader(buffer.key(), log);
    BufferDump dumper(buffer.key(), log, "/tmp/buffer_dump_test.sock",
                                      max_fill_level, nantennas, nchans,
                                      total_nchans, cfreq, bw );

    std::thread dumper_thread([&](){
        dumper.start();
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));

    dumper.stop();

    dumper_thread.join();
}


TEST_F(BufferDumpTester, read_event)
{
   /* Send in an artificial trigger and check if the paramters are being read properly*/
    boost::asio::io_service io_service;
    boost::asio::local::stream_protocol::socket socket(io_service);

    std::size_t nchans = 64;
    std::size_t total_nchans = 4096;
    std::size_t nantennas = 64;
    std::size_t ngroups = 8;
    std::size_t nblocks = 64;
    std::size_t block_size = nchans * nantennas * ngroups * 256 * sizeof(unsigned);

    float cfreq = 862.68e6;
    float bw = 856e6 / (total_nchans / nchans);
    float max_fill_level = 0.8;

    DadaDB buffer(nblocks, block_size, 4, 4096);
    buffer.create();
    MultiLog log("log");
    DadaOutputStream ostream(buffer.key(), log);

    std::vector<char> input_header_buffer(4096, 0);
    RawBytes input_header_rb(input_header_buffer.data(), 4096, 4096);
    Header header(input_header_rb);
    header.set<long double>("SAMPLE_CLOCK", 856000000.0);
    header.set<long double>("SYNC_TIME", 0.0);
    header.set<std::size_t>("SAMPLE_CLOCK_START", 0);
    ostream.init(input_header_rb);
    std::vector<char> input_data_buffer(block_size, 0);
    RawBytes input_data_rb(&input_data_buffer[0], block_size, block_size);
    for (uint32_t ii=0; ii < nblocks-1; ++ii)
    {
        ostream(input_data_rb);
    }

    //DadaReadClient reader(buffer.key(), log);
    BufferDump dumper(buffer.key(), log, "/tmp/buffer_dump_test.sock",
                                      max_fill_level, nantennas, nchans,
                                      total_nchans, cfreq, bw );

    std::thread dumper_thread([&](){
        dumper.start();
    });

    std::this_thread::sleep_for(std::chrono::seconds(10));
    // Generate a trigger //
    ASSERT_NO_THROW(send_json(0.5, 0.7, 100.0, 869.375e6, "FRB1", socket));

    socket.close();

    std::this_thread::sleep_for(std::chrono::seconds(10));

    // Generate second trigger to make sure that it works the second time
    ASSERT_NO_THROW(send_json(1.0, 1.2, 100.0, 869.375e6, "FRB2", socket));

    std::this_thread::sleep_for(std::chrono::seconds(10));

    socket.close();

    dumper.stop();

    io_service.stop();

    dumper_thread.join();
}



} //namespace test
} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

