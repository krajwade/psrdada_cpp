#include "psrdada_cpp/simple_file_writer.hpp"

namespace psrdada_cpp {

    SimpleFileWriter::SimpleFileWriter(std::string filename)
    {
        _outfile.open(filename.c_str(),std::ifstream::out | std::ifstream::binary);
        if (_outfile.is_open())
        {
            BOOST_LOG_TRIVIAL(debug) << "Opened file " << filename;
        }
        else
        {
            std::stringstream stream;
            stream << "Could not open file " << filename;
            throw std::runtime_error(stream.str().c_str());
        }
    }

    SimpleFileWriter::~SimpleFileWriter()
    {
        _outfile.close();
    }

    void SimpleFileWriter::init(RawBytes& block)
    {
        _outfile.write(block.ptr(), 4096);
    }

    void SimpleFileWriter::init(RawBytes& block, std::size_t size)
    {
        _outfile.write(block.ptr(), size);
    }

    bool SimpleFileWriter::operator()(RawBytes& block)
    {
        _outfile.write(block.ptr(), block.used_bytes());
        _outfile.flush();
        //This is specifying the number of bytes read.
        block.used_bytes(block.total_bytes());
        return false;
    }

} //psrdada_cpp
