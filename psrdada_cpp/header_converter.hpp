#ifndef PSRDADA_CPP_HEADER_CONVERTER_HPP
#define PSRDADA_CPP_HEADER_CONVERTER_HPP

#include "psrdada_cpp/raw_bytes.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp {


template <class HandlerType>
class HeaderConverter
{
public:
    typedef std::functional<void(RawBytes&, RawBytes&)> HeaderParserType;

public:
    HeaderConverter(HeaderParserType parser, HandlerType& handler);
    HeaderConverter(HeaderConverter const&) = delete;
    ~HeaderConverter();

    void init(RawBytes& block);

    bool operator()(RawBytes& block);

private:
    HeaderParserType _parser;
    HandlerType& _handler;
    char* _optr;
};


} //psrdada_cpp

#include "psrdada_cpp/detail/ascii_to_sigproc_header.cpp"
#endif //PSRDADA_CPP_HEADER_CONVERTER_HPP



