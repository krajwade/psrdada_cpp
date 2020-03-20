#include "psrdada_cpp/meerkat/tuse/transpose_to_dada.hpp"
#include "psrdada_cpp/cli_utils.hpp"
#include <ctime>
#include <mutex>
#include <iostream>


namespace psrdada_cpp {
namespace meerkat {
namespace tuse {

namespace transpose{

    /*
     * @brief This is the actual block that performs the
     * transpose. The format is based on the heap format
     * of SPEAD2 packets. This can change in time
     */
    std::mutex MyMutex;
    void do_transpose(RawBytes& transposed_data, RawBytes& input_data,std::uint32_t nchans, std::uint32_t nsamples, std::uint32_t nfreq, std::uint32_t beamnum, std::uint32_t nbeams, std::uint32_t ngroups, std::size_t tscrunch, std::size_t fscrunch)
    {
        // make copies of arrays to be transposed
        if (input_data.total_bytes() % (nfreq * nchans * nsamples * nbeams) != 0)
        {
            auto sug_size = nfreq * nchans * nsamples * nbeams * ngroups;
            throw std::runtime_error(std::string("Incorrect size of the DADA block. Should be a multiple of heap group size. Suggested size is:") + std::to_string(sug_size) + std::string("bytes")); 
        }
        const size_t tocopy = ngroups * nsamples * nfreq * nchans;
        std::vector<char> tmpindata(tocopy / ngroups);
        std::vector<char>tmpoutdata(tocopy);
        size_t skipgroup = nchans * nsamples * nfreq * nbeams;
        size_t skipbeam = beamnum * nchans * nsamples * nfreq;
        size_t skipband = nchans * nsamples;
        size_t skipallchans = nchans * nfreq;
        // actual transpose
        for (unsigned int igroup = 0; igroup < ngroups; ++igroup)
        {
            std::copy(input_data.ptr() + skipbeam + igroup * skipgroup, input_data.ptr() + skipbeam + igroup * skipgroup + tocopy / ngroups, tmpindata.begin());

            for (unsigned int isamp = 0; isamp < nsamples; ++isamp)
            {
                for (unsigned int iband = 0; iband < nfreq; ++iband)
                {
                    std::copy(tmpindata.begin() + iband * skipband + isamp * nchans, tmpindata.begin() + iband * skipband + isamp * nchans + nchans, tmpoutdata.begin() + iband * nchans + isamp * skipallchans + igroup * tocopy/ngroups);
                } // BAND LOOP

                /* Reverse the channel array */
                std::reverse(tmpoutdata.begin() + isamp * skipallchans + igroup *tocopy/ngroups, tmpoutdata.begin() + isamp * skipallchans + igroup *tocopy/ngroups + nfreq*nchans);

            } // SAMPLES LOOP
        } // GROUP LOOP

        std::size_t ii = 0;
        auto add = [&](std::uint8_t x, std::uint8_t y)
            {
                std::size_t ind = 0;
                for (std::uint32_t jj=0; jj < tscrunch; ++jj)
                {
                    std::uint8_t temp = tmpoutdata[ind*nchans + ii];
                    y += temp;
                    ++ind;
                }
                return x + y;
            };

        // Convert to unsigned (add 128.0)
        std::transform(tmpoutdata.begin(), tmpoutdata.end(), tmpoutdata.begin(), std::bind2nd(std::plus<char>(),128));

        // downsampling the data
        if (fscrunch != 1 || tscrunch !=1)
        {
            for (ii = 0; ii < (nchans*nfreq/fscrunch) * (ngroups*nsamples/tscrunch); ++ii)
            {
                tmpoutdata[ii] = (std::accumulate(tmpoutdata.begin() + (ii*fscrunch), tmpoutdata.begin() + ((ii+1)*fscrunch),0) +
                        std::accumulate(tmpoutdata.begin() + ii, tmpoutdata.begin() + ii + 1,0,add))/(tscrunch + fscrunch);
            }
            tmpoutdata.resize(nchans * nfreq/fscrunch * (ngroups*nsamples/tscrunch));
        }

        //copy to output
        std::copy(tmpoutdata.begin(),tmpoutdata.end(), transposed_data.ptr());
    }
} //transpose
} //tuse
} //meerkat
} //psrdada_cpp
