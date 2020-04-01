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
        auto add_t = [&](std::uint8_t x, std::uint8_t y)
            {
                float temp=0.0;
                for (std::uint32_t jj=1; jj < tscrunch; ++jj)
                {
                    temp += ((float)y + (float)tmpoutdata[jj*nchans + ii])/((float)(tscrunch*fscrunch));
                }
                return x + (uint8_t)temp;
            };

        auto add_f = [&](std::uint8_t x, std::uint8_t y)
            {
                return x + (uint8_t) ((float)y/(float)(tscrunch*fscrunch));
            };

        // Convert to unsigned (add 128.0)
        std::transform(tmpoutdata.begin(), tmpoutdata.end(), tmpoutdata.begin(), std::bind2nd(std::plus<char>(),128));

        std::size_t factor = tscrunch*fscrunch;

        // downsampling the data
        std:size_t freqindex = 0, timeindex=0, stepindex=0, offset=0;
        std::uint8_t sum = 0;
        //  Two methods to do this: 1) Nested for loop and 2) separate for loops
        //  Method 1
        if (fscrunch != 1 || tscrunch !=1)
        {
            for (ii = 0; ii < tocopy/factor; ++ii)
            {
                sum  = 0;

                if (ii*fscrunch < skipallchans*stepindex)
                {
                    for (std::size_t jj = 0; jj < tscrunch; ++jj)
                    {
                        freqindex = 0;
                        while (freqindex < fscrunch)
                        {
                            sum += (uint8_t)( (float)tmpoutdata[ (timeindex*fscrunch + offset ) + jj*skipallchans + freqindex]/(float)(factor));
                            ++freqindex;
                            //(std::accumulate(tmpoutdata.begin() + (ii*fscrunch), tmpoutdata.begin() + ((ii+1)*fscrunch),0,add_f) +
                            //std::accumulate(tmpoutdata.begin() + ii, tmpoutdata.begin() + ii + 1,0,add_t));
                        }
                    }
                    ++timeindex;
                }
                else
                {
                    ++stepindex;
                    timeindex=0;
                    offset += tscrunch*skipallchans;
                    --ii;
                }
                tmpoutdata[ii] = sum;
            }
            tmpoutdata.resize(skipallchans*ngroups*nsamples/factor);
        }

        //Method 2
        /*if (fscrunch != 1)
        {
            for (std::size_t ii = 0; ii < tocopy/fscrunch; ++ii)
            {
                tmpoutdata[ii] = std::accumulate(tmpoutdata.begin() + ii*fscrunch, tmpoutdata.begin() + (ii + 1)* fscrunch, 0, add_f);
            }
        }

        if (tscrunch !=1)
        {
            for (std::size_t ii = 0; ii < tocopy/factor; ++ii)
            {
                if (ii < skipallchans*stepindex/fscrunch)
                {
                    for (std::size_t jj = 0; jj < tscrunch; ++jj)
                    {
                         tmpoutdata[ii] += (uint8_t)( (float)tmpoutdata[ (ii + offset ) + jj*skipallchans/fscrunch]/(float)(factor));
                    }
                }
                else
                {
                    ++stepindex;
                    offset += tscrunch*skipallchans/fscrunch;
                    --ii;
                }
            }
        }*/


        //copy to output
        std::copy(tmpoutdata.begin(),tmpoutdata.end(), transposed_data.ptr());
    }
} //transpose
} //tuse
} //meerkat
} //psrdada_cpp
