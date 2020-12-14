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
        //
        if (input_data.total_bytes() % (nfreq * nchans * nsamples * nbeams) != 0)
        {
            auto sug_size = nfreq * nchans * nsamples * nbeams * ngroups;
            throw std::runtime_error(std::string("Incorrect size of the DADA block. Should be a multiple of heap group size. Suggested size is:") + std::to_string(sug_size) + std::string("bytes"));
        }
        const size_t tocopy = ngroups * nsamples * nfreq * nchans;
        std::vector<char> tmpindata(tocopy / ngroups);
        std::vector<char>tmpoutdata(tocopy);
        std::copy(input_data.ptr(), input_data.ptr() + tocopy, tmpindata.begin());
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
                } // BAND LOOP*/

                /* Reverse the channel array */
                std::reverse(tmpoutdata.begin() + isamp * skipallchans + igroup *tocopy/ngroups, tmpoutdata.begin() + isamp * skipallchans + igroup *tocopy/ngroups + nfreq*nchans);

            } // SAMPLES LOOP
        } // GROUP LOOP*/

                /* Scrunching the data */
        std::size_t factor = tscrunch*fscrunch;
        std::size_t nsamps = tocopy/skipallchans;
        std::size_t new_nsamples = tocopy/skipallchans/tscrunch;
        std::size_t freqindex = 0, stepindex=1, offset=0;
        std::size_t new_size = tocopy/factor;
        std::size_t new_nchans = skipallchans/fscrunch;
        std::size_t outindex = 0;
        std::vector<char>tmpoutdata_scrunch(new_size,0);

        // Method 1
        /*if (tscrunch != 1 || fscrunch !=1)
        {
            for (std::size_t ii = 0; ii < new_nsamples; ++ii)
            {
                for (std::size_t jj = 0; jj < new_nchans; ++jj)
                {
                    float sum = 0.0;
                    for (std::size_t ll = 0; ll < tscrunch; ++ll)
                    {
                        for (std::size_t mm = 0; mm < fscrunch; ++ mm)
                        {
                            sum += static_cast<float>(tmpoutdata[mm + ll*skipallchans + jj*fscrunch + ii*skipallchans*tscrunch]);
                        }
                    }
                    tmpoutdata_scrunch[outindex] = static_cast<unsigned char>( ((sum/static_cast<float>(factor)) * std::sqrt(factor)) + 128.0);
                    ++outindex;
                }
            }
            std::copy(tmpoutdata_scrunch.begin(),tmpoutdata_scrunch.end(), transposed_data.ptr());
        }
        else
        {
            std::transform(tmpoutdata.begin(), tmpoutdata.end(), tmpoutdata.begin(), std::bind2nd(std::plus<char>(),128));
            std::copy(tmpoutdata.begin(),tmpoutdata.end(), transposed_data.ptr());
        }*/

        // Method 1
        auto add_f = [&](int8_t x, int8_t y)
            {
                return x + static_cast<int8_t>(static_cast<float>(y)/static_cast<float>(fscrunch));
            };


        if (fscrunch != 1)
        {
            for (std::size_t ii = 0; ii < tocopy/fscrunch; ++ii)
            {
                tmpoutdata[ii] = std::accumulate(tmpoutdata.begin() + ii*fscrunch, tmpoutdata.begin() + (ii + 1)* fscrunch, 0, add_f);
                tmpoutdata[ii] = static_cast<char>(static_cast<float>(tmpoutdata[ii]) * std::sqrt(fscrunch));
            }
            /* Scale the fscrunched data */
        }

        float sum=0;
        if (tscrunch !=1)
        {
            for (std::size_t ii = 0; ii < new_size; ++ii)
            {
                if (ii < new_nchans*stepindex)
                {
                    sum = 0;
                    for (std::size_t jj = 0; jj < tscrunch; ++jj)
                    {
                        sum = sum + static_cast<float>(tmpoutdata[ (freqindex + offset ) + jj*new_nchans]);
                    }
                    tmpoutdata_scrunch[ii] = static_cast<char>(sum/static_cast<float>(tscrunch) * std::sqrt(tscrunch));
                    ++freqindex;
                }
                else
                {
                    sum = 0;
                    ++stepindex;
                    freqindex = 0;
                    offset += tscrunch*new_nchans;
                    for (std::size_t jj = 0; jj < tscrunch; ++jj)
                    {
                        sum += static_cast<float>(tmpoutdata[ (freqindex + offset ) + jj*new_nchans]);
                    }
                    ++freqindex;
                    tmpoutdata_scrunch[ii] = static_cast<char>(sum/static_cast<float>(tscrunch) * std::sqrt(tscrunch) );
                }
            }
            // Convert to unsigned
            std::transform(tmpoutdata_scrunch.begin(), tmpoutdata_scrunch.end(), tmpoutdata_scrunch.begin(), std::bind2nd(std::plus<char>(),128));
            std::copy(tmpoutdata_scrunch.begin(),tmpoutdata_scrunch.end(), transposed_data.ptr());
        }
        else
        {
            //convert to unsigned
            std::transform(tmpoutdata.begin(), tmpoutdata.end(), tmpoutdata.begin(), std::bind2nd(std::plus<char>(),128));
            std::copy(tmpoutdata.begin(),tmpoutdata.begin() + new_size, transposed_data.ptr());
        }
        //copy to output
    }
} //transpose
} //tuse
} //meerkat
} //psrdada_cpp
