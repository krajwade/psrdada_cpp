#include "psrdada_cpp/meerkat/fbfuse/VoltageScaling.cuh"
#include "psrdada_cpp/meerkat/fbfuse/fbfuse_constants.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "psrdada_cpp/common.hpp"

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace kernels {

    /**
     *
     * We assume here that the gains have magnitude of order 1
     * We first do one pass with the channel scalings set to 1
     * Then calculate the statistics
     * Then set the channel scalings for the next pass
     *
     */
    __global__ void scale_voltages(
        char4* taftp_voltages_out,
        char4 const* taftp_voltages_in,
        float4 const* __restrict__ afp_gains,
        float const* __restrict__ f_channel_scalings,
        unsigned n_heap_groups)
    {
        // One block can handle one tp for all outer t
        // load one channel scaling
        // load one dual-poln gain correction

        // gridDim.x == Outer time
        // gridDim.y == Antenna
        // gridDim.z == Frequency channel
        // blockDim.x == Inner time + polarisation

        for (unsigned f_idx = blockIdx.z;
            f_idx < FBFUSE_NCHANS;
            f_idx += gridDim.z)
        {
            float scaling = f_channel_scalings[f_idx];
            for (unsigned a_idx = blockIdx.y;
                a_idx < FBFUSE_TOTAL_NANTENNAS;
                a_idx += gridDim.y)
            {
                float4 gain = afp_gains[a_idx * FBFUSE_NCHANS + f_idx];
                for (unsigned t_idx = blockIdx.x;
                    t_idx < n_heap_groups;
                    t_idx += gridDim.x)
                {
                    uint64_t offset = FBFUSE_NSAMPLES_PER_HEAP * (FBFUSE_NCHANS * (t_idx * FBFUSE_TOTAL_NANTENNAS + a_idx)  + f_idx) + threadIdx.x;
                    char4 data = taftp_voltages_in[offset];
                    float xx = data.x * gain.x;
                    float yy = data.y * gain.y;
                    float xy = data.x * gain.y;
                    float yx = data.y * gain.x;
                    float zz = data.z * gain.z;
                    float ww = data.w * gain.w;
                    float zw = data.z * gain.w;
                    float wz = data.w * gain.z;
                    data.x = (int8_t) fmaxf(-127.0f, fminf(127.0f, __float2int_rn(scaling * (xx - yy))));
                    data.y = (int8_t) fmaxf(-127.0f, fminf(127.0f, __float2int_rn(scaling * (xy + yx))));
                    data.z = (int8_t) fmaxf(-127.0f, fminf(127.0f, __float2int_rn(scaling * (zz - ww))));
                    data.w = (int8_t) fmaxf(-127.0f, fminf(127.0f, __float2int_rn(scaling * (zw + wz))));
                    taftp_voltages_out[offset] = data;
                }

            }

        }

    }

} //namespace kernels

void voltage_scaling(
    VoltageVectorType& taftp_voltages_out,
    VoltageVectorType const& taftp_voltages_in,
    ComplexGainVectorType const& afp_gains,
    ChannelScalesVectorType const& f_channel_scalings,
    cudaStream_t stream)
{
    BOOST_LOG_TRIVIAL(debug) << "Rescaling input votages with complex gains and channel scalings";
    std::size_t heap_group_size = FBFUSE_TOTAL_NANTENNAS * FBFUSE_NCHANS * FBFUSE_NSAMPLES_PER_HEAP * FBFUSE_NPOL;
    std::size_t n_heap_groups = taftp_voltages_in.size() / heap_group_size;
    BOOST_LOG_TRIVIAL(debug) << "Voltage buffer contains " << n_heap_groups << " heaps";
    if (taftp_voltages_in.size() % heap_group_size != 0)
    {
        std::stringstream ss;
        ss << "Voltage array is not a multiple of the heap group size ("
           << taftp_voltages_in.size() << " % " << heap_group_size
           << " != 0)";
        throw std::runtime_error(ss.str());
    }
    taftp_voltages_out.resize(taftp_voltages_in.size());
    char4* taftp_voltages_out_ptr = reinterpret_cast<char4*>(thrust::raw_pointer_cast(taftp_voltages_out.data()));
    char4 const* taftp_voltages_in_ptr = reinterpret_cast<char4 const*>(thrust::raw_pointer_cast(taftp_voltages_in.data()));
    float4 const* afp_gains_ptr = reinterpret_cast<float4 const*>(thrust::raw_pointer_cast(afp_gains.data()));
    float const* f_channel_scalings_ptr = reinterpret_cast<float const*>(thrust::raw_pointer_cast(f_channel_scalings.data()));
    dim3 blocks(n_heap_groups, FBFUSE_TOTAL_NANTENNAS, FBFUSE_NCHANS);
    kernels::scale_voltages<<<blocks, FBFUSE_NSAMPLES_PER_HEAP, 0, stream>>>(
        taftp_voltages_out_ptr,
        taftp_voltages_in_ptr,
        afp_gains_ptr,
        f_channel_scalings_ptr,
        n_heap_groups);
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    BOOST_LOG_TRIVIAL(debug) << "Voltage scalings applied";
}

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

