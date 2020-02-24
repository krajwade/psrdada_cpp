#ifndef PSRDADA_CPP_MEERKAT_FBFUSE_VOLTAGESCALING_HPP
#define PSRDADA_CPP_MEERKAT_FBFUSE_VOLTAGESCALING_HPP

#include <thrust/device_vector.h>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {
namespace kernels {

__global__ void scale_voltages(
    char4* taftp_voltages_out,
    char4 const* taftp_voltages_in,
    float4 const* __restrict__ afp_gains,
    float const* __restrict__ channel_scalings,
    unsigned n_heap_groups);

} //namespace kernels

typedef thrust::device_vector<char2> VoltageVectorType;
typedef thrust::device_vector<float2> ComplexGainVectorType;
typedef thrust::device_vector<float> ChannelScalesVectorType;

void voltage_scaling(
    VoltageVectorType&,
    VoltageVectorType const&,
    ComplexGainVectorType const&,
    ChannelScalesVectorType const&);

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

#endif // PSRDADA_CPP_MEERKAT_FBFUSE_VOLTAGESCALING_HPP


