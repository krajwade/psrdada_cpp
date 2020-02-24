#include "psrdada_cpp/meerkat/fbfuse/PipelineConfig.hpp"
#include <fstream>

namespace psrdada_cpp {
namespace meerkat {
namespace fbfuse {

PipelineConfig::PipelineConfig()
    : _delay_buffer_shm("fbfuse_delays_shm")
    , _delay_buffer_mutex("fbfuse_delays_mutex")
    , _delay_buffer_sem("fbfuse_buffer_counter")
    , _gain_buffer_shm("fbfuse_gains_shm")
    , _gain_buffer_mutex("fbfuse_gains_mutex")
    , _gain_buffer_sem("fbfuse_gain_counter")
    , _input_dada_key(0xdada)
    , _cb_dada_key(0xcaca)
    , _ib_dada_key(0xeaea)
    , _channel_frequencies_stale(true)
    , _input_level(32.0f)
    , _output_level(24.0f)
    , _cb_power_scaling(0.0f)
    , _cb_power_offset(0.0f)
    , _ib_power_scaling(0.0f)
    , _ib_power_offset(0.0f)
{
    input_level(_input_level);
}

PipelineConfig::~PipelineConfig()
{
}

std::string const& PipelineConfig::delay_buffer_shm() const
{
    return _delay_buffer_shm;
}

void PipelineConfig::delay_buffer_shm(std::string const& key)
{
    _delay_buffer_shm = key;
}


std::string const& PipelineConfig::delay_buffer_mutex() const
{
    return _delay_buffer_mutex;
}

void PipelineConfig::delay_buffer_mutex(std::string const& key)
{
    _delay_buffer_mutex = key;
}

std::string const& PipelineConfig::delay_buffer_sem() const
{
    return _delay_buffer_sem;
}

void PipelineConfig::delay_buffer_sem(std::string const& key)
{
    _delay_buffer_sem = key;
}

std::string const& PipelineConfig::gain_buffer_shm() const
{
    return _gain_buffer_shm;
}

void PipelineConfig::gain_buffer_shm(std::string const& key)
{
    _gain_buffer_shm = key;
}


std::string const& PipelineConfig::gain_buffer_mutex() const
{
    return _gain_buffer_mutex;
}

void PipelineConfig::gain_buffer_mutex(std::string const& key)
{
    _gain_buffer_mutex = key;
}

std::string const& PipelineConfig::gain_buffer_sem() const
{
    return _gain_buffer_sem;
}

void PipelineConfig::gain_buffer_sem(std::string const& key)
{
    _gain_buffer_sem = key;
}

std::string const& PipelineConfig::channel_scaling_sem() const
{
    return _channel_scaling_sem;
}

void PipelineConfig::channel_scaling_sem(std::string const& key)
{
    _channel_scaling_sem = key;
}

key_t PipelineConfig::input_dada_key() const
{
    return _input_dada_key;
}

void PipelineConfig::input_dada_key(key_t key)
{
    _input_dada_key = key;
}

key_t PipelineConfig::cb_dada_key() const
{
    return _cb_dada_key;
}

void PipelineConfig::cb_dada_key(key_t key)
{
    _cb_dada_key = key;
}

key_t PipelineConfig::ib_dada_key() const
{
    return _ib_dada_key;
}

void PipelineConfig::ib_dada_key(key_t key)
{
    _ib_dada_key = key;
}

void PipelineConfig::output_level(float level)
{
    _output_level = level;
}

float PipelineConfig::output_level() const
{
    return _output_level;
}

double PipelineConfig::centre_frequency() const
{
    return _cfreq;
}

void PipelineConfig::centre_frequency(double cfreq)
{
    _cfreq = cfreq;
    _channel_frequencies_stale = true;
}

double PipelineConfig::bandwidth() const
{
    return _bw;
}

void PipelineConfig::bandwidth(double bw)
{
    _bw = bw;
    _channel_frequencies_stale = true;
}

std::vector<double> const& PipelineConfig::channel_frequencies() const
{
    if (_channel_frequencies_stale)
    {
        calculate_channel_frequencies();
    }
    return _channel_frequencies;
}

void PipelineConfig::calculate_channel_frequencies() const
{
    /**
     * Need to revisit this implementation as it is not clear how the
     * frequencies are labeled for the data out of the F-engine. Either
     * way is a roughly correct place-holder.
     */
    double chbw = bandwidth()/nchans();
    double fbottom = centre_frequency() - bandwidth()/2.0;
    _channel_frequencies.clear();
    for (std::size_t chan_idx=0; chan_idx < nchans(); ++chan_idx)
    {
        _channel_frequencies.push_back(fbottom + chbw/2.0 + (chbw * chan_idx));
    }
    _channel_frequencies_stale = false;
}

} //namespace fbfuse
} //namespace meerkat
} //namespace psrdada_cpp

