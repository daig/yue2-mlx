#pragma once
#include "ar.hpp"
namespace lyra {
mx::array nar_attention(const mx::array&,const mx::array&,const mx::array&,bool causal,int query_chunk_size);
}
