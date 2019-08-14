/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "utils.h"
#include <iostream>
#include <cassert>

NCCL_API(ncclResult_t, ncclGetDeviceLog, ncclProf_t* nccl_prof);
ncclResult_t ncclGetDeviceLog(ncclProf_t* nccl_prof) {
  if (nccl_prof != nullptr) {
      std::cout << "get device log\n";
      uint64_t end_micros = now_micros();

      commStat_t host_comm_stat;
      CUDACHECK(cudaMemcpy(&host_comm_stat, nccl_prof->dev_comm_stat,
			      sizeof(commStat_t), cudaMemcpyDeviceToHost));
      if (host_comm_stat.comm_type != NET_INTRA_E2E) {
        return ncclSuccess;
      }

      uint64_t start_clock = host_comm_stat.start_micros;
      uint64_t end_clock = host_comm_stat.end_micros;
      uint64_t start_micros = nccl_prof->kernel_start_micros;
      std::cout << "start_clock: " << start_clock << ", end_clock: " << end_clock << ", start_micros: " << start_micros << ", end_micros: " << end_micros << "\n";
      float micro_per_clock = float(end_micros - start_micros) / float(end_clock - start_clock);
      assert(micro_per_clock > 0);
      std::cout << "clock duraion: " << (end_clock - start_clock) << ", micro duration: " << (end_micros - start_micros) << ", micro_per_clock: " << micro_per_clock;

      int nElems = host_comm_stat.comm_bytes;
      std::cout << nElems;
      if (nElems > 0) { 
        for (int i = 0; i < nElems; i++) {
          commStat_t* intra_comm_stat = (commStat_t*) malloc(sizeof(commStat_t));
	  CUDACHECK(cudaMemcpy(intra_comm_stat, &nccl_prof->dev_comm_stat[i+1],
                              sizeof(commStat_t), cudaMemcpyDeviceToHost));
	  //intra_comm_stat->from_rank = from_rank;
	  //intra_comm_stat->to_rank = to_rank;
	  intra_comm_stat->start_micros = 
              (intra_comm_stat->start_micros - start_clock) * micro_per_clock + start_micros;
	  intra_comm_stat->end_micros =
              (intra_comm_stat->end_micros - start_clock) * micro_per_clock + start_micros;
	  nccl_prof->stat_vector->push_back(intra_comm_stat);
        }
      }
      CUDACHECK(cudaFree(nccl_prof->dev_comm_stat));
      std::cout << "cuda free\n";
  }
  return ncclSuccess;
}
