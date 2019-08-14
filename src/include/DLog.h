/***********************************************************

 *  DLog.h

 *  Simple logging mechanism for CUDA device code

 *

 *  Usage: 

 *	   1. include it in your .cu file

 *	   2. call DLog_Init(size) to reserve size bytes as output buffer 

 *		   on device

 *	   3. call DLog<<data in your CUDA device code

 *	   4. call DLog_Dump() after kernel invocation to dump the output

 *

 *  This implementation is very simple, and inefficient. It supports only 

 *	 char, int and const char *.

 *

 *  Author: Chuntao HONG (chuntao.hong@gmail.com)

 *  Last modified: Oct. 13th, 2009

************************************************************/
#ifndef NCCL_DLOG_H_
#define NCCL_DLOG_H_

#include <cuda.h>

#include <iostream>

const int MIN_ALIGN=4;



// align n to b bytes

template<class T>

__device__ T Align(T n, uint32_t b){

		return ((uint32_t)n&(b-1))==NULL ? n : n+b-((uint32_t)n&(b-1));

}



template<class T>

__device__ T MinAlign(T n){

		return ((uint32_t)n&(MIN_ALIGN-1))==NULL ? n : n+MIN_ALIGN-((uint32_t)n&(MIN_ALIGN-1));

}



// set some empty bytes, so that the logging data structures won't be polluted by out-of-bound writes

const int SAFTY_BYTES=1024*1024;



// data types supported, now only three

enum DataType{

		CHAR,

		INT,

		STRING

};



struct LogPack{

		int size;

		DataType type;

		union{

				char c;

				int i;

				char str[];

		}data;

};

struct MyLog{

public:

		char empty_bytes[SAFTY_BYTES];

		char * buf;

		unsigned int curPos;

		char empty_bytes2[SAFTY_BYTES];

public:

		template <class T>

		__device__ MyLog & operator<<(const T & t){

				int packSize=MinAlign(requiredLogSpace(t));

				int start=atomicAdd(&curPos, packSize);

				LogPack * pack=(LogPack *)(buf+start);

				assignLog(pack,t);

				pack->size=packSize;

				return *this;

		}

};

static __device__ int requiredLogSpace(const int i){

		return sizeof(i)+sizeof(DataType)+sizeof(int);

}

static __device__ int requiredLogSpace(const char * str){

		int len=0;

		while(*str++){

				len++;

		}

		return len+1+sizeof(DataType)+sizeof(int);

}

static __device__ int requiredLogSpace(const unsigned int i){

		return sizeof(i)+sizeof(DataType)+sizeof(int);

}

static __device__ int requiredLogSpace(const char c){

		return sizeof(c)+sizeof(DataType)+sizeof(int);

}



static __device__ void assignLog(LogPack * pack, const char c){

		pack->data.c=c;

		pack->type=CHAR;

}

static __device__ void assignLog(LogPack * pack, const int i){

		pack->data.i=i;

		pack->type=INT;

}

static __device__ void assignLog(LogPack * pack, const unsigned int i){

		pack->data.i=i;

		pack->type=INT;

}

static __device__ void assignLog(LogPack * pack, const char * str){

		int i=0;

		while(*str){

				pack->data.str[i++]=*str;

				str++;

		}

		pack->type=STRING;

}

#endif
