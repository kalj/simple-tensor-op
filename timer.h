/* -*- c-basic-offset:4; tab-width:4; indent-tabs-mode:nil -*-
 *
 * Based on code by
 *
 * Oliver Fringer
 * Stanford University
 *
 */
#ifndef _TIMER_H
#define _TIMER_H

#include <sys/time.h>
#include <cstdlib>

typedef double timeType;

timeType Timer(void) {
  struct timeval timeval_time;
  gettimeofday(&timeval_time,NULL);
  return (double)timeval_time.tv_sec + (double)timeval_time.tv_usec*1e-6;
}


#ifdef __CUDACC__
#define TIC(t) cudaDeviceSynchronize(); t = Timer()
#define TOC(t) (cudaDeviceSynchronize(), ( Timer() - t))
#else
#define TIC(t) t = Timer()
#define TOC(t) ( Timer() - t)
#endif

#define PTOC(t) printf("%g s\n",TOC(t))

#endif /* _TIMER_H */
