/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * You may obtain a copy of the license in the LICENSE file
 * located in the root directory of this source tree or at:
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once
#include<cuda.h>
#include <cub/cub.cuh>
#include<vector>
#include<cmath>
#include<string>
#include<fstream>
#include<iostream>
#include<sstream>
#include<stdlib.h>
#include<sys/time.h>
#include<algorithm>
#include<random>


namespace gansu::gpu{

__device__ inline double MD_Et_GPU000(double a, double b, double p, double d, double s){
	return s;
}

__device__ inline double MD_Et_GPU010(double a, double b, double p, double d, double s){
	return (a*d/p)*s;
}

__device__ inline double MD_Et_GPU011(double a, double b, double p, double d, double s){
	return (0.5/p)*s;
}

__device__ inline double MD_Et_GPU020(double a, double b, double p, double d, double s){
	return (((a*a)*(d*d) + 0.5*p)/(p*p))*s;
}

__device__ inline double MD_Et_GPU021(double a, double b, double p, double d, double s){
	return (a*d/(p*p))*s;
}

__device__ inline double MD_Et_GPU022(double a, double b, double p, double d, double s){
	return (0.25/(p*p))*s;
}

__device__ inline double MD_Et_GPU030(double a, double b, double p, double d, double s){
	return (a*d*((a*a)*(d*d) + 1.5*p)/(p*p*p))*s;
}

__device__ inline double MD_Et_GPU031(double a, double b, double p, double d, double s){
	return ((1.5*(a*a)*(d*d) + 0.75*p)/(p*p*p))*s;
}

__device__ inline double MD_Et_GPU032(double a, double b, double p, double d, double s){
	return (0.75*a*d/(p*p*p))*s;
}

__device__ inline double MD_Et_GPU033(double a, double b, double p, double d, double s){
	return (0.125/(p*p*p))*s;
}

__device__ inline double MD_Et_GPU040(double a, double b, double p, double d, double s){
	return (((a*a*a*a)*(d*d*d*d) + 3.0*(a*a)*(d*d)*p + 0.75*(p*p))/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU041(double a, double b, double p, double d, double s){
	return (a*d*(2.0*(a*a)*(d*d) + 3.0*p)/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU042(double a, double b, double p, double d, double s){
	return ((1.5*(a*a)*(d*d) + 0.75*p)/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU043(double a, double b, double p, double d, double s){
	return (0.5*a*d/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU044(double a, double b, double p, double d, double s){
	return (0.0625/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU050(double a, double b, double p, double d, double s){
	return (a*d*((a*a*a*a)*(d*d*d*d) + 5.0*(a*a)*(d*d)*p + 3.75*(p*p))/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU051(double a, double b, double p, double d, double s){
	return ((2.5*(a*a*a*a)*(d*d*d*d) + 7.5*(a*a)*(d*d)*p + 1.875*(p*p))/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU052(double a, double b, double p, double d, double s){
	return (a*d*(2.5*(a*a)*(d*d) + 3.75*p)/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU053(double a, double b, double p, double d, double s){
	return ((1.25*(a*a)*(d*d) + 0.625*p)/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU054(double a, double b, double p, double d, double s){
	return (0.3125*a*d/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU055(double a, double b, double p, double d, double s){
	return (0.03125/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU060(double a, double b, double p, double d, double s){
	return (((a*a*a*a*a*a)*(d*d*d*d*d*d) + 7.5*(a*a*a*a)*(d*d*d*d)*p + 11.25*(a*a)*(d*d)*(p*p) + 1.875*(p*p*p))/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU061(double a, double b, double p, double d, double s){
	return (a*d*(3.0*(a*a*a*a)*(d*d*d*d) + 15.0*(a*a)*(d*d)*p + 11.25*(p*p))/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU062(double a, double b, double p, double d, double s){
	return ((3.75*(a*a*a*a)*(d*d*d*d) + 11.25*(a*a)*(d*d)*p + 2.8125*(p*p))/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU063(double a, double b, double p, double d, double s){
	return (a*d*(2.5*(a*a)*(d*d) + 3.75*p)/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU064(double a, double b, double p, double d, double s){
	return ((0.9375*(a*a)*(d*d) + 0.46875*p)/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU065(double a, double b, double p, double d, double s){
	return (0.1875*a*d/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU066(double a, double b, double p, double d, double s){
	return (0.015625/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU070(double a, double b, double p, double d, double s){
	return (a*d*((a*a*a*a*a*a)*(d*d*d*d*d*d) + 10.5*(a*a*a*a)*(d*d*d*d)*p + 26.25*(a*a)*(d*d)*(p*p) + 13.125*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU071(double a, double b, double p, double d, double s){
	return ((3.5*(a*a*a*a*a*a)*(d*d*d*d*d*d) + 26.25*(a*a*a*a)*(d*d*d*d)*p + 39.375*(a*a)*(d*d)*(p*p) + 6.5625*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU072(double a, double b, double p, double d, double s){
	return (a*d*(5.25*(a*a*a*a)*(d*d*d*d) + 26.25*(a*a)*(d*d)*p + 19.6875*(p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU073(double a, double b, double p, double d, double s){
	return ((4.375*(a*a*a*a)*(d*d*d*d) + 13.125*(a*a)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU074(double a, double b, double p, double d, double s){
	return (a*d*(2.1875*(a*a)*(d*d) + 3.28125*p)/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU075(double a, double b, double p, double d, double s){
	return ((0.65625*(a*a)*(d*d) + 0.328125*p)/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU076(double a, double b, double p, double d, double s){
	return (0.109375*a*d/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU077(double a, double b, double p, double d, double s){
	return (0.0078125/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU080(double a, double b, double p, double d, double s){
	return (((a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d) + 14.0*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p + 52.5*(a*a*a*a)*(d*d*d*d)*(p*p) + 52.5*(a*a)*(d*d)*(p*p*p) + 6.5625*(p*p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU081(double a, double b, double p, double d, double s){
	return (a*d*(4.0*(a*a*a*a*a*a)*(d*d*d*d*d*d) + 42.0*(a*a*a*a)*(d*d*d*d)*p + 105.0*(a*a)*(d*d)*(p*p) + 52.5*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU082(double a, double b, double p, double d, double s){
	return ((7.0*(a*a*a*a*a*a)*(d*d*d*d*d*d) + 52.5*(a*a*a*a)*(d*d*d*d)*p + 78.75*(a*a)*(d*d)*(p*p) + 13.125*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU083(double a, double b, double p, double d, double s){
	return (a*d*(7.0*(a*a*a*a)*(d*d*d*d) + 35.0*(a*a)*(d*d)*p + 26.25*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU084(double a, double b, double p, double d, double s){
	return ((4.375*(a*a*a*a)*(d*d*d*d) + 13.125*(a*a)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU085(double a, double b, double p, double d, double s){
	return (a*d*(1.75*(a*a)*(d*d) + 2.625*p)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU086(double a, double b, double p, double d, double s){
	return ((0.4375*(a*a)*(d*d) + 0.21875*p)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU087(double a, double b, double p, double d, double s){
	return (0.0625*a*d/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU088(double a, double b, double p, double d, double s){
	return (0.00390625/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU100(double a, double b, double p, double d, double s){
	return (-b*d/p)*s;
}

__device__ inline double MD_Et_GPU101(double a, double b, double p, double d, double s){
	return (0.5/p)*s;
}

__device__ inline double MD_Et_GPU110(double a, double b, double p, double d, double s){
	return ((-a*b*(d*d) + 0.5*p)/(p*p))*s;
}

__device__ inline double MD_Et_GPU111(double a, double b, double p, double d, double s){
	return (0.5*d*(a - b)/(p*p))*s;
}

__device__ inline double MD_Et_GPU112(double a, double b, double p, double d, double s){
	return (0.25/(p*p))*s;
}

__device__ inline double MD_Et_GPU120(double a, double b, double p, double d, double s){
	return (d*(-(a*a)*b*(d*d) + a*p - 0.5*b*p)/(p*p*p))*s;
}

__device__ inline double MD_Et_GPU121(double a, double b, double p, double d, double s){
	return ((0.5*(a*a)*(d*d) - a*b*(d*d) + 0.75*p)/(p*p*p))*s;
}

__device__ inline double MD_Et_GPU122(double a, double b, double p, double d, double s){
	return (d*(0.5*a - 0.25*b)/(p*p*p))*s;
}

__device__ inline double MD_Et_GPU123(double a, double b, double p, double d, double s){
	return (0.125/(p*p*p))*s;
}

__device__ inline double MD_Et_GPU130(double a, double b, double p, double d, double s){
	return ((-(a*a*a)*b*(d*d*d*d) + 1.5*(a*a)*(d*d)*p - 1.5*a*b*(d*d)*p + 0.75*(p*p))/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU131(double a, double b, double p, double d, double s){
	return (d*(0.5*(a*a*a)*(d*d) - 1.5*(a*a)*b*(d*d) + 2.25*a*p - 0.75*b*p)/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU132(double a, double b, double p, double d, double s){
	return (0.75*((a*a)*(d*d) - a*b*(d*d) + p)/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU133(double a, double b, double p, double d, double s){
	return (d*(0.375*a - 0.125*b)/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU134(double a, double b, double p, double d, double s){
	return (0.0625/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU140(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a)*b*(d*d*d*d) + 2.0*(a*a*a)*(d*d)*p - 3.0*(a*a)*b*(d*d)*p + 3.0*a*(p*p) - 0.75*b*(p*p))/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU141(double a, double b, double p, double d, double s){
	return ((0.5*(a*a*a*a)*(d*d*d*d) - 2.0*(a*a*a)*b*(d*d*d*d) + 4.5*(a*a)*(d*d)*p - 3.0*a*b*(d*d)*p + 1.875*(p*p))/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU142(double a, double b, double p, double d, double s){
	return (d*((a*a*a)*(d*d) - 1.5*(a*a)*b*(d*d) + 3.0*a*p - 0.75*b*p)/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU143(double a, double b, double p, double d, double s){
	return ((0.75*(a*a)*(d*d) - 0.5*a*b*(d*d) + 0.625*p)/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU144(double a, double b, double p, double d, double s){
	return (d*(0.25*a - 0.0625*b)/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU145(double a, double b, double p, double d, double s){
	return (0.03125/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU150(double a, double b, double p, double d, double s){
	return ((-(a*a*a*a*a)*b*(d*d*d*d*d*d) + 2.5*(a*a*a*a)*(d*d*d*d)*p - 5.0*(a*a*a)*b*(d*d*d*d)*p + 7.5*(a*a)*(d*d)*(p*p) - 3.75*a*b*(d*d)*(p*p) + 1.875*(p*p*p))/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU151(double a, double b, double p, double d, double s){
	return (d*(0.5*(a*a*a*a*a)*(d*d*d*d) - 2.5*(a*a*a*a)*b*(d*d*d*d) + 7.5*(a*a*a)*(d*d)*p - 7.5*(a*a)*b*(d*d)*p + 9.375*a*(p*p) - 1.875*b*(p*p))/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU152(double a, double b, double p, double d, double s){
	return ((1.25*(a*a*a*a)*(d*d*d*d) - 2.5*(a*a*a)*b*(d*d*d*d) + 7.5*(a*a)*(d*d)*p - 3.75*a*b*(d*d)*p + 2.8125*(p*p))/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU153(double a, double b, double p, double d, double s){
	return (d*(1.25*(a*a*a)*(d*d) - 1.25*(a*a)*b*(d*d) + 3.125*a*p - 0.625*b*p)/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU154(double a, double b, double p, double d, double s){
	return ((0.625*(a*a)*(d*d) - 0.3125*a*b*(d*d) + 0.46875*p)/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU155(double a, double b, double p, double d, double s){
	return (d*(0.15625*a - 0.03125*b)/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU156(double a, double b, double p, double d, double s){
	return (0.015625/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU160(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 3.0*(a*a*a*a*a)*(d*d*d*d)*p - 7.5*(a*a*a*a)*b*(d*d*d*d)*p + 15.0*(a*a*a)*(d*d)*(p*p) - 11.25*(a*a)*b*(d*d)*(p*p) + 11.25*a*(p*p*p) - 1.875*b*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU161(double a, double b, double p, double d, double s){
	return ((0.5*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 3.0*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 11.25*(a*a*a*a)*(d*d*d*d)*p - 15.0*(a*a*a)*b*(d*d*d*d)*p + 28.125*(a*a)*(d*d)*(p*p) - 11.25*a*b*(d*d)*(p*p) + 6.5625*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU162(double a, double b, double p, double d, double s){
	return (d*(1.5*(a*a*a*a*a)*(d*d*d*d) - 3.75*(a*a*a*a)*b*(d*d*d*d) + 15.0*(a*a*a)*(d*d)*p - 11.25*(a*a)*b*(d*d)*p + 16.875*a*(p*p) - 2.8125*b*(p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU163(double a, double b, double p, double d, double s){
	return ((1.875*(a*a*a*a)*(d*d*d*d) - 2.5*(a*a*a)*b*(d*d*d*d) + 9.375*(a*a)*(d*d)*p - 3.75*a*b*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU164(double a, double b, double p, double d, double s){
	return (d*(1.25*(a*a*a)*(d*d) - 0.9375*(a*a)*b*(d*d) + 2.8125*a*p - 0.46875*b*p)/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU165(double a, double b, double p, double d, double s){
	return ((0.46875*(a*a)*(d*d) - 0.1875*a*b*(d*d) + 0.328125*p)/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU166(double a, double b, double p, double d, double s){
	return (d*(0.09375*a - 0.015625*b)/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU167(double a, double b, double p, double d, double s){
	return (0.0078125/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU170(double a, double b, double p, double d, double s){
	return ((-(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 3.5*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 10.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 26.25*(a*a*a*a)*(d*d*d*d)*(p*p) - 26.25*(a*a*a)*b*(d*d*d*d)*(p*p) + 39.375*(a*a)*(d*d)*(p*p*p) - 13.125*a*b*(d*d)*(p*p*p) + 6.5625*(p*p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU171(double a, double b, double p, double d, double s){
	return (d*(0.5*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 3.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 15.75*(a*a*a*a*a)*(d*d*d*d)*p - 26.25*(a*a*a*a)*b*(d*d*d*d)*p + 65.625*(a*a*a)*(d*d)*(p*p) - 39.375*(a*a)*b*(d*d)*(p*p) + 45.9375*a*(p*p*p) - 6.5625*b*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU172(double a, double b, double p, double d, double s){
	return ((1.75*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 5.25*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 26.25*(a*a*a*a)*(d*d*d*d)*p - 26.25*(a*a*a)*b*(d*d*d*d)*p + 59.0625*(a*a)*(d*d)*(p*p) - 19.6875*a*b*(d*d)*(p*p) + 13.125*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU173(double a, double b, double p, double d, double s){
	return (d*(2.625*(a*a*a*a*a)*(d*d*d*d) - 4.375*(a*a*a*a)*b*(d*d*d*d) + 21.875*(a*a*a)*(d*d)*p - 13.125*(a*a)*b*(d*d)*p + 22.96875*a*(p*p) - 3.28125*b*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU174(double a, double b, double p, double d, double s){
	return ((2.1875*(a*a*a*a)*(d*d*d*d) - 2.1875*(a*a*a)*b*(d*d*d*d) + 9.84375*(a*a)*(d*d)*p - 3.28125*a*b*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU175(double a, double b, double p, double d, double s){
	return (d*(1.09375*(a*a*a)*(d*d) - 0.65625*(a*a)*b*(d*d) + 2.296875*a*p - 0.328125*b*p)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU176(double a, double b, double p, double d, double s){
	return ((0.328125*(a*a)*(d*d) - 0.109375*a*b*(d*d) + 0.21875*p)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU177(double a, double b, double p, double d, double s){
	return (d*(0.0546875*a - 0.0078125*b)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU178(double a, double b, double p, double d, double s){
	return (0.00390625/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU180(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 4.0*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 14.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 42.0*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 52.5*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 105.0*(a*a*a)*(d*d)*(p*p*p) - 52.5*(a*a)*b*(d*d)*(p*p*p) + 52.5*a*(p*p*p*p) - 6.5625*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU181(double a, double b, double p, double d, double s){
	return ((0.5*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d) - 4.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 21.0 *(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 42.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 131.25*(a*a*a*a)*(d*d*d*d)*(p*p) - 105.0*(a*a*a)*b*(d*d*d*d)*(p*p) + 183.75*(a*a)*(d*d)*(p*p*p) - 52.5*a*b*(d*d)*(p*p*p) + 29.53125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU182(double a, double b, double p, double d, double s){
	return (d*(2.0*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 7.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 42.0*(a*a*a*a*a)*(d*d*d*d)*p - 52.5*(a*a*a*a)*b*(d*d*d*d)*p + 157.5*(a*a*a)*(d*d)*(p*p) - 78.75*(a*a)*b*(d*d)*(p*p) + 105.0*a*(p*p*p) - 13.125*b*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU183(double a, double b, double p, double d, double s){
	return ((3.5*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 7.0*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 43.75*(a*a*a*a)*(d*d*d*d)*p - 35.0*(a*a*a)*b*(d*d*d*d)*p + 91.875*(a*a)*(d*d)*(p*p) - 26.25*a*b*(d*d)*(p*p) + 19.6875*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU184(double a, double b, double p, double d, double s){
	return (d*(3.5*(a*a*a*a*a)*(d*d*d*d) - 4.375*(a*a*a*a)*b*(d*d*d*d) + 26.25*(a*a*a)*(d*d)*p - 13.125*(a*a)*b*(d*d)*p + 26.25*a*(p*p) - 3.28125*b*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU185(double a, double b, double p, double d, double s){
	return ((2.1875*(a*a*a*a)*(d*d*d*d) - 1.75*(a*a*a)*b*(d*d*d*d) + 9.1875*(a*a)*(d*d)*p - 2.625*a*b*(d*d)*p + 2.953125*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU186(double a, double b, double p, double d, double s){
	return (d*(0.875*(a*a*a)*(d*d) - 0.4375*(a*a)*b*(d*d) + 1.75*a*p - 0.21875*b*p)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU187(double a, double b, double p, double d, double s){
	return ((0.21875*(a*a)*(d*d) - 0.0625*a*b*(d*d) + 0.140625*p)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU188(double a, double b, double p, double d, double s){
	return (d*(0.03125*a - 0.00390625*b)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU189(double a, double b, double p, double d, double s){
	return (0.001953125/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU200(double a, double b, double p, double d, double s){
	return (((b*b)*(d*d) + 0.5*p)/(p*p))*s;
}

__device__ inline double MD_Et_GPU201(double a, double b, double p, double d, double s){
	return (-b*d/(p*p))*s;
}

__device__ inline double MD_Et_GPU202(double a, double b, double p, double d, double s){
	return (0.25/(p*p))*s;
}

__device__ inline double MD_Et_GPU210(double a, double b, double p, double d, double s){
	return (d*(a*(b*b)*(d*d) + 0.5*a*p - b*p)/(p*p*p))*s;
}

__device__ inline double MD_Et_GPU211(double a, double b, double p, double d, double s){
	return ((-a*b*(d*d) + 0.5*(b*b)*(d*d) + 0.75*p)/(p*p*p))*s;
}

__device__ inline double MD_Et_GPU212(double a, double b, double p, double d, double s){
	return (d*(0.25*a - 0.5*b)/(p*p*p))*s;
}

__device__ inline double MD_Et_GPU213(double a, double b, double p, double d, double s){
	return (0.125/(p*p*p))*s;
}

__device__ inline double MD_Et_GPU220(double a, double b, double p, double d, double s){
	return (((a*a)*(b*b)*(d*d*d*d) + 0.5*(a*a)*(d*d)*p - 2.0*a*b*(d*d)*p + 0.5*(b*b)*(d*d)*p + 0.75*(p*p))/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU221(double a, double b, double p, double d, double s){
	return (d*(-(a*a)*b*(d*d) + a*(b*b)*(d*d) + 1.5*a*p - 1.5*b*p)/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU222(double a, double b, double p, double d, double s){
	return ((0.25*(a*a)*(d*d) - a*b*(d*d) + 0.25*(b*b)*(d*d) + 0.75*p)/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU223(double a, double b, double p, double d, double s){
	return (0.25*d*(a - b)/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU224(double a, double b, double p, double d, double s){
	return (0.0625/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU230(double a, double b, double p, double d, double s){
	return (d*((a*a*a)*(b*b)*(d*d*d*d) + 0.5*(a*a*a)*(d*d)*p - 3.0*(a*a)*b*(d*d)*p + 1.5*a*(b*b)*(d*d)*p + 2.25*a*(p*p) - 1.5*b*(p*p))/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU231(double a, double b, double p, double d, double s){
	return ((-(a*a*a)*b*(d*d*d*d) + 1.5*(a*a)*(b*b)*(d*d*d*d) + 2.25*(a*a)*(d*d)*p - 4.5*a*b*(d*d)*p + 0.75*(b*b)*(d*d)*p + 1.875*(p*p))/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU232(double a, double b, double p, double d, double s){
	return (d*(0.25*(a*a*a)*(d*d) - 1.5*(a*a)*b*(d*d) + 0.75*a*(b*b)*(d*d) + 2.25*a*p - 1.5*b*p)/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU233(double a, double b, double p, double d, double s){
	return ((0.375*(a*a)*(d*d) - 0.75*a*b*(d*d) + 0.125*(b*b)*(d*d) + 0.625*p)/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU234(double a, double b, double p, double d, double s){
	return (d*(0.1875*a - 0.125*b)/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU235(double a, double b, double p, double d, double s){
	return (0.03125/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU240(double a, double b, double p, double d, double s){
	return (((a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.5*(a*a*a*a)*(d*d*d*d)*p - 4.0*(a*a*a)*b*(d*d*d*d)*p + 3.0*(a*a)*(b*b)*(d*d*d*d)*p + 4.5*(a*a)*(d*d)*(p*p) - 6.0*a*b*(d*d)*(p*p) + 0.75*(b*b)*(d*d)*(p*p) + 1.875*(p*p*p))/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU241(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a)*b*(d*d*d*d) + 2.0*(a*a*a)*(b*b)*(d*d*d*d) + 3.0*(a*a*a)*(d*d)*p - 9.0*(a*a)*b*(d*d)*p + 3.0*a*(b*b)*(d*d)*p + 7.5*a*(p*p) - 3.75*b*(p*p))/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU242(double a, double b, double p, double d, double s){
	return ((0.25*(a*a*a*a)*(d*d*d*d) - 2.0*(a*a*a)*b*(d*d*d*d) + 1.5*(a*a)*(b*b)*(d*d*d*d) + 4.5*(a*a)*(d*d)*p - 6.0*a*b*(d*d)*p + 0.75*(b*b)*(d*d)*p + 2.8125*(p*p))/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU243(double a, double b, double p, double d, double s){
	return (d*(0.5*(a*a*a)*(d*d) - 1.5*(a*a)*b*(d*d) + 0.5*a*(b*b)*(d*d) + 2.5*a*p - 1.25*b*p)/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU244(double a, double b, double p, double d, double s){
	return ((0.375*(a*a)*(d*d) - 0.5*a*b*(d*d) + 0.0625*(b*b)*(d*d) + 0.46875*p)/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU245(double a, double b, double p, double d, double s){
	return (d*(0.125*a - 0.0625*b)/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU246(double a, double b, double p, double d, double s){
	return (0.015625/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU250(double a, double b, double p, double d, double s){
	return (d*((a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.5*(a*a*a*a*a)*(d*d*d*d)*p - 5.0*(a*a*a*a)*b*(d*d*d*d)*p + 5.0*(a*a*a)*(b*b)*(d*d*d*d)*p + 7.5*(a*a*a)*(d*d)*(p*p) - 15.0*(a*a)*b*(d*d)*(p*p) + 3.75*a*(b*b)*(d*d)*(p*p) + 9.375*a*(p*p*p) - 3.75*b*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU251(double a, double b, double p, double d, double s){
	return ((-(a*a*a*a*a)*b*(d*d*d*d*d*d) + 2.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 3.75*(a*a*a*a)*(d*d*d*d)*p - 15.0*(a*a*a)*b*(d*d*d*d)*p + 7.5*(a*a)*(b*b)*(d*d*d*d)*p + 18.75*(a*a)*(d*d)*(p*p) - 18.75*a*b*(d*d)*(p*p) + 1.875*(b*b)*(d*d)*(p*p) + 6.5625*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU252(double a, double b, double p, double d, double s){
	return (d*(0.25*(a*a*a*a*a)*(d*d*d*d) - 2.5*(a*a*a*a)*b*(d*d*d*d) + 2.5*(a*a*a)*(b*b)*(d*d*d*d) + 7.5*(a*a*a)*(d*d)*p - 15.0*(a*a)*b*(d*d)*p + 3.75*a*(b*b)*(d*d)*p + 14.0625*a*(p*p) - 5.625*b*(p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU253(double a, double b, double p, double d, double s){
	return ((0.625*(a*a*a*a)*(d*d*d*d) - 2.5*(a*a*a)*b*(d*d*d*d) + 1.25*(a*a)*(b*b)*(d*d*d*d) + 6.25*(a*a)*(d*d)*p - 6.25*a*b*(d*d)*p + 0.625*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU254(double a, double b, double p, double d, double s){
	return (d*(0.625*(a*a*a)*(d*d) - 1.25*(a*a)*b*(d*d) + 0.3125*a*(b*b)*(d*d) + 2.34375*a*p - 0.9375*b*p)/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU255(double a, double b, double p, double d, double s){
	return ((0.3125*(a*a)*(d*d) - 0.3125*a*b*(d*d) + 0.03125*(b*b)*(d*d) + 0.328125*p)/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU256(double a, double b, double p, double d, double s){
	return (d*(0.078125*a - 0.03125*b)/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU257(double a, double b, double p, double d, double s){
	return (0.0078125/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU260(double a, double b, double p, double d, double s){
	return (((a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.5*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 6.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 7.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 11.25*(a*a*a*a)*(d*d*d*d)*(p*p) - 30.0*(a*a*a)*b*(d*d*d*d)*(p*p) + 11.25*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 28.125*(a*a)*(d*d)*(p*p*p) - 22.5*a*b*(d*d)*(p*p*p) + 1.875*(b*b)*(d*d)*(p*p*p) + 6.5625*(p*p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU261(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 3.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 4.5*(a*a*a*a*a)*(d*d*d*d)*p - 22.5*(a*a*a*a)*b*(d*d*d*d)*p + 15.0*(a*a*a)*(b*b)*(d*d*d*d)*p + 37.5*(a*a*a)*(d*d)*(p*p) - 56.25*(a*a)*b*(d*d)*(p*p) + 11.25*a*(b*b)*(d*d)*(p*p) + 39.375*a*(p*p*p) - 13.125*b*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU262(double a, double b, double p, double d, double s){
	return ((0.25*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 3.0*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 3.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 11.25*(a*a*a*a)*(d*d*d*d)*p - 30.0*(a*a*a)*b*(d*d*d*d)*p + 11.25*(a*a)*(b*b)*(d*d*d*d)*p + 42.1875*(a*a)*(d*d)*(p*p) - 33.75*a*b*(d*d)*(p*p) + 2.8125*(b*b)*(d*d)*(p*p) + 13.125*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU263(double a, double b, double p, double d, double s){
	return (d*(0.75*(a*a*a*a*a)*(d*d*d*d) - 3.75*(a*a*a*a)*b*(d*d*d*d) + 2.5*(a*a*a)*(b*b)*(d*d*d*d) + 12.5*(a*a*a)*(d*d)*p - 18.75*(a*a)*b*(d*d)*p + 3.75*a*(b*b)*(d*d)*p + 19.6875*a*(p*p) - 6.5625*b*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU264(double a, double b, double p, double d, double s){
	return ((0.9375*(a*a*a*a)*(d*d*d*d) - 2.5*(a*a*a)*b*(d*d*d*d) + 0.9375*(a*a)*(b*b)*(d*d*d*d) + 7.03125*(a*a)*(d*d)*p - 5.625*a*b*(d*d)*p + 0.46875*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU265(double a, double b, double p, double d, double s){
	return (d*(0.625*(a*a*a)*(d*d) - 0.9375*(a*a)*b*(d*d) + 0.1875*a*(b*b)*(d*d) + 1.96875*a*p - 0.65625*b*p)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU266(double a, double b, double p, double d, double s){
	return ((0.234375*(a*a)*(d*d) - 0.1875*a*b*(d*d) + 0.015625*(b*b)*(d*d) + 0.21875*p)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU267(double a, double b, double p, double d, double s){
	return (d*(0.046875*a - 0.015625*b)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU268(double a, double b, double p, double d, double s){
	return (0.00390625/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU270(double a, double b, double p, double d, double s){
	return (d*((a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.5*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 7.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 10.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 15.75*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 52.5*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 26.25*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 65.625*(a*a*a)*(d*d)*(p*p*p) - 78.75*(a*a)*b*(d*d)*(p*p*p) + 13.125*a*(b*b)*(d*d)*(p*p*p) + 45.9375*a*(p*p*p*p) - 13.125*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU271(double a, double b, double p, double d, double s){
	return ((-(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 3.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 5.25*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 31.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 26.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 65.625*(a*a*a*a)*(d*d*d*d)*(p*p) - 131.25*(a*a*a)*b*(d*d*d*d)*(p*p) + 39.375*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 137.8125*(a*a)*(d*d)*(p*p*p) - 91.875*a*b*(d*d)*(p*p*p) + 6.5625*(b*b)*(d*d)*(p*p*p) + 29.53125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU272(double a, double b, double p, double d, double s){
	return (d*(0.25*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 3.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 5.25*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 15.75*(a*a*a*a*a)*(d*d*d*d)*p - 52.5*(a*a*a*a)*b*(d*d*d*d)*p + 26.25*(a*a*a)*(b*b)*(d*d*d*d)*p + 98.4375*(a*a*a)*(d*d)*(p*p) - 118.125*(a*a)*b*(d*d)*(p*p) + 19.6875*a*(b*b)*(d*d)*(p*p) + 91.875*a*(p*p*p) - 26.25*b*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU273(double a, double b, double p, double d, double s){
	return ((0.875*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 5.25*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 4.375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 21.875*(a*a*a*a)*(d*d*d*d)*p - 43.75*(a*a*a)*b*(d*d*d*d)*p + 13.125*(a*a)*(b*b)*(d*d*d*d)*p + 68.90625*(a*a)*(d*d)*(p*p) - 45.9375*a*b*(d*d)*(p*p) + 3.28125*(b*b)*(d*d)*(p*p) + 19.6875*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU274(double a, double b, double p, double d, double s){
	return (d*(1.3125*(a*a*a*a*a)*(d*d*d*d) - 4.375*(a*a*a*a)*b*(d*d*d*d) + 2.1875*(a*a*a)*(b*b)*(d*d*d*d) + 16.40625*(a*a*a)*(d*d)*p - 19.6875*(a*a)*b*(d*d)*p + 3.28125*a*(b*b)*(d*d)*p + 22.96875*a*(p*p) - 6.5625*b*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU275(double a, double b, double p, double d, double s){
	return ((1.09375*(a*a*a*a)*(d*d*d*d) - 2.1875*(a*a*a)*b*(d*d*d*d) + 0.65625*(a*a)*(b*b)*(d*d*d*d) + 6.890625*(a*a)*(d*d)*p - 4.59375*a*b*(d*d)*p + 0.328125*(b*b)*(d*d)*p + 2.953125*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU276(double a, double b, double p, double d, double s){
	return (d*(0.546875*(a*a*a)*(d*d) - 0.65625*(a*a)*b*(d*d) + 0.109375*a*(b*b)*(d*d) + 1.53125*a*p - 0.4375*b*p)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU277(double a, double b, double p, double d, double s){
	return ((0.1640625*(a*a)*(d*d) - 0.109375*a*b*(d*d) + 0.0078125*(b*b)*(d*d) + 0.140625*p)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU278(double a, double b, double p, double d, double s){
	return (d*(0.02734375*a - 0.0078125*b)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU279(double a, double b, double p, double d, double s){
	return (0.001953125/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU280(double a, double b, double p, double d, double s){
	return (((a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d) + 0.5*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*p - 8.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 14.0*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 21.0 *(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 84.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 52.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 131.25*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 210.0*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 52.5*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 183.75*(a*a)*(d*d)*(p*p*p*p) - 105.0*a*b*(d*d)*(p*p*p*p) + 6.5625*(b*b)*(d*d)*(p*p*p*p) + 29.53125*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU281(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 4.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 6.0*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 42.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 42.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 105.0*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 262.5*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 105.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 367.5*(a*a*a)*(d*d)*(p*p*p) - 367.5*(a*a)*b*(d*d)*(p*p*p) + 52.5*a*(b*b)*(d*d)*(p*p*p) + 236.25*a*(p*p*p*p) - 59.0625*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU282(double a, double b, double p, double d, double s){
	return ((0.25*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d) - 4.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 7.0*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 21.0 *(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 84.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 52.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 196.875*(a*a*a*a)*(d*d*d*d)*(p*p) - 315.0*(a*a*a)*b*(d*d*d*d)*(p*p) + 78.75*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 367.5*(a*a)*(d*d)*(p*p*p) - 210.0*a*b*(d*d)*(p*p*p) + 13.125*(b*b)*(d*d)*(p*p*p) + 73.828125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU283(double a, double b, double p, double d, double s){
	return (d*((a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 7.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 7.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 35.0*(a*a*a*a*a)*(d*d*d*d)*p - 87.5*(a*a*a*a)*b*(d*d*d*d)*p + 35.0*(a*a*a)*(b*b)*(d*d*d*d)*p + 183.75*(a*a*a)*(d*d)*(p*p) - 183.75*(a*a)*b*(d*d)*(p*p) + 26.25*a*(b*b)*(d*d)*(p*p) + 157.5*a*(p*p*p) - 39.375*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU284(double a, double b, double p, double d, double s){
	return ((1.75*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 7.0*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 4.375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 32.8125*(a*a*a*a)*(d*d*d*d)*p - 52.5*(a*a*a)*b*(d*d*d*d)*p + 13.125*(a*a)*(b*b)*(d*d*d*d)*p + 91.875*(a*a)*(d*d)*(p*p) - 52.5*a*b*(d*d)*(p*p) + 3.28125*(b*b)*(d*d)*(p*p) + 24.609375*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU285(double a, double b, double p, double d, double s){
	return (d*(1.75*(a*a*a*a*a)*(d*d*d*d) - 4.375*(a*a*a*a)*b*(d*d*d*d) + 1.75*(a*a*a)*(b*b)*(d*d*d*d) + 18.375*(a*a*a)*(d*d)*p - 18.375*(a*a)*b*(d*d)*p + 2.625*a*(b*b)*(d*d)*p + 23.625*a*(p*p) - 5.90625*b*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU286(double a, double b, double p, double d, double s){
	return ((1.09375*(a*a*a*a)*(d*d*d*d) - 1.75*(a*a*a)*b*(d*d*d*d) + 0.4375*(a*a)*(b*b)*(d*d*d*d) + 6.125*(a*a)*(d*d)*p - 3.5*a*b*(d*d)*p + 0.21875*(b*b)*(d*d)*p + 2.4609375*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU287(double a, double b, double p, double d, double s){
	return (d*(0.4375*(a*a*a)*(d*d) - 0.4375*(a*a)*b*(d*d) + 0.0625*a*(b*b)*(d*d) + 1.125*a*p - 0.28125*b*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU288(double a, double b, double p, double d, double s){
	return ((0.109375*(a*a)*(d*d) - 0.0625*a*b*(d*d) + 0.00390625*(b*b)*(d*d) + 0.087890625*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU289(double a, double b, double p, double d, double s){
	return (d*(0.015625*a - 0.00390625*b)/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU2810(double a, double b, double p, double d, double s){
	return (0.0009765625/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU300(double a, double b, double p, double d, double s){
	return (b*d*(-(b*b)*(d*d) - 1.5*p)/(p*p*p))*s;
}

__device__ inline double MD_Et_GPU301(double a, double b, double p, double d, double s){
	return ((1.5*(b*b)*(d*d) + 0.75*p)/(p*p*p))*s;
}

__device__ inline double MD_Et_GPU302(double a, double b, double p, double d, double s){
	return (-0.75*b*d/(p*p*p))*s;
}

__device__ inline double MD_Et_GPU303(double a, double b, double p, double d, double s){
	return (0.125/(p*p*p))*s;
}

__device__ inline double MD_Et_GPU310(double a, double b, double p, double d, double s){
	return ((-a*(b*b*b)*(d*d*d*d) - 1.5*a*b*(d*d)*p + 1.5*(b*b)*(d*d)*p + 0.75*(p*p))/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU311(double a, double b, double p, double d, double s){
	return (d*(1.5*a*(b*b)*(d*d) + 0.75*a*p - 0.5*(b*b*b)*(d*d) - 2.25*b*p)/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU312(double a, double b, double p, double d, double s){
	return (0.75*(-a*b*(d*d) + (b*b)*(d*d) + p)/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU313(double a, double b, double p, double d, double s){
	return (d*(0.125*a - 0.375*b)/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU314(double a, double b, double p, double d, double s){
	return (0.0625/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU320(double a, double b, double p, double d, double s){
	return (d*(-(a*a)*(b*b*b)*(d*d*d*d) - 1.5*(a*a)*b*(d*d)*p + 3.0*a*(b*b)*(d*d)*p + 1.5*a*(p*p) - 0.5*(b*b*b)*(d*d)*p - 2.25*b*(p*p))/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU321(double a, double b, double p, double d, double s){
	return ((1.5*(a*a)*(b*b)*(d*d*d*d) + 0.75*(a*a)*(d*d)*p - a*(b*b*b)*(d*d*d*d) - 4.5*a*b*(d*d)*p + 2.25*(b*b)*(d*d)*p + 1.875*(p*p))/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU322(double a, double b, double p, double d, double s){
	return (d*(-0.75*(a*a)*b*(d*d) + 1.5*a*(b*b)*(d*d) + 1.5*a*p - 0.25*(b*b*b)*(d*d) - 2.25*b*p)/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU323(double a, double b, double p, double d, double s){
	return ((0.125*(a*a)*(d*d) - 0.75*a*b*(d*d) + 0.375*(b*b)*(d*d) + 0.625*p)/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU324(double a, double b, double p, double d, double s){
	return (d*(0.125*a - 0.1875*b)/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU325(double a, double b, double p, double d, double s){
	return (0.03125/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU330(double a, double b, double p, double d, double s){
	return ((-(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 1.5*(a*a*a)*b*(d*d*d*d)*p + 4.5*(a*a)*(b*b)*(d*d*d*d)*p + 2.25*(a*a)*(d*d)*(p*p) - 1.5*a*(b*b*b)*(d*d*d*d)*p - 6.75*a*b*(d*d)*(p*p) + 2.25*(b*b)*(d*d)*(p*p) + 1.875*(p*p*p))/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU331(double a, double b, double p, double d, double s){
	return (d*(1.5*(a*a*a)*(b*b)*(d*d*d*d) + 0.75*(a*a*a)*(d*d)*p - 1.5*(a*a)*(b*b*b)*(d*d*d*d) - 6.75*(a*a)*b*(d*d)*p + 6.75*a*(b*b)*(d*d)*p + 5.625*a*(p*p) - 0.75*(b*b*b)*(d*d)*p - 5.625*b*(p*p))/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU332(double a, double b, double p, double d, double s){
	return ((-0.75*(a*a*a)*b*(d*d*d*d) + 2.25*(a*a)*(b*b)*(d*d*d*d) + 2.25*(a*a)*(d*d)*p - 0.75*a*(b*b*b)*(d*d*d*d) - 6.75*a*b*(d*d)*p + 2.25*(b*b)*(d*d)*p + 2.8125*(p*p))/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU333(double a, double b, double p, double d, double s){
	return (d*(0.125*(a*a*a)*(d*d) - 1.125*(a*a)*b*(d*d) + 1.125*a*(b*b)*(d*d) + 1.875*a*p - 0.125*(b*b*b)*(d*d) - 1.875*b*p)/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU334(double a, double b, double p, double d, double s){
	return ((0.1875*(a*a)*(d*d) - 0.5625*a*b*(d*d) + 0.1875*(b*b)*(d*d) + 0.46875*p)/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU335(double a, double b, double p, double d, double s){
	return (0.09375*d*(a - b)/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU336(double a, double b, double p, double d, double s){
	return (0.015625/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU340(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 1.5*(a*a*a*a)*b*(d*d*d*d)*p + 6.0*(a*a*a)*(b*b)*(d*d*d*d)*p + 3.0*(a*a*a)*(d*d)*(p*p) - 3.0*(a*a)*(b*b*b)*(d*d*d*d)*p - 13.5*(a*a)*b*(d*d)*(p*p) + 9.0*a*(b*b)*(d*d)*(p*p) + 7.5*a*(p*p*p) - 0.75*(b*b*b)*(d*d)*(p*p) - 5.625*b*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU341(double a, double b, double p, double d, double s){
	return ((1.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.75*(a*a*a*a)*(d*d*d*d)*p - 2.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 9.0*(a*a*a)*b*(d*d*d*d)*p + 13.5*(a*a)*(b*b)*(d*d*d*d)*p + 11.25*(a*a)*(d*d)*(p*p) - 3.0*a*(b*b*b)*(d*d*d*d)*p - 22.5*a*b*(d*d)*(p*p) + 5.625*(b*b)*(d*d)*(p*p) + 6.5625*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU342(double a, double b, double p, double d, double s){
	return (d*(-0.75*(a*a*a*a)*b*(d*d*d*d) + 3.0*(a*a*a)*(b*b)*(d*d*d*d) + 3.0*(a*a*a)*(d*d)*p - 1.5*(a*a)*(b*b*b)*(d*d*d*d) - 13.5*(a*a)*b*(d*d)*p + 9.0*a*(b*b)*(d*d)*p + 11.25*a*(p*p) - 0.75*(b*b*b)*(d*d)*p - 8.4375*b*(p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU343(double a, double b, double p, double d, double s){
	return ((0.125*(a*a*a*a)*(d*d*d*d) - 1.5*(a*a*a)*b*(d*d*d*d) + 2.25*(a*a)*(b*b)*(d*d*d*d) + 3.75*(a*a)*(d*d)*p - 0.5*a*(b*b*b)*(d*d*d*d) - 7.5*a*b*(d*d)*p + 1.875*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU344(double a, double b, double p, double d, double s){
	return (d*(0.25*(a*a*a)*(d*d) - 1.125*(a*a)*b*(d*d) + 0.75*a*(b*b)*(d*d) + 1.875*a*p - 0.0625*(b*b*b)*(d*d) - 1.40625*b*p)/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU345(double a, double b, double p, double d, double s){
	return ((0.1875*(a*a)*(d*d) - 0.375*a*b*(d*d) + 0.09375*(b*b)*(d*d) + 0.328125*p)/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU346(double a, double b, double p, double d, double s){
	return (d*(0.0625*a - 0.046875*b)/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU347(double a, double b, double p, double d, double s){
	return (0.0078125/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU350(double a, double b, double p, double d, double s){
	return ((-(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 1.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 7.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 3.75*(a*a*a*a)*(d*d*d*d)*(p*p) - 5.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 22.5*(a*a*a)*b*(d*d*d*d)*(p*p) + 22.5*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 18.75*(a*a)*(d*d)*(p*p*p) - 3.75*a*(b*b*b)*(d*d*d*d)*(p*p) - 28.125*a*b*(d*d)*(p*p*p) + 5.625*(b*b)*(d*d)*(p*p*p) + 6.5625*(p*p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU351(double a, double b, double p, double d, double s){
	return (d*(1.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.75*(a*a*a*a*a)*(d*d*d*d)*p - 2.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 11.25*(a*a*a*a)*b*(d*d*d*d)*p + 22.5*(a*a*a)*(b*b)*(d*d*d*d)*p + 18.75*(a*a*a)*(d*d)*(p*p) - 7.5*(a*a)*(b*b*b)*(d*d*d*d)*p - 56.25*(a*a)*b*(d*d)*(p*p) + 28.125*a*(b*b)*(d*d)*(p*p) + 32.8125*a*(p*p*p) - 1.875*(b*b*b)*(d*d)*(p*p) - 19.6875*b*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU352(double a, double b, double p, double d, double s){
	return ((-0.75*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 3.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 3.75*(a*a*a*a)*(d*d*d*d)*p - 2.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 22.5*(a*a*a)*b*(d*d*d*d)*p + 22.5*(a*a)*(b*b)*(d*d*d*d)*p + 28.125*(a*a)*(d*d)*(p*p) - 3.75*a*(b*b*b)*(d*d*d*d)*p - 42.1875*a*b*(d*d)*(p*p) + 8.4375*(b*b)*(d*d)*(p*p) + 13.125*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU353(double a, double b, double p, double d, double s){
	return (d*(0.125*(a*a*a*a*a)*(d*d*d*d) - 1.875*(a*a*a*a)*b*(d*d*d*d) + 3.75*(a*a*a)*(b*b)*(d*d*d*d) + 6.25*(a*a*a)*(d*d)*p - 1.25*(a*a)*(b*b*b)*(d*d*d*d) - 18.75*(a*a)*b*(d*d)*p + 9.375*a*(b*b)*(d*d)*p + 16.40625*a*(p*p) - 0.625*(b*b*b)*(d*d)*p - 9.84375*b*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU354(double a, double b, double p, double d, double s){
	return ((0.3125*(a*a*a*a)*(d*d*d*d) - 1.875*(a*a*a)*b*(d*d*d*d) + 1.875*(a*a)*(b*b)*(d*d*d*d) + 4.6875*(a*a)*(d*d)*p - 0.3125*a*(b*b*b)*(d*d*d*d) - 7.03125*a*b*(d*d)*p + 1.40625*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU355(double a, double b, double p, double d, double s){
	return (d*(0.3125*(a*a*a)*(d*d) - 0.9375*(a*a)*b*(d*d) + 0.46875*a*(b*b)*(d*d) + 1.640625*a*p - 0.03125*(b*b*b)*(d*d) - 0.984375*b*p)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU356(double a, double b, double p, double d, double s){
	return ((0.15625*(a*a)*(d*d) - 0.234375*a*b*(d*d) + 0.046875*(b*b)*(d*d) + 0.21875*p)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU357(double a, double b, double p, double d, double s){
	return (d*(0.0390625*a - 0.0234375*b)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU358(double a, double b, double p, double d, double s){
	return (0.00390625/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU360(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 1.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 9.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 4.5*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 7.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 33.75*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 45.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 37.5*(a*a*a)*(d*d)*(p*p*p) - 11.25*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 84.375*(a*a)*b*(d*d)*(p*p*p) + 33.75*a*(b*b)*(d*d)*(p*p*p) + 39.375*a*(p*p*p*p) - 1.875*(b*b*b)*(d*d)*(p*p*p) - 19.6875*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU361(double a, double b, double p, double d, double s){
	return ((1.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.75*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 3.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 13.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 33.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 28.125*(a*a*a*a)*(d*d*d*d)*(p*p) - 15.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 112.5*(a*a*a)*b*(d*d*d*d)*(p*p) + 84.375*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 98.4375*(a*a)*(d*d)*(p*p*p) - 11.25*a*(b*b*b)*(d*d*d*d)*(p*p) - 118.125*a*b*(d*d)*(p*p*p) + 19.6875*(b*b)*(d*d)*(p*p*p) + 29.53125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU362(double a, double b, double p, double d, double s){
	return (d*(-0.75*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 4.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 4.5*(a*a*a*a*a)*(d*d*d*d)*p - 3.75*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 33.75*(a*a*a*a)*b*(d*d*d*d)*p + 45.0*(a*a*a)*(b*b)*(d*d*d*d)*p + 56.25*(a*a*a)*(d*d)*(p*p) - 11.25*(a*a)*(b*b*b)*(d*d*d*d)*p - 126.5625*(a*a)*b*(d*d)*(p*p) + 50.625*a*(b*b)*(d*d)*(p*p) + 78.75*a*(p*p*p) - 2.8125*(b*b*b)*(d*d)*(p*p) - 39.375*b*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU363(double a, double b, double p, double d, double s){
	return ((0.125*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 2.25*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 5.625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 9.375*(a*a*a*a)*(d*d*d*d)*p - 2.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 37.5*(a*a*a)*b*(d*d*d*d)*p + 28.125*(a*a)*(b*b)*(d*d*d*d)*p + 49.21875*(a*a)*(d*d)*(p*p) - 3.75*a*(b*b*b)*(d*d*d*d)*p - 59.0625*a*b*(d*d)*(p*p) + 9.84375*(b*b)*(d*d)*(p*p) + 19.6875*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU364(double a, double b, double p, double d, double s){
	return (d*(0.375*(a*a*a*a*a)*(d*d*d*d) - 2.8125*(a*a*a*a)*b*(d*d*d*d) + 3.75*(a*a*a)*(b*b)*(d*d*d*d) + 9.375*(a*a*a)*(d*d)*p - 0.9375*(a*a)*(b*b*b)*(d*d*d*d) - 21.09375*(a*a)*b*(d*d)*p + 8.4375*a*(b*b)*(d*d)*p + 19.6875*a*(p*p) - 0.46875*(b*b*b)*(d*d)*p - 9.84375*b*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU365(double a, double b, double p, double d, double s){
	return ((0.46875*(a*a*a*a)*(d*d*d*d) - 1.875*(a*a*a)*b*(d*d*d*d) + 1.40625*(a*a)*(b*b)*(d*d*d*d) + 4.921875*(a*a)*(d*d)*p - 0.1875*a*(b*b*b)*(d*d*d*d) - 5.90625*a*b*(d*d)*p + 0.984375*(b*b)*(d*d)*p + 2.953125*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU366(double a, double b, double p, double d, double s){
	return (d*(0.3125*(a*a*a)*(d*d) - 0.703125*(a*a)*b*(d*d) + 0.28125*a*(b*b)*(d*d) + 1.3125*a*p - 0.015625*(b*b*b)*(d*d) - 0.65625*b*p)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU367(double a, double b, double p, double d, double s){
	return ((0.1171875*(a*a)*(d*d) - 0.140625*a*b*(d*d) + 0.0234375*(b*b)*(d*d) + 0.140625*p)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU368(double a, double b, double p, double d, double s){
	return (d*(0.0234375*a - 0.01171875*b)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU369(double a, double b, double p, double d, double s){
	return (0.001953125/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU370(double a, double b, double p, double d, double s){
	return ((-(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 1.5*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 10.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 5.25*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 10.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 47.25*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 78.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 65.625*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 26.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 196.875*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 118.125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 137.8125*(a*a)*(d*d)*(p*p*p*p) - 13.125*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 137.8125*a*b*(d*d)*(p*p*p*p) + 19.6875*(b*b)*(d*d)*(p*p*p*p) + 29.53125*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU371(double a, double b, double p, double d, double s){
	return (d*(1.5*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.75*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 3.5*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 15.75*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 47.25*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 39.375*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 26.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 196.875*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 196.875*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 229.6875*(a*a*a)*(d*d)*(p*p*p) - 39.375*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 413.4375*(a*a)*b*(d*d)*(p*p*p) + 137.8125*a*(b*b)*(d*d)*(p*p*p) + 206.71875*a*(p*p*p*p) - 6.5625*(b*b*b)*(d*d)*(p*p*p) - 88.59375*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU372(double a, double b, double p, double d, double s){
	return ((-0.75*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 5.25*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 5.25*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 5.25*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 47.25*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 78.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 98.4375*(a*a*a*a)*(d*d*d*d)*(p*p) - 26.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 295.3125*(a*a*a)*b*(d*d*d*d)*(p*p) + 177.1875*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 275.625*(a*a)*(d*d)*(p*p*p) - 19.6875*a*(b*b*b)*(d*d*d*d)*(p*p) - 275.625*a*b*(d*d)*(p*p*p) + 39.375*(b*b)*(d*d)*(p*p*p) + 73.828125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU373(double a, double b, double p, double d, double s){
	return (d*(0.125*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 2.625*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 7.875*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 13.125*(a*a*a*a*a)*(d*d*d*d)*p - 4.375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 65.625*(a*a*a*a)*b*(d*d*d*d)*p + 65.625*(a*a*a)*(b*b)*(d*d*d*d)*p + 114.84375*(a*a*a)*(d*d)*(p*p) - 13.125*(a*a)*(b*b*b)*(d*d*d*d)*p - 206.71875*(a*a)*b*(d*d)*(p*p) + 68.90625*a*(b*b)*(d*d)*(p*p) + 137.8125*a*(p*p*p) - 3.28125*(b*b*b)*(d*d)*(p*p) - 59.0625*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU374(double a, double b, double p, double d, double s){
	return ((0.4375*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 3.9375*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 6.5625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 16.40625*(a*a*a*a)*(d*d*d*d)*p - 2.1875*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 49.21875*(a*a*a)*b*(d*d*d*d)*p + 29.53125*(a*a)*(b*b)*(d*d*d*d)*p + 68.90625*(a*a)*(d*d)*(p*p) - 3.28125*a*(b*b*b)*(d*d*d*d)*p - 68.90625*a*b*(d*d)*(p*p) + 9.84375*(b*b)*(d*d)*(p*p) + 24.609375*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU375(double a, double b, double p, double d, double s){
	return (d*(0.65625*(a*a*a*a*a)*(d*d*d*d) - 3.28125*(a*a*a*a)*b*(d*d*d*d) + 3.28125*(a*a*a)*(b*b)*(d*d*d*d) + 11.484375*(a*a*a)*(d*d)*p - 0.65625*(a*a)*(b*b*b)*(d*d*d*d) - 20.671875*(a*a)*b*(d*d)*p + 6.890625*a*(b*b)*(d*d)*p + 20.671875*a*(p*p) - 0.328125*(b*b*b)*(d*d)*p - 8.859375*b*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU376(double a, double b, double p, double d, double s){
	return ((0.546875*(a*a*a*a)*(d*d*d*d) - 1.640625*(a*a*a)*b*(d*d*d*d) + 0.984375*(a*a)*(b*b)*(d*d*d*d) + 4.59375*(a*a)*(d*d)*p - 0.109375*a*(b*b*b)*(d*d*d*d) - 4.59375*a*b*(d*d)*p + 0.65625*(b*b)*(d*d)*p + 2.4609375*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU377(double a, double b, double p, double d, double s){
	return (d*(0.2734375*(a*a*a)*(d*d) - 0.4921875*(a*a)*b*(d*d) + 0.1640625*a*(b*b)*(d*d) + 0.984375*a*p - 0.0078125*(b*b*b)*(d*d) - 0.421875*b*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU378(double a, double b, double p, double d, double s){
	return ((0.08203125*(a*a)*(d*d) - 0.08203125*a*b*(d*d) + 0.01171875*(b*b)*(d*d) + 0.087890625*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU379(double a, double b, double p, double d, double s){
	return (d*(0.013671875*a - 0.005859375*b)/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU3710(double a, double b, double p, double d, double s){
	return (0.0009765625/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU380(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 1.5*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 12.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 6.0*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 14.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 63.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 126.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 105.0*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 52.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 393.75*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 315.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 367.5*(a*a*a)*(d*d)*(p*p*p*p) - 52.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 551.25*(a*a)*b*(d*d)*(p*p*p*p) + 157.5*a*(b*b)*(d*d)*(p*p*p*p) + 236.25*a*(p*p*p*p*p) - 6.5625*(b*b*b)*(d*d)*(p*p*p*p) - 88.59375*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU381(double a, double b, double p, double d, double s){
	return ((1.5*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d) + 0.75*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*p - 4.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 18.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 63.0*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 52.5*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 42.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 315.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 393.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 459.375*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 105.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 1102.5*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 551.25*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 826.875*(a*a)*(d*d)*(p*p*p*p) - 52.5*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 708.75*a*b*(d*d)*(p*p*p*p) + 88.59375*(b*b)*(d*d)*(p*p*p*p) + 162.421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU382(double a, double b, double p, double d, double s){
	return (d*(-0.75*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 6.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 6.0*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 7.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 63.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 126.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 157.5*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 52.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 590.625*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 472.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 735.0*(a*a*a)*(d*d)*(p*p*p) - 78.75*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 1102.5*(a*a)*b*(d*d)*(p*p*p) + 315.0*a*(b*b)*(d*d)*(p*p*p) + 590.625*a*(p*p*p*p) - 13.125*(b*b*b)*(d*d)*(p*p*p) - 221.484375*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU383(double a, double b, double p, double d, double s){
	return ((0.125*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d) - 3.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 10.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 17.5*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 7.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 105.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 131.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 229.6875*(a*a*a*a)*(d*d*d*d)*(p*p) - 35.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 551.25*(a*a*a)*b*(d*d*d*d)*(p*p) + 275.625*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 551.25*(a*a)*(d*d)*(p*p*p) - 26.25*a*(b*b*b)*(d*d*d*d)*(p*p) - 472.5*a*b*(d*d)*(p*p*p) + 59.0625*(b*b)*(d*d)*(p*p*p) + 135.3515625*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU384(double a, double b, double p, double d, double s){
	return (d*(0.5*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 5.25*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 10.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 26.25*(a*a*a*a*a)*(d*d*d*d)*p - 4.375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 98.4375*(a*a*a*a)*b*(d*d*d*d)*p + 78.75*(a*a*a)*(b*b)*(d*d*d*d)*p + 183.75*(a*a*a)*(d*d)*(p*p) - 13.125*(a*a)*(b*b*b)*(d*d*d*d)*p - 275.625*(a*a)*b*(d*d)*(p*p) + 78.75*a*(b*b)*(d*d)*(p*p) + 196.875*a*(p*p*p) - 3.28125*(b*b*b)*(d*d)*(p*p) - 73.828125*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU385(double a, double b, double p, double d, double s){
	return ((0.875*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 5.25*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 6.5625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 22.96875*(a*a*a*a)*(d*d*d*d)*p - 1.75*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 55.125*(a*a*a)*b*(d*d*d*d)*p + 27.5625*(a*a)*(b*b)*(d*d*d*d)*p + 82.6875*(a*a)*(d*d)*(p*p) - 2.625*a*(b*b*b)*(d*d*d*d)*p - 70.875*a*b*(d*d)*(p*p) + 8.859375*(b*b)*(d*d)*(p*p) + 27.0703125*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU386(double a, double b, double p, double d, double s){
	return (d*(0.875*(a*a*a*a*a)*(d*d*d*d) - 3.28125*(a*a*a*a)*b*(d*d*d*d) + 2.625*(a*a*a)*(b*b)*(d*d*d*d) + 12.25*(a*a*a)*(d*d)*p - 0.4375*(a*a)*(b*b*b)*(d*d*d*d) - 18.375*(a*a)*b*(d*d)*p + 5.25*a*(b*b)*(d*d)*p + 19.6875*a*(p*p) - 0.21875*(b*b*b)*(d*d)*p - 7.3828125*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU387(double a, double b, double p, double d, double s){
	return ((0.546875*(a*a*a*a)*(d*d*d*d) - 1.3125*(a*a*a)*b*(d*d*d*d) + 0.65625*(a*a)*(b*b)*(d*d*d*d) + 3.9375*(a*a)*(d*d)*p - 0.0625*a*(b*b*b)*(d*d*d*d) - 3.375*a*b*(d*d)*p + 0.421875*(b*b)*(d*d)*p + 1.93359375*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU388(double a, double b, double p, double d, double s){
	return (d*(0.21875*(a*a*a)*(d*d) - 0.328125*(a*a)*b*(d*d) + 0.09375*a*(b*b)*(d*d) + 0.703125*a*p - 0.00390625*(b*b*b)*(d*d) - 0.263671875*b*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU389(double a, double b, double p, double d, double s){
	return ((0.0546875*(a*a)*(d*d) - 0.046875*a*b*(d*d) + 0.005859375*(b*b)*(d*d) + 0.0537109375*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU3810(double a, double b, double p, double d, double s){
	return (d*(0.0078125*a - 0.0029296875*b)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU3811(double a, double b, double p, double d, double s){
	return (0.00048828125/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU400(double a, double b, double p, double d, double s){
	return (((b*b*b*b)*(d*d*d*d) + 3.0*(b*b)*(d*d)*p + 0.75*(p*p))/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU401(double a, double b, double p, double d, double s){
	return (b*d*(-2.0*(b*b)*(d*d) - 3.0*p)/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU402(double a, double b, double p, double d, double s){
	return ((1.5*(b*b)*(d*d) + 0.75*p)/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU403(double a, double b, double p, double d, double s){
	return (-0.5*b*d/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU404(double a, double b, double p, double d, double s){
	return (0.0625/(p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU410(double a, double b, double p, double d, double s){
	return (d*(a*(b*b*b*b)*(d*d*d*d) + 3.0*a*(b*b)*(d*d)*p + 0.75*a*(p*p) - 2.0*(b*b*b)*(d*d)*p - 3.0*b*(p*p))/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU411(double a, double b, double p, double d, double s){
	return ((-2.0*a*(b*b*b)*(d*d*d*d) - 3.0*a*b*(d*d)*p + 0.5*(b*b*b*b)*(d*d*d*d) + 4.5*(b*b)*(d*d)*p + 1.875*(p*p))/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU412(double a, double b, double p, double d, double s){
	return (d*(1.5*a*(b*b)*(d*d) + 0.75*a*p - (b*b*b)*(d*d) - 3.0*b*p)/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU413(double a, double b, double p, double d, double s){
	return ((-0.5*a*b*(d*d) + 0.75*(b*b)*(d*d) + 0.625*p)/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU414(double a, double b, double p, double d, double s){
	return (d*(0.0625*a - 0.25*b)/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU415(double a, double b, double p, double d, double s){
	return (0.03125/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU420(double a, double b, double p, double d, double s){
	return (((a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 3.0*(a*a)*(b*b)*(d*d*d*d)*p + 0.75*(a*a)*(d*d)*(p*p) - 4.0*a*(b*b*b)*(d*d*d*d)*p - 6.0*a*b*(d*d)*(p*p) + 0.5*(b*b*b*b)*(d*d*d*d)*p + 4.5*(b*b)*(d*d)*(p*p) + 1.875*(p*p*p))/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU421(double a, double b, double p, double d, double s){
	return (d*(-2.0*(a*a)*(b*b*b)*(d*d*d*d) - 3.0*(a*a)*b*(d*d)*p + a*(b*b*b*b)*(d*d*d*d) + 9.0*a*(b*b)*(d*d)*p + 3.75*a*(p*p) - 3.0*(b*b*b)*(d*d)*p - 7.5*b*(p*p))/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU422(double a, double b, double p, double d, double s){
	return ((1.5*(a*a)*(b*b)*(d*d*d*d) + 0.75*(a*a)*(d*d)*p - 2.0*a*(b*b*b)*(d*d*d*d) - 6.0*a*b*(d*d)*p + 0.25*(b*b*b*b)*(d*d*d*d) + 4.5*(b*b)*(d*d)*p + 2.8125*(p*p))/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU423(double a, double b, double p, double d, double s){
	return (d*(-0.5*(a*a)*b*(d*d) + 1.5*a*(b*b)*(d*d) + 1.25*a*p - 0.5*(b*b*b)*(d*d) - 2.5*b*p)/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU424(double a, double b, double p, double d, double s){
	return ((0.0625*(a*a)*(d*d) - 0.5*a*b*(d*d) + 0.375*(b*b)*(d*d) + 0.46875*p)/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU425(double a, double b, double p, double d, double s){
	return (d*(0.0625*a - 0.125*b)/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU426(double a, double b, double p, double d, double s){
	return (0.015625/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU430(double a, double b, double p, double d, double s){
	return (d*((a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 3.0*(a*a*a)*(b*b)*(d*d*d*d)*p + 0.75*(a*a*a)*(d*d)*(p*p) - 6.0*(a*a)*(b*b*b)*(d*d*d*d)*p - 9.0*(a*a)*b*(d*d)*(p*p) + 1.5*a*(b*b*b*b)*(d*d*d*d)*p + 13.5*a*(b*b)*(d*d)*(p*p) + 5.625*a*(p*p*p) - 3.0*(b*b*b)*(d*d)*(p*p) - 7.5*b*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU431(double a, double b, double p, double d, double s){
	return ((-2.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 3.0*(a*a*a)*b*(d*d*d*d)*p + 1.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 13.5*(a*a)*(b*b)*(d*d*d*d)*p + 5.625*(a*a)*(d*d)*(p*p) - 9.0*a*(b*b*b)*(d*d*d*d)*p - 22.5*a*b*(d*d)*(p*p) + 0.75*(b*b*b*b)*(d*d*d*d)*p + 11.25*(b*b)*(d*d)*(p*p) + 6.5625*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU432(double a, double b, double p, double d, double s){
	return (d*(1.5*(a*a*a)*(b*b)*(d*d*d*d) + 0.75*(a*a*a)*(d*d)*p - 3.0*(a*a)*(b*b*b)*(d*d*d*d) - 9.0*(a*a)*b*(d*d)*p + 0.75*a*(b*b*b*b)*(d*d*d*d) + 13.5*a*(b*b)*(d*d)*p + 8.4375*a*(p*p) - 3.0*(b*b*b)*(d*d)*p - 11.25*b*(p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU433(double a, double b, double p, double d, double s){
	return ((-0.5*(a*a*a)*b*(d*d*d*d) + 2.25*(a*a)*(b*b)*(d*d*d*d) + 1.875*(a*a)*(d*d)*p - 1.5*a*(b*b*b)*(d*d*d*d) - 7.5*a*b*(d*d)*p + 0.125*(b*b*b*b)*(d*d*d*d) + 3.75*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU434(double a, double b, double p, double d, double s){
	return (d*(0.0625*(a*a*a)*(d*d) - 0.75*(a*a)*b*(d*d) + 1.125*a*(b*b)*(d*d) + 1.40625*a*p - 0.25*(b*b*b)*(d*d) - 1.875*b*p)/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU435(double a, double b, double p, double d, double s){
	return ((0.09375*(a*a)*(d*d) - 0.375*a*b*(d*d) + 0.1875*(b*b)*(d*d) + 0.328125*p)/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU436(double a, double b, double p, double d, double s){
	return (d*(0.046875*a - 0.0625*b)/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU437(double a, double b, double p, double d, double s){
	return (0.0078125/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU440(double a, double b, double p, double d, double s){
	return (((a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 3.0*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 0.75*(a*a*a*a)*(d*d*d*d)*(p*p) - 8.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 12.0*(a*a*a)*b*(d*d*d*d)*(p*p) + 3.0*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 27.0*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 11.25*(a*a)*(d*d)*(p*p*p) - 12.0*a*(b*b*b)*(d*d*d*d)*(p*p) - 30.0*a*b*(d*d)*(p*p*p) + 0.75*(b*b*b*b)*(d*d*d*d)*(p*p) + 11.25*(b*b)*(d*d)*(p*p*p) + 6.5625*(p*p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU441(double a, double b, double p, double d, double s){
	return (d*(-2.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 3.0*(a*a*a*a)*b*(d*d*d*d)*p + 2.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 18.0*(a*a*a)*(b*b)*(d*d*d*d)*p + 7.5*(a*a*a)*(d*d)*(p*p) - 18.0*(a*a)*(b*b*b)*(d*d*d*d)*p - 45.0*(a*a)*b*(d*d)*(p*p) + 3.0*a*(b*b*b*b)*(d*d*d*d)*p + 45.0*a*(b*b)*(d*d)*(p*p) + 26.25*a*(p*p*p) - 7.5*(b*b*b)*(d*d)*(p*p) - 26.25*b*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU442(double a, double b, double p, double d, double s){
	return ((1.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.75*(a*a*a*a)*(d*d*d*d)*p - 4.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 12.0*(a*a*a)*b*(d*d*d*d)*p + 1.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 27.0*(a*a)*(b*b)*(d*d*d*d)*p + 16.875*(a*a)*(d*d)*(p*p) - 12.0*a*(b*b*b)*(d*d*d*d)*p - 45.0*a*b*(d*d)*(p*p) + 0.75*(b*b*b*b)*(d*d*d*d)*p + 16.875*(b*b)*(d*d)*(p*p) + 13.125*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU443(double a, double b, double p, double d, double s){
	return (d*(-0.5*(a*a*a*a)*b*(d*d*d*d) + 3.0*(a*a*a)*(b*b)*(d*d*d*d) + 2.5*(a*a*a)*(d*d)*p - 3.0*(a*a)*(b*b*b)*(d*d*d*d) - 15.0*(a*a)*b*(d*d)*p + 0.5*a*(b*b*b*b)*(d*d*d*d) + 15.0*a*(b*b)*(d*d)*p + 13.125*a*(p*p) - 2.5*(b*b*b)*(d*d)*p - 13.125*b*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU444(double a, double b, double p, double d, double s){
	return ((0.0625*(a*a*a*a)*(d*d*d*d) - (a*a*a)*b*(d*d*d*d) + 2.25*(a*a)*(b*b)*(d*d*d*d) + 2.8125*(a*a)*(d*d)*p - a*(b*b*b)*(d*d*d*d) - 7.5*a*b*(d*d)*p + 0.0625*(b*b*b*b)*(d*d*d*d) + 2.8125*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU445(double a, double b, double p, double d, double s){
	return (d*(0.125*(a*a*a)*(d*d) - 0.75*(a*a)*b*(d*d) + 0.75*a*(b*b)*(d*d) + 1.3125*a*p - 0.125*(b*b*b)*(d*d) - 1.3125*b*p)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU446(double a, double b, double p, double d, double s){
	return ((0.09375*(a*a)*(d*d) - 0.25*a*b*(d*d) + 0.09375*(b*b)*(d*d) + 0.21875*p)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU447(double a, double b, double p, double d, double s){
	return (0.03125*d*(a - b)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU448(double a, double b, double p, double d, double s){
	return (0.00390625/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU450(double a, double b, double p, double d, double s){
	return (d*((a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 3.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 0.75*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 10.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 15.0*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 5.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 45.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 18.75*(a*a*a)*(d*d)*(p*p*p) - 30.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 75.0*(a*a)*b*(d*d)*(p*p*p) + 3.75*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 56.25*a*(b*b)*(d*d)*(p*p*p) + 32.8125*a*(p*p*p*p) - 7.5*(b*b*b)*(d*d)*(p*p*p) - 26.25*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU451(double a, double b, double p, double d, double s){
	return ((-2.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 3.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 2.5*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 22.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 9.375*(a*a*a*a)*(d*d*d*d)*(p*p) - 30.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 75.0*(a*a*a)*b*(d*d*d*d)*(p*p) + 7.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 112.5*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 65.625*(a*a)*(d*d)*(p*p*p) - 37.5*a*(b*b*b)*(d*d*d*d)*(p*p) - 131.25*a*b*(d*d)*(p*p*p) + 1.875*(b*b*b*b)*(d*d*d*d)*(p*p) + 39.375*(b*b)*(d*d)*(p*p*p) + 29.53125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU452(double a, double b, double p, double d, double s){
	return (d*(1.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.75*(a*a*a*a*a)*(d*d*d*d)*p - 5.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 15.0*(a*a*a*a)*b*(d*d*d*d)*p + 2.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 45.0*(a*a*a)*(b*b)*(d*d*d*d)*p + 28.125*(a*a*a)*(d*d)*(p*p) - 30.0*(a*a)*(b*b*b)*(d*d*d*d)*p - 112.5*(a*a)*b*(d*d)*(p*p) + 3.75*a*(b*b*b*b)*(d*d*d*d)*p + 84.375*a*(b*b)*(d*d)*(p*p) + 65.625*a*(p*p*p) - 11.25*(b*b*b)*(d*d)*(p*p) - 52.5*b*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU453(double a, double b, double p, double d, double s){
	return ((-0.5*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 3.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 3.125*(a*a*a*a)*(d*d*d*d)*p - 5.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 25.0*(a*a*a)*b*(d*d*d*d)*p + 1.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 37.5*(a*a)*(b*b)*(d*d*d*d)*p + 32.8125*(a*a)*(d*d)*(p*p) - 12.5*a*(b*b*b)*(d*d*d*d)*p - 65.625*a*b*(d*d)*(p*p) + 0.625*(b*b*b*b)*(d*d*d*d)*p + 19.6875*(b*b)*(d*d)*(p*p) + 19.6875*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU454(double a, double b, double p, double d, double s){
	return (d*(0.0625*(a*a*a*a*a)*(d*d*d*d) - 1.25*(a*a*a*a)*b*(d*d*d*d) + 3.75*(a*a*a)*(b*b)*(d*d*d*d) + 4.6875*(a*a*a)*(d*d)*p - 2.5*(a*a)*(b*b*b)*(d*d*d*d) - 18.75*(a*a)*b*(d*d)*p + 0.3125*a*(b*b*b*b)*(d*d*d*d) + 14.0625*a*(b*b)*(d*d)*p + 16.40625*a*(p*p) - 1.875*(b*b*b)*(d*d)*p - 13.125*b*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU455(double a, double b, double p, double d, double s){
	return ((0.15625*(a*a*a*a)*(d*d*d*d) - 1.25*(a*a*a)*b*(d*d*d*d) + 1.875*(a*a)*(b*b)*(d*d*d*d) + 3.28125*(a*a)*(d*d)*p - 0.625*a*(b*b*b)*(d*d*d*d) - 6.5625*a*b*(d*d)*p + 0.03125*(b*b*b*b)*(d*d*d*d) + 1.96875*(b*b)*(d*d)*p + 2.953125*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU456(double a, double b, double p, double d, double s){
	return (d*(0.15625*(a*a*a)*(d*d) - 0.625*(a*a)*b*(d*d) + 0.46875*a*(b*b)*(d*d) + 1.09375*a*p - 0.0625*(b*b*b)*(d*d) - 0.875*b*p)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU457(double a, double b, double p, double d, double s){
	return ((0.078125*(a*a)*(d*d) - 0.15625*a*b*(d*d) + 0.046875*(b*b)*(d*d) + 0.140625*p)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU458(double a, double b, double p, double d, double s){
	return (d*(0.01953125*a - 0.015625*b)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU459(double a, double b, double p, double d, double s){
	return (0.001953125/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU460(double a, double b, double p, double d, double s){
	return (((a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 3.0*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 0.75*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 12.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 18.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 7.5*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 67.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 28.125*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 60.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 150.0*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 11.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 168.75*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 98.4375*(a*a)*(d*d)*(p*p*p*p) - 45.0*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 157.5*a*b*(d*d)*(p*p*p*p) + 1.875*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 39.375*(b*b)*(d*d)*(p*p*p*p) + 29.53125*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU461(double a, double b, double p, double d, double s){
	return (d*(-2.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 3.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 3.0*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 27.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 11.25*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 45.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 112.5*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 15.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 225.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 131.25*(a*a*a)*(d*d)*(p*p*p) - 112.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 393.75*(a*a)*b*(d*d)*(p*p*p) + 11.25*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 236.25*a*(b*b)*(d*d)*(p*p*p) + 177.1875*a*(p*p*p*p) - 26.25*(b*b*b)*(d*d)*(p*p*p) - 118.125*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU462(double a, double b, double p, double d, double s){
	return ((1.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.75*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 6.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 18.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 3.75*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 67.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 42.1875*(a*a*a*a)*(d*d*d*d)*(p*p) - 60.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 225.0*(a*a*a)*b*(d*d*d*d)*(p*p) + 11.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 253.125*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 196.875*(a*a)*(d*d)*(p*p*p) - 67.5*a*(b*b*b)*(d*d*d*d)*(p*p) - 315.0*a*b*(d*d)*(p*p*p) + 2.8125*(b*b*b*b)*(d*d*d*d)*(p*p) + 78.75*(b*b)*(d*d)*(p*p*p) + 73.828125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU463(double a, double b, double p, double d, double s){
	return (d*(-0.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 4.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 3.75*(a*a*a*a*a)*(d*d*d*d)*p - 7.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 37.5*(a*a*a*a)*b*(d*d*d*d)*p + 2.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 75.0*(a*a*a)*(b*b)*(d*d*d*d)*p + 65.625*(a*a*a)*(d*d)*(p*p) - 37.5*(a*a)*(b*b*b)*(d*d*d*d)*p - 196.875*(a*a)*b*(d*d)*(p*p) + 3.75*a*(b*b*b*b)*(d*d*d*d)*p + 118.125*a*(b*b)*(d*d)*(p*p) + 118.125*a*(p*p*p) - 13.125*(b*b*b)*(d*d)*(p*p) - 78.75*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU464(double a, double b, double p, double d, double s){
	return ((0.0625*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 1.5*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 5.625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 7.03125*(a*a*a*a)*(d*d*d*d)*p - 5.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 37.5*(a*a*a)*b*(d*d*d*d)*p + 0.9375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 42.1875*(a*a)*(b*b)*(d*d*d*d)*p + 49.21875*(a*a)*(d*d)*(p*p) - 11.25*a*(b*b*b)*(d*d*d*d)*p - 78.75*a*b*(d*d)*(p*p) + 0.46875*(b*b*b*b)*(d*d*d*d)*p + 19.6875*(b*b)*(d*d)*(p*p) + 24.609375*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU465(double a, double b, double p, double d, double s){
	return (d*(0.1875*(a*a*a*a*a)*(d*d*d*d) - 1.875*(a*a*a*a)*b*(d*d*d*d) + 3.75*(a*a*a)*(b*b)*(d*d*d*d) + 6.5625*(a*a*a)*(d*d)*p - 1.875*(a*a)*(b*b*b)*(d*d*d*d) - 19.6875*(a*a)*b*(d*d)*p + 0.1875*a*(b*b*b*b)*(d*d*d*d) + 11.8125*a*(b*b)*(d*d)*p + 17.71875*a*(p*p) - 1.3125*(b*b*b)*(d*d)*p - 11.8125*b*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU466(double a, double b, double p, double d, double s){
	return ((0.234375*(a*a*a*a)*(d*d*d*d) - 1.25*(a*a*a)*b*(d*d*d*d) + 1.40625*(a*a)*(b*b)*(d*d*d*d) + 3.28125*(a*a)*(d*d)*p - 0.375*a*(b*b*b)*(d*d*d*d) - 5.25*a*b*(d*d)*p + 0.015625*(b*b*b*b)*(d*d*d*d) + 1.3125*(b*b)*(d*d)*p + 2.4609375*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU467(double a, double b, double p, double d, double s){
	return (d*(0.15625*(a*a*a)*(d*d) - 0.46875*(a*a)*b*(d*d) + 0.28125*a*(b*b)*(d*d) + 0.84375*a*p - 0.03125*(b*b*b)*(d*d) - 0.5625*b*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU468(double a, double b, double p, double d, double s){
	return ((0.05859375*(a*a)*(d*d) - 0.09375*a*b*(d*d) + 0.0234375*(b*b)*(d*d) + 0.087890625*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU469(double a, double b, double p, double d, double s){
	return (d*(0.01171875*a - 0.0078125*b)/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU4610(double a, double b, double p, double d, double s){
	return (0.0009765625/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU470(double a, double b, double p, double d, double s){
	return (d*((a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 3.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 0.75*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 14.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 21.0 *(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 10.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 94.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 39.375*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 105.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 262.5*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 26.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 393.75*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 229.6875*(a*a*a)*(d*d)*(p*p*p*p) - 157.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 551.25*(a*a)*b*(d*d)*(p*p*p*p) + 13.125*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 275.625*a*(b*b)*(d*d)*(p*p*p*p) + 206.71875*a*(p*p*p*p*p) - 26.25*(b*b*b)*(d*d)*(p*p*p*p) - 118.125*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU471(double a, double b, double p, double d, double s){
	return ((-2.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 3.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 3.5*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 31.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 13.125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 63.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 157.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 26.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 393.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 229.6875*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 262.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 918.75*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 39.375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 826.875*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 620.15625*(a*a)*(d*d)*(p*p*p*p) - 183.75*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 826.875*a*b*(d*d)*(p*p*p*p) + 6.5625*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 177.1875*(b*b)*(d*d)*(p*p*p*p) + 162.421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU472(double a, double b, double p, double d, double s){
	return (d*(1.5*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.75*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 7.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 21.0 *(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 5.25*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 94.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 59.0625*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 105.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 393.75*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 26.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 590.625*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 459.375*(a*a*a)*(d*d)*(p*p*p) - 236.25*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 1102.5*(a*a)*b*(d*d)*(p*p*p) + 19.6875*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 551.25*a*(b*b)*(d*d)*(p*p*p) + 516.796875*a*(p*p*p*p) - 52.5*(b*b*b)*(d*d)*(p*p*p) - 295.3125*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU473(double a, double b, double p, double d, double s){
	return ((-0.5*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 5.25*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 4.375*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 10.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 52.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 4.375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 131.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 114.84375*(a*a*a*a)*(d*d*d*d)*(p*p) - 87.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 459.375*(a*a*a)*b*(d*d*d*d)*(p*p) + 13.125*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 413.4375*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 413.4375*(a*a)*(d*d)*(p*p*p) - 91.875*a*(b*b*b)*(d*d*d*d)*(p*p) - 551.25*a*b*(d*d)*(p*p*p) + 3.28125*(b*b*b*b)*(d*d*d*d)*(p*p) + 118.125*(b*b)*(d*d)*(p*p*p) + 135.3515625*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU474(double a, double b, double p, double d, double s){
	return (d*(0.0625*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 1.75*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 7.875*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 9.84375*(a*a*a*a*a)*(d*d*d*d)*p - 8.75*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 65.625*(a*a*a*a)*b*(d*d*d*d)*p + 2.1875*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 98.4375*(a*a*a)*(b*b)*(d*d*d*d)*p + 114.84375*(a*a*a)*(d*d)*(p*p) - 39.375*(a*a)*(b*b*b)*(d*d*d*d)*p - 275.625*(a*a)*b*(d*d)*(p*p) + 3.28125*a*(b*b*b*b)*(d*d*d*d)*p + 137.8125*a*(b*b)*(d*d)*(p*p) + 172.265625*a*(p*p*p) - 13.125*(b*b*b)*(d*d)*(p*p) - 98.4375*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU475(double a, double b, double p, double d, double s){
	return ((0.21875*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 2.625*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 6.5625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 11.484375*(a*a*a*a)*(d*d*d*d)*p - 4.375*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 45.9375*(a*a*a)*b*(d*d*d*d)*p + 0.65625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 41.34375*(a*a)*(b*b)*(d*d*d*d)*p + 62.015625*(a*a)*(d*d)*(p*p) - 9.1875*a*(b*b*b)*(d*d*d*d)*p - 82.6875*a*b*(d*d)*(p*p) + 0.328125*(b*b*b*b)*(d*d*d*d)*p + 17.71875*(b*b)*(d*d)*(p*p) + 27.0703125*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU476(double a, double b, double p, double d, double s){
	return (d*(0.328125*(a*a*a*a*a)*(d*d*d*d) - 2.1875*(a*a*a*a)*b*(d*d*d*d) + 3.28125*(a*a*a)*(b*b)*(d*d*d*d) + 7.65625*(a*a*a)*(d*d)*p - 1.3125*(a*a)*(b*b*b)*(d*d*d*d) - 18.375*(a*a)*b*(d*d)*p + 0.109375*a*(b*b*b*b)*(d*d*d*d) + 9.1875*a*(b*b)*(d*d)*p + 17.2265625*a*(p*p) - 0.875*(b*b*b)*(d*d)*p - 9.84375*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU477(double a, double b, double p, double d, double s){
	return ((0.2734375*(a*a*a*a)*(d*d*d*d) - 1.09375*(a*a*a)*b*(d*d*d*d) + 0.984375*(a*a)*(b*b)*(d*d*d*d) + 2.953125*(a*a)*(d*d)*p - 0.21875*a*(b*b*b)*(d*d*d*d) - 3.9375*a*b*(d*d)*p + 0.0078125*(b*b*b*b)*(d*d*d*d) + 0.84375*(b*b)*(d*d)*p + 1.93359375*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU478(double a, double b, double p, double d, double s){
	return (d*(0.13671875*(a*a*a)*(d*d) - 0.328125*(a*a)*b*(d*d) + 0.1640625*a*(b*b)*(d*d) + 0.615234375*a*p - 0.015625*(b*b*b)*(d*d) - 0.3515625*b*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU479(double a, double b, double p, double d, double s){
	return ((0.041015625*(a*a)*(d*d) - 0.0546875*a*b*(d*d) + 0.01171875*(b*b)*(d*d) + 0.0537109375*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU4710(double a, double b, double p, double d, double s){
	return (d*(0.0068359375*a - 0.00390625*b)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU4711(double a, double b, double p, double d, double s){
	return (0.00048828125/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU480(double a, double b, double p, double d, double s){
	return (((a*a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 3.0*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 0.75*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*(p*p) - 16.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 24.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 14.0*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 126.0*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 52.5*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 168.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 420.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 52.5*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 787.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 459.375*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 420.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 1470.0*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 52.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 1102.5*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 826.875*(a*a)*(d*d)*(p*p*p*p*p) - 210.0*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 945.0*a*b*(d*d)*(p*p*p*p*p) + 6.5625*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 177.1875*(b*b)*(d*d)*(p*p*p*p*p) + 162.421875*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU481(double a, double b, double p, double d, double s){
	return (d*(-2.0*(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 3.0*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 4.0*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 36.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 15.0*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 84.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 210.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 42.0*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 630.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 367.5*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 525.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 1837.5*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 105.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 2205.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 1653.75*(a*a*a)*(d*d)*(p*p*p*p) - 735.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 3307.5*(a*a)*b*(d*d)*(p*p*p*p) + 52.5*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 1417.5*a*(b*b)*(d*d)*(p*p*p*p) + 1299.375*a*(p*p*p*p*p) - 118.125*(b*b*b)*(d*d)*(p*p*p*p) - 649.6875*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU482(double a, double b, double p, double d, double s){
	return ((1.5*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d) + 0.75*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*p - 8.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 24.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 7.0*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 126.0*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 78.75*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 168.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 630.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 52.5*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1181.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 918.75*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 630.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 2940.0*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 78.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 2205.0*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 2067.1875*(a*a)*(d*d)*(p*p*p*p) - 420.0*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 2362.5*a*b*(d*d)*(p*p*p*p) + 13.125*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 442.96875*(b*b)*(d*d)*(p*p*p*p) + 487.265625*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU483(double a, double b, double p, double d, double s){
	return (d*(-0.5*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 6.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 5.0*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 14.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 70.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 7.0*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 210.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 183.75*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 175.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 918.75*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 35.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1102.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 1102.5*(a*a*a)*(d*d)*(p*p*p) - 367.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 2205.0*(a*a)*b*(d*d)*(p*p*p) + 26.25*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 945.0*a*(b*b)*(d*d)*(p*p*p) + 1082.8125*a*(p*p*p*p) - 78.75*(b*b*b)*(d*d)*(p*p*p) - 541.40625*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU484(double a, double b, double p, double d, double s){
	return ((0.0625*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d) - 2.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 10.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 13.125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 14.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 105.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 4.375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 196.875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 229.6875*(a*a*a*a)*(d*d*d*d)*(p*p) - 105.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 735.0*(a*a*a)*b*(d*d*d*d)*(p*p) + 13.125*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 551.25*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 689.0625*(a*a)*(d*d)*(p*p*p) - 105.0*a*(b*b*b)*(d*d*d*d)*(p*p) - 787.5*a*b*(d*d)*(p*p*p) + 3.28125*(b*b*b*b)*(d*d*d*d)*(p*p) + 147.65625*(b*b)*(d*d)*(p*p*p) + 203.02734375*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU485(double a, double b, double p, double d, double s){
	return (d*(0.25*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 3.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 10.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 18.375*(a*a*a*a*a)*(d*d*d*d)*p - 8.75*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 91.875*(a*a*a*a)*b*(d*d*d*d)*p + 1.75*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 110.25*(a*a*a)*(b*b)*(d*d*d*d)*p + 165.375*(a*a*a)*(d*d)*(p*p) - 36.75*(a*a)*(b*b*b)*(d*d*d*d)*p - 330.75*(a*a)*b*(d*d)*(p*p) + 2.625*a*(b*b*b*b)*(d*d*d*d)*p + 141.75*a*(b*b)*(d*d)*(p*p) + 216.5625*a*(p*p*p) - 11.8125*(b*b*b)*(d*d)*(p*p) - 108.28125*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU486(double a, double b, double p, double d, double s){
	return ((0.4375*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 3.5*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 6.5625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 15.3125*(a*a*a*a)*(d*d*d*d)*p - 3.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 49.0*(a*a*a)*b*(d*d*d*d)*p + 0.4375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 36.75*(a*a)*(b*b)*(d*d*d*d)*p + 68.90625*(a*a)*(d*d)*(p*p) - 7.0*a*(b*b*b)*(d*d*d*d)*p - 78.75*a*b*(d*d)*(p*p) + 0.21875*(b*b*b*b)*(d*d*d*d)*p + 14.765625*(b*b)*(d*d)*(p*p) + 27.0703125*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU487(double a, double b, double p, double d, double s){
	return (d*(0.4375*(a*a*a*a*a)*(d*d*d*d) - 2.1875*(a*a*a*a)*b*(d*d*d*d) + 2.625*(a*a*a)*(b*b)*(d*d*d*d) + 7.875*(a*a*a)*(d*d)*p - 0.875*(a*a)*(b*b*b)*(d*d*d*d) - 15.75*(a*a)*b*(d*d)*p + 0.0625*a*(b*b*b*b)*(d*d*d*d) + 6.75*a*(b*b)*(d*d)*p + 15.46875*a*(p*p) - 0.5625*(b*b*b)*(d*d)*p - 7.734375*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU488(double a, double b, double p, double d, double s){
	return ((0.2734375*(a*a*a*a)*(d*d*d*d) - 0.875*(a*a*a)*b*(d*d*d*d) + 0.65625*(a*a)*(b*b)*(d*d*d*d) + 2.4609375*(a*a)*(d*d)*p - 0.125*a*(b*b*b)*(d*d*d*d) - 2.8125*a*b*(d*d)*p + 0.00390625*(b*b*b*b)*(d*d*d*d) + 0.52734375*(b*b)*(d*d)*p + 1.4501953125*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU489(double a, double b, double p, double d, double s){
	return (d*(0.109375*(a*a*a)*(d*d) - 0.21875*(a*a)*b*(d*d) + 0.09375*a*(b*b)*(d*d) + 0.4296875*a*p - 0.0078125*(b*b*b)*(d*d) - 0.21484375*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU4810(double a, double b, double p, double d, double s){
	return ((0.02734375*(a*a)*(d*d) - 0.03125*a*b*(d*d) + 0.005859375*(b*b)*(d*d) + 0.0322265625*p)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU4811(double a, double b, double p, double d, double s){
	return (d*(0.00390625*a - 0.001953125*b)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU4812(double a, double b, double p, double d, double s){
	return (0.000244140625/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU500(double a, double b, double p, double d, double s){
	return (b*d*(-(b*b*b*b)*(d*d*d*d) - 5.0*(b*b)*(d*d)*p - 3.75*(p*p))/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU501(double a, double b, double p, double d, double s){
	return ((2.5*(b*b*b*b)*(d*d*d*d) + 7.5*(b*b)*(d*d)*p + 1.875*(p*p))/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU502(double a, double b, double p, double d, double s){
	return (b*d*(-2.5*(b*b)*(d*d) - 3.75*p)/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU503(double a, double b, double p, double d, double s){
	return ((1.25*(b*b)*(d*d) + 0.625*p)/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU504(double a, double b, double p, double d, double s){
	return (-0.3125*b*d/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU505(double a, double b, double p, double d, double s){
	return (0.03125/(p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU510(double a, double b, double p, double d, double s){
	return ((-a*(b*b*b*b*b)*(d*d*d*d*d*d) - 5.0*a*(b*b*b)*(d*d*d*d)*p - 3.75*a*b*(d*d)*(p*p) + 2.5*(b*b*b*b)*(d*d*d*d)*p + 7.5*(b*b)*(d*d)*(p*p) + 1.875*(p*p*p))/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU511(double a, double b, double p, double d, double s){
	return (d*(2.5*a*(b*b*b*b)*(d*d*d*d) + 7.5*a*(b*b)*(d*d)*p + 1.875*a*(p*p) - 0.5*(b*b*b*b*b)*(d*d*d*d) - 7.5*(b*b*b)*(d*d)*p - 9.375*b*(p*p))/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU512(double a, double b, double p, double d, double s){
	return ((-2.5*a*(b*b*b)*(d*d*d*d) - 3.75*a*b*(d*d)*p + 1.25*(b*b*b*b)*(d*d*d*d) + 7.5*(b*b)*(d*d)*p + 2.8125*(p*p))/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU513(double a, double b, double p, double d, double s){
	return (d*(1.25*a*(b*b)*(d*d) + 0.625*a*p - 1.25*(b*b*b)*(d*d) - 3.125*b*p)/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU514(double a, double b, double p, double d, double s){
	return ((-0.3125*a*b*(d*d) + 0.625*(b*b)*(d*d) + 0.46875*p)/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU515(double a, double b, double p, double d, double s){
	return (d*(0.03125*a - 0.15625*b)/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU516(double a, double b, double p, double d, double s){
	return (0.015625/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU520(double a, double b, double p, double d, double s){
	return (d*(-(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 5.0*(a*a)*(b*b*b)*(d*d*d*d)*p - 3.75*(a*a)*b*(d*d)*(p*p) + 5.0*a*(b*b*b*b)*(d*d*d*d)*p + 15.0*a*(b*b)*(d*d)*(p*p) + 3.75*a*(p*p*p) - 0.5*(b*b*b*b*b)*(d*d*d*d)*p - 7.5*(b*b*b)*(d*d)*(p*p) - 9.375*b*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU521(double a, double b, double p, double d, double s){
	return ((2.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 7.5*(a*a)*(b*b)*(d*d*d*d)*p + 1.875*(a*a)*(d*d)*(p*p) - a*(b*b*b*b*b)*(d*d*d*d*d*d) - 15.0*a*(b*b*b)*(d*d*d*d)*p - 18.75*a*b*(d*d)*(p*p) + 3.75*(b*b*b*b)*(d*d*d*d)*p + 18.75*(b*b)*(d*d)*(p*p) + 6.5625*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU522(double a, double b, double p, double d, double s){
	return (d*(-2.5*(a*a)*(b*b*b)*(d*d*d*d) - 3.75*(a*a)*b*(d*d)*p + 2.5*a*(b*b*b*b)*(d*d*d*d) + 15.0*a*(b*b)*(d*d)*p + 5.625*a*(p*p) - 0.25*(b*b*b*b*b)*(d*d*d*d) - 7.5*(b*b*b)*(d*d)*p - 14.0625*b*(p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU523(double a, double b, double p, double d, double s){
	return ((1.25*(a*a)*(b*b)*(d*d*d*d) + 0.625*(a*a)*(d*d)*p - 2.5*a*(b*b*b)*(d*d*d*d) - 6.25*a*b*(d*d)*p + 0.625*(b*b*b*b)*(d*d*d*d) + 6.25*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU524(double a, double b, double p, double d, double s){
	return (d*(-0.3125*(a*a)*b*(d*d) + 1.25*a*(b*b)*(d*d) + 0.9375*a*p - 0.625*(b*b*b)*(d*d) - 2.34375*b*p)/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU525(double a, double b, double p, double d, double s){
	return ((0.03125*(a*a)*(d*d) - 0.3125*a*b*(d*d) + 0.3125*(b*b)*(d*d) + 0.328125*p)/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU526(double a, double b, double p, double d, double s){
	return (d*(0.03125*a - 0.078125*b)/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU527(double a, double b, double p, double d, double s){
	return (0.0078125/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU530(double a, double b, double p, double d, double s){
	return ((-(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 5.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 3.75*(a*a*a)*b*(d*d*d*d)*(p*p) + 7.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 22.5*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 5.625*(a*a)*(d*d)*(p*p*p) - 1.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 22.5*a*(b*b*b)*(d*d*d*d)*(p*p) - 28.125*a*b*(d*d)*(p*p*p) + 3.75*(b*b*b*b)*(d*d*d*d)*(p*p) + 18.75*(b*b)*(d*d)*(p*p*p) + 6.5625*(p*p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU531(double a, double b, double p, double d, double s){
	return (d*(2.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 7.5*(a*a*a)*(b*b)*(d*d*d*d)*p + 1.875*(a*a*a)*(d*d)*(p*p) - 1.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 22.5*(a*a)*(b*b*b)*(d*d*d*d)*p - 28.125*(a*a)*b*(d*d)*(p*p) + 11.25*a*(b*b*b*b)*(d*d*d*d)*p + 56.25*a*(b*b)*(d*d)*(p*p) + 19.6875*a*(p*p*p) - 0.75*(b*b*b*b*b)*(d*d*d*d)*p - 18.75*(b*b*b)*(d*d)*(p*p) - 32.8125*b*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU532(double a, double b, double p, double d, double s){
	return ((-2.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 3.75*(a*a*a)*b*(d*d*d*d)*p + 3.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 22.5*(a*a)*(b*b)*(d*d*d*d)*p + 8.4375*(a*a)*(d*d)*(p*p) - 0.75*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 22.5*a*(b*b*b)*(d*d*d*d)*p - 42.1875*a*b*(d*d)*(p*p) + 3.75*(b*b*b*b)*(d*d*d*d)*p + 28.125*(b*b)*(d*d)*(p*p) + 13.125*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU533(double a, double b, double p, double d, double s){
	return (d*(1.25*(a*a*a)*(b*b)*(d*d*d*d) + 0.625*(a*a*a)*(d*d)*p - 3.75*(a*a)*(b*b*b)*(d*d*d*d) - 9.375*(a*a)*b*(d*d)*p + 1.875*a*(b*b*b*b)*(d*d*d*d) + 18.75*a*(b*b)*(d*d)*p + 9.84375*a*(p*p) - 0.125*(b*b*b*b*b)*(d*d*d*d) - 6.25*(b*b*b)*(d*d)*p - 16.40625*b*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU534(double a, double b, double p, double d, double s){
	return ((-0.3125*(a*a*a)*b*(d*d*d*d) + 1.875*(a*a)*(b*b)*(d*d*d*d) + 1.40625*(a*a)*(d*d)*p - 1.875*a*(b*b*b)*(d*d*d*d) - 7.03125*a*b*(d*d)*p + 0.3125*(b*b*b*b)*(d*d*d*d) + 4.6875*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU535(double a, double b, double p, double d, double s){
	return (d*(0.03125*(a*a*a)*(d*d) - 0.46875*(a*a)*b*(d*d) + 0.9375*a*(b*b)*(d*d) + 0.984375*a*p - 0.3125*(b*b*b)*(d*d) - 1.640625*b*p)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU536(double a, double b, double p, double d, double s){
	return ((0.046875*(a*a)*(d*d) - 0.234375*a*b*(d*d) + 0.15625*(b*b)*(d*d) + 0.21875*p)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU537(double a, double b, double p, double d, double s){
	return (d*(0.0234375*a - 0.0390625*b)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU538(double a, double b, double p, double d, double s){
	return (0.00390625/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU540(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 5.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 3.75*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 10.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 30.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 7.5*(a*a*a)*(d*d)*(p*p*p) - 3.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 45.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 56.25*(a*a)*b*(d*d)*(p*p*p) + 15.0*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 75.0*a*(b*b)*(d*d)*(p*p*p) + 26.25*a*(p*p*p*p) - 0.75*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 18.75*(b*b*b)*(d*d)*(p*p*p) - 32.8125*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU541(double a, double b, double p, double d, double s){
	return ((2.5*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 7.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 1.875*(a*a*a*a)*(d*d*d*d)*(p*p) - 2.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 30.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 37.5*(a*a*a)*b*(d*d*d*d)*(p*p) + 22.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 112.5*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 39.375*(a*a)*(d*d)*(p*p*p) - 3.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 75.0*a*(b*b*b)*(d*d*d*d)*(p*p) - 131.25*a*b*(d*d)*(p*p*p) + 9.375*(b*b*b*b)*(d*d*d*d)*(p*p) + 65.625*(b*b)*(d*d)*(p*p*p) + 29.53125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU542(double a, double b, double p, double d, double s){
	return (d*(-2.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 3.75*(a*a*a*a)*b*(d*d*d*d)*p + 5.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 30.0*(a*a*a)*(b*b)*(d*d*d*d)*p + 11.25*(a*a*a)*(d*d)*(p*p) - 1.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 45.0*(a*a)*(b*b*b)*(d*d*d*d)*p - 84.375*(a*a)*b*(d*d)*(p*p) + 15.0*a*(b*b*b*b)*(d*d*d*d)*p + 112.5*a*(b*b)*(d*d)*(p*p) + 52.5*a*(p*p*p) - 0.75*(b*b*b*b*b)*(d*d*d*d)*p - 28.125*(b*b*b)*(d*d)*(p*p) - 65.625*b*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU543(double a, double b, double p, double d, double s){
	return ((1.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.625*(a*a*a*a)*(d*d*d*d)*p - 5.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 12.5*(a*a*a)*b*(d*d*d*d)*p + 3.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 37.5*(a*a)*(b*b)*(d*d*d*d)*p + 19.6875*(a*a)*(d*d)*(p*p) - 0.5*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 25.0*a*(b*b*b)*(d*d*d*d)*p - 65.625*a*b*(d*d)*(p*p) + 3.125*(b*b*b*b)*(d*d*d*d)*p + 32.8125*(b*b)*(d*d)*(p*p) + 19.6875*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU544(double a, double b, double p, double d, double s){
	return (d*(-0.3125*(a*a*a*a)*b*(d*d*d*d) + 2.5*(a*a*a)*(b*b)*(d*d*d*d) + 1.875*(a*a*a)*(d*d)*p - 3.75*(a*a)*(b*b*b)*(d*d*d*d) - 14.0625*(a*a)*b*(d*d)*p + 1.25*a*(b*b*b*b)*(d*d*d*d) + 18.75*a*(b*b)*(d*d)*p + 13.125*a*(p*p) - 0.0625*(b*b*b*b*b)*(d*d*d*d) - 4.6875*(b*b*b)*(d*d)*p - 16.40625*b*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU545(double a, double b, double p, double d, double s){
	return ((0.03125*(a*a*a*a)*(d*d*d*d) - 0.625*(a*a*a)*b*(d*d*d*d) + 1.875*(a*a)*(b*b)*(d*d*d*d) + 1.96875*(a*a)*(d*d)*p - 1.25*a*(b*b*b)*(d*d*d*d) - 6.5625*a*b*(d*d)*p + 0.15625*(b*b*b*b)*(d*d*d*d) + 3.28125*(b*b)*(d*d)*p + 2.953125*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU546(double a, double b, double p, double d, double s){
	return (d*(0.0625*(a*a*a)*(d*d) - 0.46875*(a*a)*b*(d*d) + 0.625*a*(b*b)*(d*d) + 0.875*a*p - 0.15625*(b*b*b)*(d*d) - 1.09375*b*p)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU547(double a, double b, double p, double d, double s){
	return ((0.046875*(a*a)*(d*d) - 0.15625*a*b*(d*d) + 0.078125*(b*b)*(d*d) + 0.140625*p)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU548(double a, double b, double p, double d, double s){
	return (d*(0.015625*a - 0.01953125*b)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU549(double a, double b, double p, double d, double s){
	return (0.001953125/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU550(double a, double b, double p, double d, double s){
	return ((-(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 5.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 3.75*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 12.5*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 37.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 9.375*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 5.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 75.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 93.75*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 37.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 187.5*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 65.625*(a*a)*(d*d)*(p*p*p*p) - 3.75*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 93.75*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 164.0625*a*b*(d*d)*(p*p*p*p) + 9.375*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 65.625*(b*b)*(d*d)*(p*p*p*p) + 29.53125*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU551(double a, double b, double p, double d, double s){
	return (d*(2.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 7.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 1.875*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 2.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 37.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 46.875*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 37.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 187.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 65.625*(a*a*a)*(d*d)*(p*p*p) - 7.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 187.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 328.125*(a*a)*b*(d*d)*(p*p*p) + 46.875*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 328.125*a*(b*b)*(d*d)*(p*p*p) + 147.65625*a*(p*p*p*p) - 1.875*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 65.625*(b*b*b)*(d*d)*(p*p*p) - 147.65625*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU552(double a, double b, double p, double d, double s){
	return ((-2.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 3.75*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 6.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 37.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 14.0625*(a*a*a*a)*(d*d*d*d)*(p*p) - 2.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 75.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 140.625*(a*a*a)*b*(d*d*d*d)*(p*p) + 37.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 281.25*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 131.25*(a*a)*(d*d)*(p*p*p) - 3.75*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 140.625*a*(b*b*b)*(d*d*d*d)*(p*p) - 328.125*a*b*(d*d)*(p*p*p) + 14.0625*(b*b*b*b)*(d*d*d*d)*(p*p) + 131.25*(b*b)*(d*d)*(p*p*p) + 73.828125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU553(double a, double b, double p, double d, double s){
	return (d*(1.25*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.625*(a*a*a*a*a)*(d*d*d*d)*p - 6.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 15.625*(a*a*a*a)*b*(d*d*d*d)*p + 6.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 62.5*(a*a*a)*(b*b)*(d*d*d*d)*p + 32.8125*(a*a*a)*(d*d)*(p*p) - 1.25*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 62.5*(a*a)*(b*b*b)*(d*d*d*d)*p - 164.0625*(a*a)*b*(d*d)*(p*p) + 15.625*a*(b*b*b*b)*(d*d*d*d)*p + 164.0625*a*(b*b)*(d*d)*(p*p) + 98.4375*a*(p*p*p) - 0.625*(b*b*b*b*b)*(d*d*d*d)*p - 32.8125*(b*b*b)*(d*d)*(p*p) - 98.4375*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU554(double a, double b, double p, double d, double s){
	return ((-0.3125*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 3.125*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 2.34375*(a*a*a*a)*(d*d*d*d)*p - 6.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 23.4375*(a*a*a)*b*(d*d*d*d)*p + 3.125*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 46.875*(a*a)*(b*b)*(d*d*d*d)*p + 32.8125*(a*a)*(d*d)*(p*p) - 0.3125*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 23.4375*a*(b*b*b)*(d*d*d*d)*p - 82.03125*a*b*(d*d)*(p*p) + 2.34375*(b*b*b*b)*(d*d*d*d)*p + 32.8125*(b*b)*(d*d)*(p*p) + 24.609375*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU555(double a, double b, double p, double d, double s){
	return (d*(0.03125*(a*a*a*a*a)*(d*d*d*d) - 0.78125*(a*a*a*a)*b*(d*d*d*d) + 3.125*(a*a*a)*(b*b)*(d*d*d*d) + 3.28125*(a*a*a)*(d*d)*p - 3.125*(a*a)*(b*b*b)*(d*d*d*d) - 16.40625*(a*a)*b*(d*d)*p + 0.78125*a*(b*b*b*b)*(d*d*d*d) + 16.40625*a*(b*b)*(d*d)*p + 14.765625*a*(p*p) - 0.03125*(b*b*b*b*b)*(d*d*d*d) - 3.28125*(b*b*b)*(d*d)*p - 14.765625*b*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU556(double a, double b, double p, double d, double s){
	return ((0.078125*(a*a*a*a)*(d*d*d*d) - 0.78125*(a*a*a)*b*(d*d*d*d) + 1.5625*(a*a)*(b*b)*(d*d*d*d) + 2.1875*(a*a)*(d*d)*p - 0.78125*a*(b*b*b)*(d*d*d*d) - 5.46875*a*b*(d*d)*p + 0.078125*(b*b*b*b)*(d*d*d*d) + 2.1875*(b*b)*(d*d)*p + 2.4609375*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU557(double a, double b, double p, double d, double s){
	return (d*(0.078125*(a*a*a)*(d*d) - 0.390625*(a*a)*b*(d*d) + 0.390625*a*(b*b)*(d*d) + 0.703125*a*p - 0.078125*(b*b*b)*(d*d) - 0.703125*b*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU558(double a, double b, double p, double d, double s){
	return ((0.0390625*(a*a)*(d*d) - 0.09765625*a*b*(d*d) + 0.0390625*(b*b)*(d*d) + 0.087890625*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU559(double a, double b, double p, double d, double s){
	return (0.009765625*d*(a - b)/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU5510(double a, double b, double p, double d, double s){
	return (0.0009765625/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU560(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 5.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 3.75*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 15.0*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 45.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 11.25*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 7.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 112.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 140.625*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 75.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 375.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 131.25*(a*a*a)*(d*d)*(p*p*p*p) - 11.25*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 281.25*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 492.1875*(a*a)*b*(d*d)*(p*p*p*p) + 56.25*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 393.75*a*(b*b)*(d*d)*(p*p*p*p) + 177.1875*a*(p*p*p*p*p) - 1.875*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 65.625*(b*b*b)*(d*d)*(p*p*p*p) - 147.65625*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU561(double a, double b, double p, double d, double s){
	return ((2.5*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 7.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 1.875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 3.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 45.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 56.25*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 56.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 281.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 98.4375*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 15.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 375.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 656.25*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 140.625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 984.375*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 442.96875*(a*a)*(d*d)*(p*p*p*p) - 11.25*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 393.75*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 885.9375*a*b*(d*d)*(p*p*p*p) + 32.8125*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 295.3125*(b*b)*(d*d)*(p*p*p*p) + 162.421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU562(double a, double b, double p, double d, double s){
	return (d*(-2.5*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 3.75*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 7.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 45.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 16.875*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 3.75*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 112.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 210.9375*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 75.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 562.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 262.5*(a*a*a)*(d*d)*(p*p*p) - 11.25*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 421.875*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 984.375*(a*a)*b*(d*d)*(p*p*p) + 84.375*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 787.5*a*(b*b)*(d*d)*(p*p*p) + 442.96875*a*(p*p*p*p) - 2.8125*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 131.25*(b*b*b)*(d*d)*(p*p*p) - 369.140625*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU563(double a, double b, double p, double d, double s){
	return ((1.25*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.625*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 7.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 18.75*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 9.375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 93.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 49.21875*(a*a*a*a)*(d*d*d*d)*(p*p) - 2.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 125.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 328.125*(a*a*a)*b*(d*d*d*d)*(p*p) + 46.875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 492.1875*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 295.3125*(a*a)*(d*d)*(p*p*p) - 3.75*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 196.875*a*(b*b*b)*(d*d*d*d)*(p*p) - 590.625*a*b*(d*d)*(p*p*p) + 16.40625*(b*b*b*b)*(d*d*d*d)*(p*p) + 196.875*(b*b)*(d*d)*(p*p*p) + 135.3515625*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU564(double a, double b, double p, double d, double s){
	return (d*(-0.3125*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 3.75*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 2.8125*(a*a*a*a*a)*(d*d*d*d)*p - 9.375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 35.15625*(a*a*a*a)*b*(d*d*d*d)*p + 6.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 93.75*(a*a*a)*(b*b)*(d*d*d*d)*p + 65.625*(a*a*a)*(d*d)*(p*p) - 0.9375*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 70.3125*(a*a)*(b*b*b)*(d*d*d*d)*p - 246.09375*(a*a)*b*(d*d)*(p*p) + 14.0625*a*(b*b*b*b)*(d*d*d*d)*p + 196.875*a*(b*b)*(d*d)*(p*p) + 147.65625*a*(p*p*p) - 0.46875*(b*b*b*b*b)*(d*d*d*d)*p - 32.8125*(b*b*b)*(d*d)*(p*p) - 123.046875*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU565(double a, double b, double p, double d, double s){
	return ((0.03125*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.9375*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 4.6875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 4.921875*(a*a*a*a)*(d*d*d*d)*p - 6.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 32.8125*(a*a*a)*b*(d*d*d*d)*p + 2.34375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 49.21875*(a*a)*(b*b)*(d*d*d*d)*p + 44.296875*(a*a)*(d*d)*(p*p) - 0.1875*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 19.6875*a*(b*b*b)*(d*d*d*d)*p - 88.59375*a*b*(d*d)*(p*p) + 1.640625*(b*b*b*b)*(d*d*d*d)*p + 29.53125*(b*b)*(d*d)*(p*p) + 27.0703125*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU566(double a, double b, double p, double d, double s){
	return (d*(0.09375*(a*a*a*a*a)*(d*d*d*d) - 1.171875*(a*a*a*a)*b*(d*d*d*d) + 3.125*(a*a*a)*(b*b)*(d*d*d*d) + 4.375*(a*a*a)*(d*d)*p - 2.34375*(a*a)*(b*b*b)*(d*d*d*d) - 16.40625*(a*a)*b*(d*d)*p + 0.46875*a*(b*b*b*b)*(d*d*d*d) + 13.125*a*(b*b)*(d*d)*p + 14.765625*a*(p*p) - 0.015625*(b*b*b*b*b)*(d*d*d*d) - 2.1875*(b*b*b)*(d*d)*p - 12.3046875*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU567(double a, double b, double p, double d, double s){
	return ((0.1171875*(a*a*a*a)*(d*d*d*d) - 0.78125*(a*a*a)*b*(d*d*d*d) + 1.171875*(a*a)*(b*b)*(d*d*d*d) + 2.109375*(a*a)*(d*d)*p - 0.46875*a*(b*b*b)*(d*d*d*d) - 4.21875*a*b*(d*d)*p + 0.0390625*(b*b*b*b)*(d*d*d*d) + 1.40625*(b*b)*(d*d)*p + 1.93359375*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU568(double a, double b, double p, double d, double s){
	return (d*(0.078125*(a*a*a)*(d*d) - 0.29296875*(a*a)*b*(d*d) + 0.234375*a*(b*b)*(d*d) + 0.52734375*a*p - 0.0390625*(b*b*b)*(d*d) - 0.439453125*b*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU569(double a, double b, double p, double d, double s){
	return ((0.029296875*(a*a)*(d*d) - 0.05859375*a*b*(d*d) + 0.01953125*(b*b)*(d*d) + 0.0537109375*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU5610(double a, double b, double p, double d, double s){
	return (d*(0.005859375*a - 0.0048828125*b)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU5611(double a, double b, double p, double d, double s){
	return (0.00048828125/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU570(double a, double b, double p, double d, double s){
	return ((-(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 5.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 3.75*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 17.5*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 52.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 13.125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 10.5*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 157.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 196.875*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 131.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 656.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 229.6875*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 26.25*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 656.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 1148.4375*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 196.875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 1378.125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 620.15625*(a*a)*(d*d)*(p*p*p*p*p) - 13.125*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 459.375*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 1033.59375*a*b*(d*d)*(p*p*p*p*p) + 32.8125*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 295.3125*(b*b)*(d*d)*(p*p*p*p*p) + 162.421875*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU571(double a, double b, double p, double d, double s){
	return (d*(2.5*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 7.5*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 1.875*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 3.5*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 52.5*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 65.625*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 78.75*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 393.75*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 137.8125*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 26.25*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 656.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 1148.4375*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 328.125*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 2296.875*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 1033.59375*(a*a*a)*(d*d)*(p*p*p*p) - 39.375*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 1378.125*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 3100.78125*(a*a)*b*(d*d)*(p*p*p*p) + 229.6875*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 2067.1875*a*(b*b)*(d*d)*(p*p*p*p) + 1136.953125*a*(p*p*p*p*p) - 6.5625*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 295.3125*(b*b*b)*(d*d)*(p*p*p*p) - 812.109375*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU572(double a, double b, double p, double d, double s){
	return ((-2.5*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 3.75*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 8.75*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 52.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 19.6875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 5.25*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 157.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 295.3125*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 131.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 984.375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 459.375*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 26.25*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 984.375*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 2296.875*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 295.3125*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 2756.25*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 1550.390625*(a*a)*(d*d)*(p*p*p*p) - 19.6875*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 918.75*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 2583.984375*a*b*(d*d)*(p*p*p*p) + 65.625*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 738.28125*(b*b)*(d*d)*(p*p*p*p) + 487.265625*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU573(double a, double b, double p, double d, double s){
	return (d*(1.25*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.625*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 8.75*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 21.875*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 13.125*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 131.25*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 68.90625*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 4.375*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 218.75*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 574.21875*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 109.375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1148.4375*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 689.0625*(a*a*a)*(d*d)*(p*p*p) - 13.125*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 689.0625*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 2067.1875*(a*a)*b*(d*d)*(p*p*p) + 114.84375*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 1378.125*a*(b*b)*(d*d)*(p*p*p) + 947.4609375*a*(p*p*p*p) - 3.28125*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 196.875*(b*b*b)*(d*d)*(p*p*p) - 676.7578125*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU574(double a, double b, double p, double d, double s){
	return ((-0.3125*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 4.375*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 3.28125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 13.125*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 49.21875*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 10.9375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 164.0625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 114.84375*(a*a*a*a)*(d*d*d*d)*(p*p) - 2.1875*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 164.0625*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 574.21875*(a*a*a)*b*(d*d*d*d)*(p*p) + 49.21875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 689.0625*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 516.796875*(a*a)*(d*d)*(p*p*p) - 3.28125*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 229.6875*a*(b*b*b)*(d*d*d*d)*(p*p) - 861.328125*a*b*(d*d)*(p*p*p) + 16.40625*(b*b*b*b)*(d*d*d*d)*(p*p) + 246.09375*(b*b)*(d*d)*(p*p*p) + 203.02734375*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU575(double a, double b, double p, double d, double s){
	return (d*(0.03125*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 1.09375*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 6.5625*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 6.890625*(a*a*a*a*a)*(d*d*d*d)*p - 10.9375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 57.421875*(a*a*a*a)*b*(d*d*d*d)*p + 5.46875*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 114.84375*(a*a*a)*(b*b)*(d*d*d*d)*p + 103.359375*(a*a*a)*(d*d)*(p*p) - 0.65625*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 68.90625*(a*a)*(b*b*b)*(d*d*d*d)*p - 310.078125*(a*a)*b*(d*d)*(p*p) + 11.484375*a*(b*b*b*b)*(d*d*d*d)*p + 206.71875*a*(b*b)*(d*d)*(p*p) + 189.4921875*a*(p*p*p) - 0.328125*(b*b*b*b*b)*(d*d*d*d)*p - 29.53125*(b*b*b)*(d*d)*(p*p) - 135.3515625*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU576(double a, double b, double p, double d, double s){
	return ((0.109375*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 1.640625*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 5.46875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 7.65625*(a*a*a*a)*(d*d*d*d)*p - 5.46875*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 38.28125*(a*a*a)*b*(d*d*d*d)*p + 1.640625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 45.9375*(a*a)*(b*b)*(d*d*d*d)*p + 51.6796875*(a*a)*(d*d)*(p*p) - 0.109375*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 15.3125*a*(b*b*b)*(d*d*d*d)*p - 86.1328125*a*b*(d*d)*(p*p) + 1.09375*(b*b*b*b)*(d*d*d*d)*p + 24.609375*(b*b)*(d*d)*(p*p) + 27.0703125*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU577(double a, double b, double p, double d, double s){
	return (d*(0.1640625*(a*a*a*a*a)*(d*d*d*d) - 1.3671875*(a*a*a*a)*b*(d*d*d*d) + 2.734375*(a*a*a)*(b*b)*(d*d*d*d) + 4.921875*(a*a*a)*(d*d)*p - 1.640625*(a*a)*(b*b*b)*(d*d*d*d) - 14.765625*(a*a)*b*(d*d)*p + 0.2734375*a*(b*b*b*b)*(d*d*d*d) + 9.84375*a*(b*b)*(d*d)*p + 13.53515625*a*(p*p) - 0.0078125*(b*b*b*b*b)*(d*d*d*d) - 1.40625*(b*b*b)*(d*d)*p - 9.66796875*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU578(double a, double b, double p, double d, double s){
	return ((0.13671875*(a*a*a*a)*(d*d*d*d) - 0.68359375*(a*a*a)*b*(d*d*d*d) + 0.8203125*(a*a)*(b*b)*(d*d*d*d) + 1.845703125*(a*a)*(d*d)*p - 0.2734375*a*(b*b*b)*(d*d*d*d) - 3.076171875*a*b*(d*d)*p + 0.01953125*(b*b*b*b)*(d*d*d*d) + 0.87890625*(b*b)*(d*d)*p + 1.4501953125*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU579(double a, double b, double p, double d, double s){
	return (d*(0.068359375*(a*a*a)*(d*d) - 0.205078125*(a*a)*b*(d*d) + 0.13671875*a*(b*b)*(d*d) + 0.3759765625*a*p - 0.01953125*(b*b*b)*(d*d) - 0.2685546875*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU5710(double a, double b, double p, double d, double s){
	return ((0.0205078125*(a*a)*(d*d) - 0.0341796875*a*b*(d*d) + 0.009765625*(b*b)*(d*d) + 0.0322265625*p)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU5711(double a, double b, double p, double d, double s){
	return (d*(0.00341796875*a - 0.00244140625*b)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU5712(double a, double b, double p, double d, double s){
	return (0.000244140625/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU580(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 5.0*(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 3.75*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 20.0*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 60.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 15.0*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 14.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 210.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 262.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 210.0*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 1050.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 367.5*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 52.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 1312.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 2296.875*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 525.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 3675.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 1653.75*(a*a*a)*(d*d)*(p*p*p*p*p) - 52.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 1837.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 4134.375*(a*a)*b*(d*d)*(p*p*p*p*p) + 262.5*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 2362.5*a*(b*b)*(d*d)*(p*p*p*p*p) + 1299.375*a*(p*p*p*p*p*p) - 6.5625*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p) - 295.3125*(b*b*b)*(d*d)*(p*p*p*p*p) - 812.109375*b*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU581(double a, double b, double p, double d, double s){
	return ((2.5*(a*a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 7.5*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 1.875*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*(p*p) - 4.0*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 60.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 75.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 105.0*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 525.0*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 183.75*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 42.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 1050.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 1837.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 656.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 4593.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 2067.1875*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 105.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 3675.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 8268.75*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 918.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 8268.75*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 4547.8125*(a*a)*(d*d)*(p*p*p*p*p) - 52.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 2362.5*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 6496.875*a*b*(d*d)*(p*p*p*p*p) + 147.65625*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 1624.21875*(b*b)*(d*d)*(p*p*p*p*p) + 1055.7421875*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU582(double a, double b, double p, double d, double s){
	return (d*(-2.5*(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 3.75*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 10.0*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 60.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 22.5*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 7.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 210.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 393.75*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 210.0*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1575.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 735.0*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 52.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1968.75*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 4593.75*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 787.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 7350.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 4134.375*(a*a*a)*(d*d)*(p*p*p*p) - 78.75*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 3675.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 10335.9375*(a*a)*b*(d*d)*(p*p*p*p) + 525.0*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 5906.25*a*(b*b)*(d*d)*(p*p*p*p) + 3898.125*a*(p*p*p*p*p) - 13.125*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 738.28125*(b*b*b)*(d*d)*(p*p*p*p) - 2436.328125*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU583(double a, double b, double p, double d, double s){
	return ((1.25*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d) + 0.625*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*p - 10.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 25.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 17.5*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 175.0*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 91.875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 7.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 350.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 918.75*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 218.75*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 2296.875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 1378.125*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 35.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1837.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 5512.5*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 459.375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 5512.5*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 3789.84375*(a*a)*(d*d)*(p*p*p*p) - 26.25*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 1575.0*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 5414.0625*a*b*(d*d)*(p*p*p*p) + 98.4375*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 1353.515625*(b*b)*(d*d)*(p*p*p*p) + 1055.7421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU584(double a, double b, double p, double d, double s){
	return (d*(-0.3125*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 5.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 3.75*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 17.5*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 65.625*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 17.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 262.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 183.75*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 4.375*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 328.125*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 1148.4375*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 131.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1837.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 1378.125*(a*a*a)*(d*d)*(p*p*p) - 13.125*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 918.75*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 3445.3125*(a*a)*b*(d*d)*(p*p*p) + 131.25*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 1968.75*a*(b*b)*(d*d)*(p*p*p) + 1624.21875*a*(p*p*p*p) - 3.28125*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 246.09375*(b*b*b)*(d*d)*(p*p*p) - 1015.13671875*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU585(double a, double b, double p, double d, double s){
	return ((0.03125*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d) - 1.25*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 8.75*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 9.1875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 17.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 91.875*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 10.9375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 229.6875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 206.71875*(a*a*a*a)*(d*d*d*d)*(p*p) - 1.75*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 183.75*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 826.875*(a*a*a)*b*(d*d*d*d)*(p*p) + 45.9375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 826.875*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 757.96875*(a*a)*(d*d)*(p*p*p) - 2.625*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 236.25*a*(b*b*b)*(d*d*d*d)*(p*p) - 1082.8125*a*b*(d*d)*(p*p*p) + 14.765625*(b*b*b*b)*(d*d*d*d)*(p*p) + 270.703125*(b*b)*(d*d)*(p*p*p) + 263.935546875*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU586(double a, double b, double p, double d, double s){
	return (d*(0.125*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 2.1875*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 8.75*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 12.25*(a*a*a*a*a)*(d*d*d*d)*p - 10.9375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 76.5625*(a*a*a*a)*b*(d*d*d*d)*p + 4.375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 122.5*(a*a*a)*(b*b)*(d*d*d*d)*p + 137.8125*(a*a*a)*(d*d)*(p*p) - 0.4375*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 61.25*(a*a)*(b*b*b)*(d*d*d*d)*p - 344.53125*(a*a)*b*(d*d)*(p*p) + 8.75*a*(b*b*b*b)*(d*d*d*d)*p + 196.875*a*(b*b)*(d*d)*(p*p) + 216.5625*a*(p*p*p) - 0.21875*(b*b*b*b*b)*(d*d*d*d)*p - 24.609375*(b*b*b)*(d*d)*(p*p) - 135.3515625*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU587(double a, double b, double p, double d, double s){
	return ((0.21875*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 2.1875*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 5.46875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 9.84375*(a*a*a*a)*(d*d*d*d)*p - 4.375*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 39.375*(a*a*a)*b*(d*d*d*d)*p + 1.09375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 39.375*(a*a)*(b*b)*(d*d*d*d)*p + 54.140625*(a*a)*(d*d)*(p*p) - 0.0625*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 11.25*a*(b*b*b)*(d*d*d*d)*p - 77.34375*a*b*(d*d)*(p*p) + 0.703125*(b*b*b*b)*(d*d*d*d)*p + 19.3359375*(b*b)*(d*d)*(p*p) + 25.13671875*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU588(double a, double b, double p, double d, double s){
	return (d*(0.21875*(a*a*a*a*a)*(d*d*d*d) - 1.3671875*(a*a*a*a)*b*(d*d*d*d) + 2.1875*(a*a*a)*(b*b)*(d*d*d*d) + 4.921875*(a*a*a)*(d*d)*p - 1.09375*(a*a)*(b*b*b)*(d*d*d*d) - 12.3046875*(a*a)*b*(d*d)*p + 0.15625*a*(b*b*b*b)*(d*d*d*d) + 7.03125*a*(b*b)*(d*d)*p + 11.6015625*a*(p*p) - 0.00390625*(b*b*b*b*b)*(d*d*d*d) - 0.87890625*(b*b*b)*(d*d)*p - 7.2509765625*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU589(double a, double b, double p, double d, double s){
	return ((0.13671875*(a*a*a*a)*(d*d*d*d) - 0.546875*(a*a*a)*b*(d*d*d*d) + 0.546875*(a*a)*(b*b)*(d*d*d*d) + 1.50390625*(a*a)*(d*d)*p - 0.15625*a*(b*b*b)*(d*d*d*d) - 2.1484375*a*b*(d*d)*p + 0.009765625*(b*b*b*b)*(d*d*d*d) + 0.537109375*(b*b)*(d*d)*p + 1.04736328125*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU5810(double a, double b, double p, double d, double s){
	return (d*(0.0546875*(a*a*a)*(d*d) - 0.13671875*(a*a)*b*(d*d) + 0.078125*a*(b*b)*(d*d) + 0.2578125*a*p - 0.009765625*(b*b*b)*(d*d) - 0.1611328125*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU5811(double a, double b, double p, double d, double s){
	return ((0.013671875*(a*a)*(d*d) - 0.01953125*a*b*(d*d) + 0.0048828125*(b*b)*(d*d) + 0.01904296875*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU5812(double a, double b, double p, double d, double s){
	return (d*(0.001953125*a - 0.001220703125*b)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU5813(double a, double b, double p, double d, double s){
	return (0.0001220703125/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU600(double a, double b, double p, double d, double s){
	return (((b*b*b*b*b*b)*(d*d*d*d*d*d) + 7.5*(b*b*b*b)*(d*d*d*d)*p + 11.25*(b*b)*(d*d)*(p*p) + 1.875*(p*p*p))/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU601(double a, double b, double p, double d, double s){
	return (b*d*(-3.0*(b*b*b*b)*(d*d*d*d) - 15.0*(b*b)*(d*d)*p - 11.25*(p*p))/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU602(double a, double b, double p, double d, double s){
	return ((3.75*(b*b*b*b)*(d*d*d*d) + 11.25*(b*b)*(d*d)*p + 2.8125*(p*p))/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU603(double a, double b, double p, double d, double s){
	return (b*d*(-2.5*(b*b)*(d*d) - 3.75*p)/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU604(double a, double b, double p, double d, double s){
	return ((0.9375*(b*b)*(d*d) + 0.46875*p)/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU605(double a, double b, double p, double d, double s){
	return (-0.1875*b*d/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU606(double a, double b, double p, double d, double s){
	return (0.015625/(p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU610(double a, double b, double p, double d, double s){
	return (d*(a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 7.5*a*(b*b*b*b)*(d*d*d*d)*p + 11.25*a*(b*b)*(d*d)*(p*p) + 1.875*a*(p*p*p) - 3.0*(b*b*b*b*b)*(d*d*d*d)*p - 15.0*(b*b*b)*(d*d)*(p*p) - 11.25*b*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU611(double a, double b, double p, double d, double s){
	return ((-3.0*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 15.0*a*(b*b*b)*(d*d*d*d)*p - 11.25*a*b*(d*d)*(p*p) + 0.5*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 11.25*(b*b*b*b)*(d*d*d*d)*p + 28.125*(b*b)*(d*d)*(p*p) + 6.5625*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU612(double a, double b, double p, double d, double s){
	return (d*(3.75*a*(b*b*b*b)*(d*d*d*d) + 11.25*a*(b*b)*(d*d)*p + 2.8125*a*(p*p) - 1.5*(b*b*b*b*b)*(d*d*d*d) - 15.0*(b*b*b)*(d*d)*p - 16.875*b*(p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU613(double a, double b, double p, double d, double s){
	return ((-2.5*a*(b*b*b)*(d*d*d*d) - 3.75*a*b*(d*d)*p + 1.875*(b*b*b*b)*(d*d*d*d) + 9.375*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU614(double a, double b, double p, double d, double s){
	return (d*(0.9375*a*(b*b)*(d*d) + 0.46875*a*p - 1.25*(b*b*b)*(d*d) - 2.8125*b*p)/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU615(double a, double b, double p, double d, double s){
	return ((-0.1875*a*b*(d*d) + 0.46875*(b*b)*(d*d) + 0.328125*p)/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU616(double a, double b, double p, double d, double s){
	return (d*(0.015625*a - 0.09375*b)/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU617(double a, double b, double p, double d, double s){
	return (0.0078125/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU620(double a, double b, double p, double d, double s){
	return (((a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 7.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 11.25*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 1.875*(a*a)*(d*d)*(p*p*p) - 6.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 30.0*a*(b*b*b)*(d*d*d*d)*(p*p) - 22.5*a*b*(d*d)*(p*p*p) + 0.5*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 11.25*(b*b*b*b)*(d*d*d*d)*(p*p) + 28.125*(b*b)*(d*d)*(p*p*p) + 6.5625*(p*p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU621(double a, double b, double p, double d, double s){
	return (d*(-3.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 15.0*(a*a)*(b*b*b)*(d*d*d*d)*p - 11.25*(a*a)*b*(d*d)*(p*p) + a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 22.5*a*(b*b*b*b)*(d*d*d*d)*p + 56.25*a*(b*b)*(d*d)*(p*p) + 13.125*a*(p*p*p) - 4.5*(b*b*b*b*b)*(d*d*d*d)*p - 37.5*(b*b*b)*(d*d)*(p*p) - 39.375*b*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU622(double a, double b, double p, double d, double s){
	return ((3.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 11.25*(a*a)*(b*b)*(d*d*d*d)*p + 2.8125*(a*a)*(d*d)*(p*p) - 3.0*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 30.0*a*(b*b*b)*(d*d*d*d)*p - 33.75*a*b*(d*d)*(p*p) + 0.25*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 11.25*(b*b*b*b)*(d*d*d*d)*p + 42.1875*(b*b)*(d*d)*(p*p) + 13.125*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU623(double a, double b, double p, double d, double s){
	return (d*(-2.5*(a*a)*(b*b*b)*(d*d*d*d) - 3.75*(a*a)*b*(d*d)*p + 3.75*a*(b*b*b*b)*(d*d*d*d) + 18.75*a*(b*b)*(d*d)*p + 6.5625*a*(p*p) - 0.75*(b*b*b*b*b)*(d*d*d*d) - 12.5*(b*b*b)*(d*d)*p - 19.6875*b*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU624(double a, double b, double p, double d, double s){
	return ((0.9375*(a*a)*(b*b)*(d*d*d*d) + 0.46875*(a*a)*(d*d)*p - 2.5*a*(b*b*b)*(d*d*d*d) - 5.625*a*b*(d*d)*p + 0.9375*(b*b*b*b)*(d*d*d*d) + 7.03125*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU625(double a, double b, double p, double d, double s){
	return (d*(-0.1875*(a*a)*b*(d*d) + 0.9375*a*(b*b)*(d*d) + 0.65625*a*p - 0.625*(b*b*b)*(d*d) - 1.96875*b*p)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU626(double a, double b, double p, double d, double s){
	return ((0.015625*(a*a)*(d*d) - 0.1875*a*b*(d*d) + 0.234375*(b*b)*(d*d) + 0.21875*p)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU627(double a, double b, double p, double d, double s){
	return (d*(0.015625*a - 0.046875*b)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU628(double a, double b, double p, double d, double s){
	return (0.00390625/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU630(double a, double b, double p, double d, double s){
	return (d*((a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 7.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 11.25*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 1.875*(a*a*a)*(d*d)*(p*p*p) - 9.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 45.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 33.75*(a*a)*b*(d*d)*(p*p*p) + 1.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 33.75*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 84.375*a*(b*b)*(d*d)*(p*p*p) + 19.6875*a*(p*p*p*p) - 4.5*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 37.5*(b*b*b)*(d*d)*(p*p*p) - 39.375*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU631(double a, double b, double p, double d, double s){
	return ((-3.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 15.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 11.25*(a*a*a)*b*(d*d*d*d)*(p*p) + 1.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 33.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 84.375*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 19.6875*(a*a)*(d*d)*(p*p*p) - 13.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 112.5*a*(b*b*b)*(d*d*d*d)*(p*p) - 118.125*a*b*(d*d)*(p*p*p) + 0.75*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 28.125*(b*b*b*b)*(d*d*d*d)*(p*p) + 98.4375*(b*b)*(d*d)*(p*p*p) + 29.53125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU632(double a, double b, double p, double d, double s){
	return (d*(3.75*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 11.25*(a*a*a)*(b*b)*(d*d*d*d)*p + 2.8125*(a*a*a)*(d*d)*(p*p) - 4.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 45.0*(a*a)*(b*b*b)*(d*d*d*d)*p - 50.625*(a*a)*b*(d*d)*(p*p) + 0.75*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 33.75*a*(b*b*b*b)*(d*d*d*d)*p + 126.5625*a*(b*b)*(d*d)*(p*p) + 39.375*a*(p*p*p) - 4.5*(b*b*b*b*b)*(d*d*d*d)*p - 56.25*(b*b*b)*(d*d)*(p*p) - 78.75*b*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU633(double a, double b, double p, double d, double s){
	return ((-2.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 3.75*(a*a*a)*b*(d*d*d*d)*p + 5.625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 28.125*(a*a)*(b*b)*(d*d*d*d)*p + 9.84375*(a*a)*(d*d)*(p*p) - 2.25*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 37.5*a*(b*b*b)*(d*d*d*d)*p - 59.0625*a*b*(d*d)*(p*p) + 0.125*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 9.375*(b*b*b*b)*(d*d*d*d)*p + 49.21875*(b*b)*(d*d)*(p*p) + 19.6875*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU634(double a, double b, double p, double d, double s){
	return (d*(0.9375*(a*a*a)*(b*b)*(d*d*d*d) + 0.46875*(a*a*a)*(d*d)*p - 3.75*(a*a)*(b*b*b)*(d*d*d*d) - 8.4375*(a*a)*b*(d*d)*p + 2.8125*a*(b*b*b*b)*(d*d*d*d) + 21.09375*a*(b*b)*(d*d)*p + 9.84375*a*(p*p) - 0.375*(b*b*b*b*b)*(d*d*d*d) - 9.375*(b*b*b)*(d*d)*p - 19.6875*b*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU635(double a, double b, double p, double d, double s){
	return ((-0.1875*(a*a*a)*b*(d*d*d*d) + 1.40625*(a*a)*(b*b)*(d*d*d*d) + 0.984375*(a*a)*(d*d)*p - 1.875*a*(b*b*b)*(d*d*d*d) - 5.90625*a*b*(d*d)*p + 0.46875*(b*b*b*b)*(d*d*d*d) + 4.921875*(b*b)*(d*d)*p + 2.953125*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU636(double a, double b, double p, double d, double s){
	return (d*(0.015625*(a*a*a)*(d*d) - 0.28125*(a*a)*b*(d*d) + 0.703125*a*(b*b)*(d*d) + 0.65625*a*p - 0.3125*(b*b*b)*(d*d) - 1.3125*b*p)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU637(double a, double b, double p, double d, double s){
	return ((0.0234375*(a*a)*(d*d) - 0.140625*a*b*(d*d) + 0.1171875*(b*b)*(d*d) + 0.140625*p)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU638(double a, double b, double p, double d, double s){
	return (d*(0.01171875*a - 0.0234375*b)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU639(double a, double b, double p, double d, double s){
	return (0.001953125/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU640(double a, double b, double p, double d, double s){
	return (((a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 7.5*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 11.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 1.875*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 12.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 60.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 45.0*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 3.0*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 67.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 168.75*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 39.375*(a*a)*(d*d)*(p*p*p*p) - 18.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 150.0*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 157.5*a*b*(d*d)*(p*p*p*p) + 0.75*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 28.125*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 98.4375*(b*b)*(d*d)*(p*p*p*p) + 29.53125*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU641(double a, double b, double p, double d, double s){
	return (d*(-3.0*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 15.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 11.25*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 2.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 45.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 112.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 26.25*(a*a*a)*(d*d)*(p*p*p) - 27.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 225.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 236.25*(a*a)*b*(d*d)*(p*p*p) + 3.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 112.5*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 393.75*a*(b*b)*(d*d)*(p*p*p) + 118.125*a*(p*p*p*p) - 11.25*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 131.25*(b*b*b)*(d*d)*(p*p*p) - 177.1875*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU642(double a, double b, double p, double d, double s){
	return ((3.75*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 11.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 2.8125*(a*a*a*a)*(d*d*d*d)*(p*p) - 6.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 60.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 67.5*(a*a*a)*b*(d*d*d*d)*(p*p) + 1.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 67.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 253.125*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 78.75*(a*a)*(d*d)*(p*p*p) - 18.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 225.0*a*(b*b*b)*(d*d*d*d)*(p*p) - 315.0*a*b*(d*d)*(p*p*p) + 0.75*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 42.1875*(b*b*b*b)*(d*d*d*d)*(p*p) + 196.875*(b*b)*(d*d)*(p*p*p) + 73.828125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU643(double a, double b, double p, double d, double s){
	return (d*(-2.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 3.75*(a*a*a*a)*b*(d*d*d*d)*p + 7.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 37.5*(a*a*a)*(b*b)*(d*d*d*d)*p + 13.125*(a*a*a)*(d*d)*(p*p) - 4.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 75.0*(a*a)*(b*b*b)*(d*d*d*d)*p - 118.125*(a*a)*b*(d*d)*(p*p) + 0.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 37.5*a*(b*b*b*b)*(d*d*d*d)*p + 196.875*a*(b*b)*(d*d)*(p*p) + 78.75*a*(p*p*p) - 3.75*(b*b*b*b*b)*(d*d*d*d)*p - 65.625*(b*b*b)*(d*d)*(p*p) - 118.125*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU644(double a, double b, double p, double d, double s){
	return ((0.9375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.46875*(a*a*a*a)*(d*d*d*d)*p - 5.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 11.25*(a*a*a)*b*(d*d*d*d)*p + 5.625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 42.1875*(a*a)*(b*b)*(d*d*d*d)*p + 19.6875*(a*a)*(d*d)*(p*p) - 1.5*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 37.5*a*(b*b*b)*(d*d*d*d)*p - 78.75*a*b*(d*d)*(p*p) + 0.0625*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 7.03125*(b*b*b*b)*(d*d*d*d)*p + 49.21875*(b*b)*(d*d)*(p*p) + 24.609375*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU645(double a, double b, double p, double d, double s){
	return (d*(-0.1875*(a*a*a*a)*b*(d*d*d*d) + 1.875*(a*a*a)*(b*b)*(d*d*d*d) + 1.3125*(a*a*a)*(d*d)*p - 3.75*(a*a)*(b*b*b)*(d*d*d*d) - 11.8125*(a*a)*b*(d*d)*p + 1.875*a*(b*b*b*b)*(d*d*d*d) + 19.6875*a*(b*b)*(d*d)*p + 11.8125*a*(p*p) - 0.1875*(b*b*b*b*b)*(d*d*d*d) - 6.5625*(b*b*b)*(d*d)*p - 17.71875*b*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU646(double a, double b, double p, double d, double s){
	return ((0.015625*(a*a*a*a)*(d*d*d*d) - 0.375*(a*a*a)*b*(d*d*d*d) + 1.40625*(a*a)*(b*b)*(d*d*d*d) + 1.3125*(a*a)*(d*d)*p - 1.25*a*(b*b*b)*(d*d*d*d) - 5.25*a*b*(d*d)*p + 0.234375*(b*b*b*b)*(d*d*d*d) + 3.28125*(b*b)*(d*d)*p + 2.4609375*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU647(double a, double b, double p, double d, double s){
	return (d*(0.03125*(a*a*a)*(d*d) - 0.28125*(a*a)*b*(d*d) + 0.46875*a*(b*b)*(d*d) + 0.5625*a*p - 0.15625*(b*b*b)*(d*d) - 0.84375*b*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU648(double a, double b, double p, double d, double s){
	return ((0.0234375*(a*a)*(d*d) - 0.09375*a*b*(d*d) + 0.05859375*(b*b)*(d*d) + 0.087890625*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU649(double a, double b, double p, double d, double s){
	return (d*(0.0078125*a - 0.01171875*b)/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU6410(double a, double b, double p, double d, double s){
	return (0.0009765625/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU650(double a, double b, double p, double d, double s){
	return (d*((a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 7.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 11.25*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 1.875*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 15.0*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 75.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 56.25*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 5.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 112.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 281.25*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 65.625*(a*a*a)*(d*d)*(p*p*p*p) - 45.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 375.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 393.75*(a*a)*b*(d*d)*(p*p*p*p) + 3.75*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 140.625*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 492.1875*a*(b*b)*(d*d)*(p*p*p*p) + 147.65625*a*(p*p*p*p*p) - 11.25*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 131.25*(b*b*b)*(d*d)*(p*p*p*p) - 177.1875*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU651(double a, double b, double p, double d, double s){
	return ((-3.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 15.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 11.25*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 2.5*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 56.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 140.625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 32.8125*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 45.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 375.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 393.75*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 7.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 281.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 984.375*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 295.3125*(a*a)*(d*d)*(p*p*p*p) - 56.25*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 656.25*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 885.9375*a*b*(d*d)*(p*p*p*p) + 1.875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 98.4375*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 442.96875*(b*b)*(d*d)*(p*p*p*p) + 162.421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU652(double a, double b, double p, double d, double s){
	return (d*(3.75*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 11.25*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 2.8125*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 7.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 75.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 84.375*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 2.5*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 112.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 421.875*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 131.25*(a*a*a)*(d*d)*(p*p*p) - 45.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 562.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 787.5*(a*a)*b*(d*d)*(p*p*p) + 3.75*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 210.9375*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 984.375*a*(b*b)*(d*d)*(p*p*p) + 369.140625*a*(p*p*p*p) - 16.875*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 262.5*(b*b*b)*(d*d)*(p*p*p) - 442.96875*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU653(double a, double b, double p, double d, double s){
	return ((-2.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 3.75*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 9.375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 46.875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 16.40625*(a*a*a*a)*(d*d*d*d)*(p*p) - 7.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 125.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 196.875*(a*a*a)*b*(d*d*d*d)*(p*p) + 1.25*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 93.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 492.1875*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 196.875*(a*a)*(d*d)*(p*p*p) - 18.75*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 328.125*a*(b*b*b)*(d*d*d*d)*(p*p) - 590.625*a*b*(d*d)*(p*p*p) + 0.625*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 49.21875*(b*b*b*b)*(d*d*d*d)*(p*p) + 295.3125*(b*b)*(d*d)*(p*p*p) + 135.3515625*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU654(double a, double b, double p, double d, double s){
	return (d*(0.9375*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.46875*(a*a*a*a*a)*(d*d*d*d)*p - 6.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 14.0625*(a*a*a*a)*b*(d*d*d*d)*p + 9.375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 70.3125*(a*a*a)*(b*b)*(d*d*d*d)*p + 32.8125*(a*a*a)*(d*d)*(p*p) - 3.75*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 93.75*(a*a)*(b*b*b)*(d*d*d*d)*p - 196.875*(a*a)*b*(d*d)*(p*p) + 0.3125*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 35.15625*a*(b*b*b*b)*(d*d*d*d)*p + 246.09375*a*(b*b)*(d*d)*(p*p) + 123.046875*a*(p*p*p) - 2.8125*(b*b*b*b*b)*(d*d*d*d)*p - 65.625*(b*b*b)*(d*d)*(p*p) - 147.65625*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU655(double a, double b, double p, double d, double s){
	return ((-0.1875*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 2.34375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 1.640625*(a*a*a*a)*(d*d*d*d)*p - 6.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 19.6875*(a*a*a)*b*(d*d*d*d)*p + 4.6875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 49.21875*(a*a)*(b*b)*(d*d*d*d)*p + 29.53125*(a*a)*(d*d)*(p*p) - 0.9375*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 32.8125*a*(b*b*b)*(d*d*d*d)*p - 88.59375*a*b*(d*d)*(p*p) + 0.03125*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 4.921875*(b*b*b*b)*(d*d*d*d)*p + 44.296875*(b*b)*(d*d)*(p*p) + 27.0703125*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU656(double a, double b, double p, double d, double s){
	return (d*(0.015625*(a*a*a*a*a)*(d*d*d*d) - 0.46875*(a*a*a*a)*b*(d*d*d*d) + 2.34375*(a*a*a)*(b*b)*(d*d*d*d) + 2.1875*(a*a*a)*(d*d)*p - 3.125*(a*a)*(b*b*b)*(d*d*d*d) - 13.125*(a*a)*b*(d*d)*p + 1.171875*a*(b*b*b*b)*(d*d*d*d) + 16.40625*a*(b*b)*(d*d)*p + 12.3046875*a*(p*p) - 0.09375*(b*b*b*b*b)*(d*d*d*d) - 4.375*(b*b*b)*(d*d)*p - 14.765625*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU657(double a, double b, double p, double d, double s){
	return ((0.0390625*(a*a*a*a)*(d*d*d*d) - 0.46875*(a*a*a)*b*(d*d*d*d) + 1.171875*(a*a)*(b*b)*(d*d*d*d) + 1.40625*(a*a)*(d*d)*p - 0.78125*a*(b*b*b)*(d*d*d*d) - 4.21875*a*b*(d*d)*p + 0.1171875*(b*b*b*b)*(d*d*d*d) + 2.109375*(b*b)*(d*d)*p + 1.93359375*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU658(double a, double b, double p, double d, double s){
	return (d*(0.0390625*(a*a*a)*(d*d) - 0.234375*(a*a)*b*(d*d) + 0.29296875*a*(b*b)*(d*d) + 0.439453125*a*p - 0.078125*(b*b*b)*(d*d) - 0.52734375*b*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU659(double a, double b, double p, double d, double s){
	return ((0.01953125*(a*a)*(d*d) - 0.05859375*a*b*(d*d) + 0.029296875*(b*b)*(d*d) + 0.0537109375*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU6510(double a, double b, double p, double d, double s){
	return (d*(0.0048828125*a - 0.005859375*b)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU6511(double a, double b, double p, double d, double s){
	return (0.00048828125/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU660(double a, double b, double p, double d, double s){
	return (((a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 7.5*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 11.25*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 1.875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 18.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 90.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 67.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 7.5*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 168.75*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 421.875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 98.4375*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 90.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 750.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 787.5*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 11.25*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 421.875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 1476.5625*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 442.96875*(a*a)*(d*d)*(p*p*p*p*p) - 67.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 787.5*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 1063.125*a*b*(d*d)*(p*p*p*p*p) + 1.875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 98.4375*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 442.96875*(b*b)*(d*d)*(p*p*p*p*p) + 162.421875*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU661(double a, double b, double p, double d, double s){
	return (d*(-3.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 15.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 11.25*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 3.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 67.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 168.75*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 39.375*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 67.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 562.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 590.625*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 15.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 562.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 1968.75*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 590.625*(a*a*a)*(d*d)*(p*p*p*p) - 168.75*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 1968.75*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 2657.8125*(a*a)*b*(d*d)*(p*p*p*p) + 11.25*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 590.625*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 2657.8125*a*(b*b)*(d*d)*(p*p*p*p) + 974.53125*a*(p*p*p*p*p) - 39.375*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 590.625*(b*b*b)*(d*d)*(p*p*p*p) - 974.53125*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU662(double a, double b, double p, double d, double s){
	return ((3.75*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 11.25*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 2.8125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 9.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 90.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 101.25*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 3.75*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 168.75*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 632.8125*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 196.875*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 90.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1125.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 1575.0*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 11.25*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 632.8125*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 2953.125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 1107.421875*(a*a)*(d*d)*(p*p*p*p) - 101.25*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 1575.0*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 2657.8125*a*b*(d*d)*(p*p*p*p) + 2.8125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 196.875*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 1107.421875*(b*b)*(d*d)*(p*p*p*p) + 487.265625*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU663(double a, double b, double p, double d, double s){
	return (d*(-2.5*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 3.75*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 11.25*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 56.25*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 19.6875*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 11.25*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 187.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 295.3125*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 2.5*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 187.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 984.375*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 393.75*(a*a*a)*(d*d)*(p*p*p) - 56.25*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 984.375*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 1771.875*(a*a)*b*(d*d)*(p*p*p) + 3.75*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 295.3125*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 1771.875*a*(b*b)*(d*d)*(p*p*p) + 812.109375*a*(p*p*p*p) - 19.6875*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 393.75*(b*b*b)*(d*d)*(p*p*p) - 812.109375*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU664(double a, double b, double p, double d, double s){
	return ((0.9375*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.46875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 7.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 16.875*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 14.0625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 105.46875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 49.21875*(a*a*a*a)*(d*d*d*d)*(p*p) - 7.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 187.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 393.75*(a*a*a)*b*(d*d*d*d)*(p*p) + 0.9375*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 105.46875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 738.28125*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 369.140625*(a*a)*(d*d)*(p*p*p) - 16.875*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 393.75*a*(b*b*b)*(d*d*d*d)*(p*p) - 885.9375*a*b*(d*d)*(p*p*p) + 0.46875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 49.21875*(b*b*b*b)*(d*d*d*d)*(p*p) + 369.140625*(b*b)*(d*d)*(p*p*p) + 203.02734375*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU665(double a, double b, double p, double d, double s){
	return (d*(-0.1875*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 2.8125*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 1.96875*(a*a*a*a*a)*(d*d*d*d)*p - 9.375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 29.53125*(a*a*a*a)*b*(d*d*d*d)*p + 9.375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 98.4375*(a*a*a)*(b*b)*(d*d*d*d)*p + 59.0625*(a*a*a)*(d*d)*(p*p) - 2.8125*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 98.4375*(a*a)*(b*b*b)*(d*d*d*d)*p - 265.78125*(a*a)*b*(d*d)*(p*p) + 0.1875*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 29.53125*a*(b*b*b*b)*(d*d*d*d)*p + 265.78125*a*(b*b)*(d*d)*(p*p) + 162.421875*a*(p*p*p) - 1.96875*(b*b*b*b*b)*(d*d*d*d)*p - 59.0625*(b*b*b)*(d*d)*(p*p) - 162.421875*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU666(double a, double b, double p, double d, double s){
	return ((0.015625*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.5625*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 3.515625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 3.28125*(a*a*a*a)*(d*d*d*d)*p - 6.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 26.25*(a*a*a)*b*(d*d*d*d)*p + 3.515625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 49.21875*(a*a)*(b*b)*(d*d*d*d)*p + 36.9140625*(a*a)*(d*d)*(p*p) - 0.5625*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 26.25*a*(b*b*b)*(d*d*d*d)*p - 88.59375*a*b*(d*d)*(p*p) + 0.015625*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 3.28125*(b*b*b*b)*(d*d*d*d)*p + 36.9140625*(b*b)*(d*d)*(p*p) + 27.0703125*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU667(double a, double b, double p, double d, double s){
	return (d*(0.046875*(a*a*a*a*a)*(d*d*d*d) - 0.703125*(a*a*a*a)*b*(d*d*d*d) + 2.34375*(a*a*a)*(b*b)*(d*d*d*d) + 2.8125*(a*a*a)*(d*d)*p - 2.34375*(a*a)*(b*b*b)*(d*d*d*d) - 12.65625*(a*a)*b*(d*d)*p + 0.703125*a*(b*b*b*b)*(d*d*d*d) + 12.65625*a*(b*b)*(d*d)*p + 11.6015625*a*(p*p) - 0.046875*(b*b*b*b*b)*(d*d*d*d) - 2.8125*(b*b*b)*(d*d)*p - 11.6015625*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU668(double a, double b, double p, double d, double s){
	return ((0.05859375*(a*a*a*a)*(d*d*d*d) - 0.46875*(a*a*a)*b*(d*d*d*d) + 0.87890625*(a*a)*(b*b)*(d*d*d*d) + 1.318359375*(a*a)*(d*d)*p - 0.46875*a*(b*b*b)*(d*d*d*d) - 3.1640625*a*b*(d*d)*p + 0.05859375*(b*b*b*b)*(d*d*d*d) + 1.318359375*(b*b)*(d*d)*p + 1.4501953125*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU669(double a, double b, double p, double d, double s){
	return (d*(0.0390625*(a*a*a)*(d*d) - 0.17578125*(a*a)*b*(d*d) + 0.17578125*a*(b*b)*(d*d) + 0.322265625*a*p - 0.0390625*(b*b*b)*(d*d) - 0.322265625*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU6610(double a, double b, double p, double d, double s){
	return ((0.0146484375*(a*a)*(d*d) - 0.03515625*a*b*(d*d) + 0.0146484375*(b*b)*(d*d) + 0.0322265625*p)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU6611(double a, double b, double p, double d, double s){
	return (0.0029296875*d*(a - b)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU6612(double a, double b, double p, double d, double s){
	return (0.000244140625/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU670(double a, double b, double p, double d, double s){
	return (d*((a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 7.5*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 11.25*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 1.875*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 21.0 *(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 105.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 78.75*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 10.5*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 236.25*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 590.625*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 137.8125*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 157.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 1312.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 1378.125*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 26.25*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 984.375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 3445.3125*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 1033.59375*(a*a*a)*(d*d)*(p*p*p*p*p) - 236.25*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 2756.25*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 3720.9375*(a*a)*b*(d*d)*(p*p*p*p*p) + 13.125*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 689.0625*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 3100.78125*a*(b*b)*(d*d)*(p*p*p*p*p) + 1136.953125*a*(p*p*p*p*p*p) - 39.375*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p) - 590.625*(b*b*b)*(d*d)*(p*p*p*p*p) - 974.53125*b*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU671(double a, double b, double p, double d, double s){
	return ((-3.0*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 15.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 11.25*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 3.5*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 78.75*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 196.875*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 45.9375*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 94.5*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 787.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 826.875*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 26.25*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 984.375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 3445.3125*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 1033.59375*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 393.75*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 4593.75*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 6201.5625*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 39.375*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 2067.1875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 9302.34375*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 3410.859375*(a*a)*(d*d)*(p*p*p*p*p) - 275.625*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 4134.375*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 6821.71875*a*b*(d*d)*(p*p*p*p*p) + 6.5625*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 442.96875*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 2436.328125*(b*b)*(d*d)*(p*p*p*p*p) + 1055.7421875*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU672(double a, double b, double p, double d, double s){
	return (d*(3.75*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 11.25*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 2.8125*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 10.5*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 105.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 118.125*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 5.25*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 236.25*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 885.9375*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 275.625*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 157.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1968.75*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 2756.25*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 26.25*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1476.5625*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 6890.625*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 2583.984375*(a*a*a)*(d*d)*(p*p*p*p) - 354.375*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 5512.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 9302.34375*(a*a)*b*(d*d)*(p*p*p*p) + 19.6875*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 1378.125*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 7751.953125*a*(b*b)*(d*d)*(p*p*p*p) + 3410.859375*a*(p*p*p*p*p) - 78.75*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 1476.5625*(b*b*b)*(d*d)*(p*p*p*p) - 2923.59375*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU673(double a, double b, double p, double d, double s){
	return ((-2.5*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 3.75*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 13.125*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 65.625*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 22.96875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 15.75*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 262.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 413.4375*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 4.375*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 328.125*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1722.65625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 689.0625*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 131.25*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 2296.875*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 4134.375*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 13.125*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1033.59375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 6201.5625*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 2842.3828125*(a*a)*(d*d)*(p*p*p*p) - 137.8125*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 2756.25*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 5684.765625*a*b*(d*d)*(p*p*p*p) + 3.28125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 295.3125*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 2030.2734375*(b*b)*(d*d)*(p*p*p*p) + 1055.7421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU674(double a, double b, double p, double d, double s){
	return (d*(0.9375*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.46875*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 8.75*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 19.6875*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 19.6875*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 147.65625*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 68.90625*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 13.125*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 328.125*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 689.0625*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 2.1875*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 246.09375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1722.65625*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 861.328125*(a*a*a)*(d*d)*(p*p*p) - 59.0625*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 1378.125*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 3100.78125*(a*a)*b*(d*d)*(p*p*p) + 3.28125*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 344.53125*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 2583.984375*a*(b*b)*(d*d)*(p*p*p) + 1421.19140625*a*(p*p*p*p) - 19.6875*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 492.1875*(b*b*b)*(d*d)*(p*p*p) - 1218.1640625*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU675(double a, double b, double p, double d, double s){
	return ((-0.1875*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 3.28125*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 2.296875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 13.125*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 41.34375*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 16.40625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 172.265625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 103.359375*(a*a*a*a)*(d*d*d*d)*(p*p) - 6.5625*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 229.6875*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 620.15625*(a*a*a)*b*(d*d*d*d)*(p*p) + 0.65625*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 103.359375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 930.234375*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 568.4765625*(a*a)*(d*d)*(p*p*p) - 13.78125*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 413.4375*a*(b*b*b)*(d*d*d*d)*(p*p) - 1136.953125*a*b*(d*d)*(p*p*p) + 0.328125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 44.296875*(b*b*b*b)*(d*d*d*d)*(p*p) + 406.0546875*(b*b)*(d*d)*(p*p*p) + 263.935546875*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU676(double a, double b, double p, double d, double s){
	return (d*(0.015625*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.65625*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 4.921875*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 4.59375*(a*a*a*a*a)*(d*d*d*d)*p - 10.9375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 45.9375*(a*a*a*a)*b*(d*d*d*d)*p + 8.203125*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 114.84375*(a*a*a)*(b*b)*(d*d*d*d)*p + 86.1328125*(a*a*a)*(d*d)*(p*p) - 1.96875*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 91.875*(a*a)*(b*b*b)*(d*d*d*d)*p - 310.078125*(a*a)*b*(d*d)*(p*p) + 0.109375*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 22.96875*a*(b*b*b*b)*(d*d*d*d)*p + 258.3984375*a*(b*b)*(d*d)*(p*p) + 189.4921875*a*(p*p*p) - 1.3125*(b*b*b*b*b)*(d*d*d*d)*p - 49.21875*(b*b*b)*(d*d)*(p*p) - 162.421875*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU677(double a, double b, double p, double d, double s){
	return ((0.0546875*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.984375*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 4.1015625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 4.921875*(a*a*a*a)*(d*d*d*d)*p - 5.46875*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 29.53125*(a*a*a)*b*(d*d*d*d)*p + 2.4609375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 44.296875*(a*a)*(b*b)*(d*d*d*d)*p + 40.60546875*(a*a)*(d*d)*(p*p) - 0.328125*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 19.6875*a*(b*b*b)*(d*d*d*d)*p - 81.2109375*a*b*(d*d)*(p*p) + 0.0078125*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 2.109375*(b*b*b*b)*(d*d*d*d)*p + 29.00390625*(b*b)*(d*d)*(p*p) + 25.13671875*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU678(double a, double b, double p, double d, double s){
	return (d*(0.08203125*(a*a*a*a*a)*(d*d*d*d) - 0.8203125*(a*a*a*a)*b*(d*d*d*d) + 2.05078125*(a*a*a)*(b*b)*(d*d*d*d) + 3.076171875*(a*a*a)*(d*d)*p - 1.640625*(a*a)*(b*b*b)*(d*d*d*d) - 11.07421875*(a*a)*b*(d*d)*p + 0.41015625*a*(b*b*b*b)*(d*d*d*d) + 9.228515625*a*(b*b)*(d*d)*p + 10.1513671875*a*(p*p) - 0.0234375*(b*b*b*b*b)*(d*d*d*d) - 1.7578125*(b*b*b)*(d*d)*p - 8.701171875*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU679(double a, double b, double p, double d, double s){
	return ((0.068359375*(a*a*a*a)*(d*d*d*d) - 0.41015625*(a*a*a)*b*(d*d*d*d) + 0.615234375*(a*a)*(b*b)*(d*d*d*d) + 1.1279296875*(a*a)*(d*d)*p - 0.2734375*a*(b*b*b)*(d*d*d*d) - 2.255859375*a*b*(d*d)*p + 0.029296875*(b*b*b*b)*(d*d*d*d) + 0.8056640625*(b*b)*(d*d)*p + 1.04736328125*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU6710(double a, double b, double p, double d, double s){
	return (d*(0.0341796875*(a*a*a)*(d*d) - 0.123046875*(a*a)*b*(d*d) + 0.1025390625*a*(b*b)*(d*d) + 0.2255859375*a*p - 0.01953125*(b*b*b)*(d*d) - 0.193359375*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU6711(double a, double b, double p, double d, double s){
	return ((0.01025390625*(a*a)*(d*d) - 0.0205078125*a*b*(d*d) + 0.00732421875*(b*b)*(d*d) + 0.01904296875*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU6712(double a, double b, double p, double d, double s){
	return (d*(0.001708984375*a - 0.00146484375*b)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU6713(double a, double b, double p, double d, double s){
	return (0.0001220703125/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU680(double a, double b, double p, double d, double s){
	return (((a*a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) + 7.5*(a*a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 11.25*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 1.875*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*(p*p*p) - 24.0*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 120.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 90.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p*p) + 14.0*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 315.0*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 787.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 183.75*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p*p) - 252.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 2100.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 2205.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p*p) + 52.5*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 1968.75*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 6890.625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 2067.1875*(a*a*a*a)*(d*d*d*d)*(p*p*p*p*p) - 630.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 7350.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 9922.5*(a*a*a)*b*(d*d*d*d)*(p*p*p*p*p) + 52.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 2756.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 12403.125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p*p) + 4547.8125*(a*a)*(d*d)*(p*p*p*p*p*p) - 315.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 4725.0*a*(b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 7796.25*a*b*(d*d)*(p*p*p*p*p*p) + 6.5625*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 442.96875*(b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) + 2436.328125*(b*b)*(d*d)*(p*p*p*p*p*p) + 1055.7421875*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU681(double a, double b, double p, double d, double s){
	return (d*(-3.0*(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 15.0*(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 11.25*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 4.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 90.0*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 225.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 52.5*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 126.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 1050.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 1102.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 42.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 1575.0*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 5512.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 1653.75*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 787.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 9187.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 12403.125*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 105.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 5512.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 24806.25*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 9095.625*(a*a*a)*(d*d)*(p*p*p*p*p) - 1102.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 16537.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 27286.875*(a*a)*b*(d*d)*(p*p*p*p*p) + 52.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 3543.75*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 19490.625*a*(b*b)*(d*d)*(p*p*p*p*p) + 8445.9375*a*(p*p*p*p*p*p) - 177.1875*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p) - 3248.4375*(b*b*b)*(d*d)*(p*p*p*p*p) - 6334.453125*b*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU682(double a, double b, double p, double d, double s){
	return ((3.75*(a*a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 11.25*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 2.8125*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*(p*p) - 12.0*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 120.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 135.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 7.0*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 315.0*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 1181.25*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 367.5*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 252.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 3150.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 4410.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 52.5*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 2953.125*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 13781.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 5167.96875*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 945.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 14700.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 24806.25*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 78.75*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 5512.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 31007.8125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 13643.4375*(a*a)*(d*d)*(p*p*p*p*p) - 630.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 11812.5*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 23388.75*a*b*(d*d)*(p*p*p*p*p) + 13.125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 1107.421875*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 7308.984375*(b*b)*(d*d)*(p*p*p*p*p) + 3695.09765625*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU683(double a, double b, double p, double d, double s){
	return (d*(-2.5*(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 3.75*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 15.0*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 75.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 26.25*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 21.0 *(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 350.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 551.25*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 7.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 525.0*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 2756.25*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 1102.5*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 262.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 4593.75*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 8268.75*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 35.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 2756.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 16537.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 7579.6875*(a*a*a)*(d*d)*(p*p*p*p) - 551.25*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 11025.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 22739.0625*(a*a)*b*(d*d)*(p*p*p*p) + 26.25*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 2362.5*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 16242.1875*a*(b*b)*(d*d)*(p*p*p*p) + 8445.9375*a*(p*p*p*p*p) - 118.125*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 2707.03125*(b*b*b)*(d*d)*(p*p*p*p) - 6334.453125*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU684(double a, double b, double p, double d, double s){
	return ((0.9375*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d) + 0.46875*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*p - 10.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 22.5*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 26.25*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 196.875*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 91.875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 21.0 *(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 525.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 1102.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 4.375*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 492.1875*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 3445.3125*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 1722.65625*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 157.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 3675.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 8268.75*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 13.125*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1378.125*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 10335.9375*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 5684.765625*(a*a)*(d*d)*(p*p*p*p) - 157.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 3937.5*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 9745.3125*a*b*(d*d)*(p*p*p*p) + 3.28125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 369.140625*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 3045.41015625*(b*b)*(d*d)*(p*p*p*p) + 1847.548828125*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU685(double a, double b, double p, double d, double s){
	return (d*(-0.1875*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 3.75*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 2.625*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 17.5*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 55.125*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 26.25*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 275.625*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 165.375*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 13.125*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 459.375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 1240.3125*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 1.75*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 275.625*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 2480.625*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 1515.9375*(a*a*a)*(d*d)*(p*p*p) - 55.125*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 1653.75*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 4547.8125*(a*a)*b*(d*d)*(p*p*p) + 2.625*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 354.375*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 3248.4375*a*(b*b)*(d*d)*(p*p*p) + 2111.484375*a*(p*p*p*p) - 17.71875*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 541.40625*(b*b*b)*(d*d)*(p*p*p) - 1583.61328125*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU686(double a, double b, double p, double d, double s){
	return ((0.015625*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d) - 0.75*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 6.5625*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 6.125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 17.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 73.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 16.40625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 229.6875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 172.265625*(a*a*a*a)*(d*d*d*d)*(p*p) - 5.25*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 245.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 826.875*(a*a*a)*b*(d*d*d*d)*(p*p) + 0.4375*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 91.875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1033.59375*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 757.96875*(a*a)*(d*d)*(p*p*p) - 10.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 393.75*a*(b*b*b)*(d*d*d*d)*(p*p) - 1299.375*a*b*(d*d)*(p*p*p) + 0.21875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 36.9140625*(b*b*b*b)*(d*d*d*d)*(p*p) + 406.0546875*(b*b)*(d*d)*(p*p*p) + 307.9248046875*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU687(double a, double b, double p, double d, double s){
	return (d*(0.0625*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 1.3125*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 6.5625*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 7.875*(a*a*a*a*a)*(d*d*d*d)*p - 10.9375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 59.0625*(a*a*a*a)*b*(d*d*d*d)*p + 6.5625*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 118.125*(a*a*a)*(b*b)*(d*d*d*d)*p + 108.28125*(a*a*a)*(d*d)*(p*p) - 1.3125*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 78.75*(a*a)*(b*b*b)*(d*d*d*d)*p - 324.84375*(a*a)*b*(d*d)*(p*p) + 0.0625*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 16.875*a*(b*b*b*b)*(d*d*d*d)*p + 232.03125*a*(b*b)*(d*d)*(p*p) + 201.09375*a*(p*p*p) - 0.84375*(b*b*b*b*b)*(d*d*d*d)*p - 38.671875*(b*b*b)*(d*d)*(p*p) - 150.8203125*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU688(double a, double b, double p, double d, double s){
	return ((0.109375*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 1.3125*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 4.1015625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 6.15234375*(a*a*a*a)*(d*d*d*d)*p - 4.375*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 29.53125*(a*a*a)*b*(d*d*d*d)*p + 1.640625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 36.9140625*(a*a)*(b*b)*(d*d*d*d)*p + 40.60546875*(a*a)*(d*d)*(p*p) - 0.1875*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 14.0625*a*(b*b*b)*(d*d*d*d)*p - 69.609375*a*b*(d*d)*(p*p) + 0.00390625*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 1.318359375*(b*b*b*b)*(d*d*d*d)*p + 21.7529296875*(b*b)*(d*d)*(p*p) + 21.99462890625*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU689(double a, double b, double p, double d, double s){
	return (d*(0.109375*(a*a*a*a*a)*(d*d*d*d) - 0.8203125*(a*a*a*a)*b*(d*d*d*d) + 1.640625*(a*a*a)*(b*b)*(d*d*d*d) + 3.0078125*(a*a*a)*(d*d)*p - 1.09375*(a*a)*(b*b*b)*(d*d*d*d) - 9.0234375*(a*a)*b*(d*d)*p + 0.234375*a*(b*b*b*b)*(d*d*d*d) + 6.4453125*a*(b*b)*(d*d)*p + 8.37890625*a*(p*p) - 0.01171875*(b*b*b*b*b)*(d*d*d*d) - 1.07421875*(b*b*b)*(d*d)*p - 6.2841796875*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU6810(double a, double b, double p, double d, double s){
	return ((0.068359375*(a*a*a*a)*(d*d*d*d) - 0.328125*(a*a*a)*b*(d*d*d*d) + 0.41015625*(a*a)*(b*b)*(d*d*d*d) + 0.90234375*(a*a)*(d*d)*p - 0.15625*a*(b*b*b)*(d*d*d*d) - 1.546875*a*b*(d*d)*p + 0.0146484375*(b*b*b*b)*(d*d*d*d) + 0.4833984375*(b*b)*(d*d)*p + 0.733154296875*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU6811(double a, double b, double p, double d, double s){
	return (d*(0.02734375*(a*a*a)*(d*d) - 0.08203125*(a*a)*b*(d*d) + 0.05859375*a*(b*b)*(d*d) + 0.15234375*a*p - 0.009765625*(b*b*b)*(d*d) - 0.1142578125*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU6812(double a, double b, double p, double d, double s){
	return ((0.0068359375*(a*a)*(d*d) - 0.01171875*a*b*(d*d) + 0.003662109375*(b*b)*(d*d) + 0.0111083984375*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU6813(double a, double b, double p, double d, double s){
	return (d*(0.0009765625*a - 0.000732421875*b)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU6814(double a, double b, double p, double d, double s){
	return (6.103515625e-5/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU700(double a, double b, double p, double d, double s){
	return (b*d*(-(b*b*b*b*b*b)*(d*d*d*d*d*d) - 10.5*(b*b*b*b)*(d*d*d*d)*p - 26.25*(b*b)*(d*d)*(p*p) - 13.125*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU701(double a, double b, double p, double d, double s){
	return ((3.5*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 26.25*(b*b*b*b)*(d*d*d*d)*p + 39.375*(b*b)*(d*d)*(p*p) + 6.5625*(p*p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU702(double a, double b, double p, double d, double s){
	return (b*d*(-5.25*(b*b*b*b)*(d*d*d*d) - 26.25*(b*b)*(d*d)*p - 19.6875*(p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU703(double a, double b, double p, double d, double s){
	return ((4.375*(b*b*b*b)*(d*d*d*d) + 13.125*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU704(double a, double b, double p, double d, double s){
	return (b*d*(-2.1875*(b*b)*(d*d) - 3.28125*p)/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU705(double a, double b, double p, double d, double s){
	return ((0.65625*(b*b)*(d*d) + 0.328125*p)/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU706(double a, double b, double p, double d, double s){
	return (-0.109375*b*d/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU707(double a, double b, double p, double d, double s){
	return (0.0078125/(p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU710(double a, double b, double p, double d, double s){
	return ((-a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 10.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 26.25*a*(b*b*b)*(d*d*d*d)*(p*p) - 13.125*a*b*(d*d)*(p*p*p) + 3.5*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 26.25*(b*b*b*b)*(d*d*d*d)*(p*p) + 39.375*(b*b)*(d*d)*(p*p*p) + 6.5625*(p*p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU711(double a, double b, double p, double d, double s){
	return (d*(3.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 26.25*a*(b*b*b*b)*(d*d*d*d)*p + 39.375*a*(b*b)*(d*d)*(p*p) + 6.5625*a*(p*p*p) - 0.5*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 15.75*(b*b*b*b*b)*(d*d*d*d)*p - 65.625*(b*b*b)*(d*d)*(p*p) - 45.9375*b*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU712(double a, double b, double p, double d, double s){
	return ((-5.25*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 26.25*a*(b*b*b)*(d*d*d*d)*p - 19.6875*a*b*(d*d)*(p*p) + 1.75*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 26.25*(b*b*b*b)*(d*d*d*d)*p + 59.0625*(b*b)*(d*d)*(p*p) + 13.125*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU713(double a, double b, double p, double d, double s){
	return (d*(4.375*a*(b*b*b*b)*(d*d*d*d) + 13.125*a*(b*b)*(d*d)*p + 3.28125*a*(p*p) - 2.625*(b*b*b*b*b)*(d*d*d*d) - 21.875*(b*b*b)*(d*d)*p - 22.96875*b*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU714(double a, double b, double p, double d, double s){
	return ((-2.1875*a*(b*b*b)*(d*d*d*d) - 3.28125*a*b*(d*d)*p + 2.1875*(b*b*b*b)*(d*d*d*d) + 9.84375*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU715(double a, double b, double p, double d, double s){
	return (d*(0.65625*a*(b*b)*(d*d) + 0.328125*a*p - 1.09375*(b*b*b)*(d*d) - 2.296875*b*p)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU716(double a, double b, double p, double d, double s){
	return ((-0.109375*a*b*(d*d) + 0.328125*(b*b)*(d*d) + 0.21875*p)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU717(double a, double b, double p, double d, double s){
	return (d*(0.0078125*a - 0.0546875*b)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU718(double a, double b, double p, double d, double s){
	return (0.00390625/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU720(double a, double b, double p, double d, double s){
	return (d*(-(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 10.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 26.25*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 13.125*(a*a)*b*(d*d)*(p*p*p) + 7.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 52.5*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 78.75*a*(b*b)*(d*d)*(p*p*p) + 13.125*a*(p*p*p*p) - 0.5*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 15.75*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 65.625*(b*b*b)*(d*d)*(p*p*p) - 45.9375*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU721(double a, double b, double p, double d, double s){
	return ((3.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 26.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 39.375*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 6.5625*(a*a)*(d*d)*(p*p*p) - a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 31.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 131.25*a*(b*b*b)*(d*d*d*d)*(p*p) - 91.875*a*b*(d*d)*(p*p*p) + 5.25*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 65.625*(b*b*b*b)*(d*d*d*d)*(p*p) + 137.8125*(b*b)*(d*d)*(p*p*p) + 29.53125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU722(double a, double b, double p, double d, double s){
	return (d*(-5.25*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 26.25*(a*a)*(b*b*b)*(d*d*d*d)*p - 19.6875*(a*a)*b*(d*d)*(p*p) + 3.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 52.5*a*(b*b*b*b)*(d*d*d*d)*p + 118.125*a*(b*b)*(d*d)*(p*p) + 26.25*a*(p*p*p) - 0.25*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 15.75*(b*b*b*b*b)*(d*d*d*d)*p - 98.4375*(b*b*b)*(d*d)*(p*p) - 91.875*b*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU723(double a, double b, double p, double d, double s){
	return ((4.375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 13.125*(a*a)*(b*b)*(d*d*d*d)*p + 3.28125*(a*a)*(d*d)*(p*p) - 5.25*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 43.75*a*(b*b*b)*(d*d*d*d)*p - 45.9375*a*b*(d*d)*(p*p) + 0.875*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 21.875*(b*b*b*b)*(d*d*d*d)*p + 68.90625*(b*b)*(d*d)*(p*p) + 19.6875*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU724(double a, double b, double p, double d, double s){
	return (d*(-2.1875*(a*a)*(b*b*b)*(d*d*d*d) - 3.28125*(a*a)*b*(d*d)*p + 4.375*a*(b*b*b*b)*(d*d*d*d) + 19.6875*a*(b*b)*(d*d)*p + 6.5625*a*(p*p) - 1.3125*(b*b*b*b*b)*(d*d*d*d) - 16.40625*(b*b*b)*(d*d)*p - 22.96875*b*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU725(double a, double b, double p, double d, double s){
	return ((0.65625*(a*a)*(b*b)*(d*d*d*d) + 0.328125*(a*a)*(d*d)*p - 2.1875*a*(b*b*b)*(d*d*d*d) - 4.59375*a*b*(d*d)*p + 1.09375*(b*b*b*b)*(d*d*d*d) + 6.890625*(b*b)*(d*d)*p + 2.953125*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU726(double a, double b, double p, double d, double s){
	return (d*(-0.109375*(a*a)*b*(d*d) + 0.65625*a*(b*b)*(d*d) + 0.4375*a*p - 0.546875*(b*b*b)*(d*d) - 1.53125*b*p)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU727(double a, double b, double p, double d, double s){
	return ((0.0078125*(a*a)*(d*d) - 0.109375*a*b*(d*d) + 0.1640625*(b*b)*(d*d) + 0.140625*p)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU728(double a, double b, double p, double d, double s){
	return (d*(0.0078125*a - 0.02734375*b)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU729(double a, double b, double p, double d, double s){
	return (0.001953125/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU730(double a, double b, double p, double d, double s){
	return ((-(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 10.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 26.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 13.125*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 10.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 78.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 118.125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 19.6875*(a*a)*(d*d)*(p*p*p*p) - 1.5*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 47.25*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 196.875*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 137.8125*a*b*(d*d)*(p*p*p*p) + 5.25*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 65.625*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 137.8125*(b*b)*(d*d)*(p*p*p*p) + 29.53125*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU731(double a, double b, double p, double d, double s){
	return (d*(3.5*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 26.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 39.375*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 6.5625*(a*a*a)*(d*d)*(p*p*p) - 1.5*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 47.25*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 196.875*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 137.8125*(a*a)*b*(d*d)*(p*p*p) + 15.75*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 196.875*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 413.4375*a*(b*b)*(d*d)*(p*p*p) + 88.59375*a*(p*p*p*p) - 0.75*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 39.375*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 229.6875*(b*b*b)*(d*d)*(p*p*p) - 206.71875*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU732(double a, double b, double p, double d, double s){
	return ((-5.25*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 26.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 19.6875*(a*a*a)*b*(d*d*d*d)*(p*p) + 5.25*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 78.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 177.1875*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 39.375*(a*a)*(d*d)*(p*p*p) - 0.75*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 47.25*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 295.3125*a*(b*b*b)*(d*d*d*d)*(p*p) - 275.625*a*b*(d*d)*(p*p*p) + 5.25*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 98.4375*(b*b*b*b)*(d*d*d*d)*(p*p) + 275.625*(b*b)*(d*d)*(p*p*p) + 73.828125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU733(double a, double b, double p, double d, double s){
	return (d*(4.375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 13.125*(a*a*a)*(b*b)*(d*d*d*d)*p + 3.28125*(a*a*a)*(d*d)*(p*p) - 7.875*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 65.625*(a*a)*(b*b*b)*(d*d*d*d)*p - 68.90625*(a*a)*b*(d*d)*(p*p) + 2.625*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 65.625*a*(b*b*b*b)*(d*d*d*d)*p + 206.71875*a*(b*b)*(d*d)*(p*p) + 59.0625*a*(p*p*p) - 0.125*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 13.125*(b*b*b*b*b)*(d*d*d*d)*p - 114.84375*(b*b*b)*(d*d)*(p*p) - 137.8125*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU734(double a, double b, double p, double d, double s){
	return ((-2.1875*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 3.28125*(a*a*a)*b*(d*d*d*d)*p + 6.5625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 29.53125*(a*a)*(b*b)*(d*d*d*d)*p + 9.84375*(a*a)*(d*d)*(p*p) - 3.9375*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 49.21875*a*(b*b*b)*(d*d*d*d)*p - 68.90625*a*b*(d*d)*(p*p) + 0.4375*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 16.40625*(b*b*b*b)*(d*d*d*d)*p + 68.90625*(b*b)*(d*d)*(p*p) + 24.609375*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU735(double a, double b, double p, double d, double s){
	return (d*(0.65625*(a*a*a)*(b*b)*(d*d*d*d) + 0.328125*(a*a*a)*(d*d)*p - 3.28125*(a*a)*(b*b*b)*(d*d*d*d) - 6.890625*(a*a)*b*(d*d)*p + 3.28125*a*(b*b*b*b)*(d*d*d*d) + 20.671875*a*(b*b)*(d*d)*p + 8.859375*a*(p*p) - 0.65625*(b*b*b*b*b)*(d*d*d*d) - 11.484375*(b*b*b)*(d*d)*p - 20.671875*b*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU736(double a, double b, double p, double d, double s){
	return ((-0.109375*(a*a*a)*b*(d*d*d*d) + 0.984375*(a*a)*(b*b)*(d*d*d*d) + 0.65625*(a*a)*(d*d)*p - 1.640625*a*(b*b*b)*(d*d*d*d) - 4.59375*a*b*(d*d)*p + 0.546875*(b*b*b*b)*(d*d*d*d) + 4.59375*(b*b)*(d*d)*p + 2.4609375*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU737(double a, double b, double p, double d, double s){
	return (d*(0.0078125*(a*a*a)*(d*d) - 0.1640625*(a*a)*b*(d*d) + 0.4921875*a*(b*b)*(d*d) + 0.421875*a*p - 0.2734375*(b*b*b)*(d*d) - 0.984375*b*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU738(double a, double b, double p, double d, double s){
	return ((0.01171875*(a*a)*(d*d) - 0.08203125*a*b*(d*d) + 0.08203125*(b*b)*(d*d) + 0.087890625*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU739(double a, double b, double p, double d, double s){
	return (d*(0.005859375*a - 0.013671875*b)/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU7310(double a, double b, double p, double d, double s){
	return (0.0009765625/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU740(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 10.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 26.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 13.125*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 14.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 105.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 157.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 26.25*(a*a*a)*(d*d)*(p*p*p*p) - 3.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 94.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 393.75*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 275.625*(a*a)*b*(d*d)*(p*p*p*p) + 21.0 *a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 262.5*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 551.25*a*(b*b)*(d*d)*(p*p*p*p) + 118.125*a*(p*p*p*p*p) - 0.75*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 39.375*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 229.6875*(b*b*b)*(d*d)*(p*p*p*p) - 206.71875*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU741(double a, double b, double p, double d, double s){
	return ((3.5*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 26.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 39.375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 6.5625*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 2.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 63.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 262.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 183.75*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 31.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 393.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 826.875*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 177.1875*(a*a)*(d*d)*(p*p*p*p) - 3.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 157.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 918.75*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 826.875*a*b*(d*d)*(p*p*p*p) + 13.125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 229.6875*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 620.15625*(b*b)*(d*d)*(p*p*p*p) + 162.421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU742(double a, double b, double p, double d, double s){
	return (d*(-5.25*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 26.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 19.6875*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 7.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 105.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 236.25*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 52.5*(a*a*a)*(d*d)*(p*p*p) - 1.5*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 94.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 590.625*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 551.25*(a*a)*b*(d*d)*(p*p*p) + 21.0 *a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 393.75*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 1102.5*a*(b*b)*(d*d)*(p*p*p) + 295.3125*a*(p*p*p*p) - 0.75*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 59.0625*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 459.375*(b*b*b)*(d*d)*(p*p*p) - 516.796875*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU743(double a, double b, double p, double d, double s){
	return ((4.375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 13.125*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 3.28125*(a*a*a*a)*(d*d*d*d)*(p*p) - 10.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 87.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 91.875*(a*a*a)*b*(d*d*d*d)*(p*p) + 5.25*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 131.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 413.4375*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 118.125*(a*a)*(d*d)*(p*p*p) - 0.5*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 52.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 459.375*a*(b*b*b)*(d*d*d*d)*(p*p) - 551.25*a*b*(d*d)*(p*p*p) + 4.375*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 114.84375*(b*b*b*b)*(d*d*d*d)*(p*p) + 413.4375*(b*b)*(d*d)*(p*p*p) + 135.3515625*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU744(double a, double b, double p, double d, double s){
	return (d*(-2.1875*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 3.28125*(a*a*a*a)*b*(d*d*d*d)*p + 8.75*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 39.375*(a*a*a)*(b*b)*(d*d*d*d)*p + 13.125*(a*a*a)*(d*d)*(p*p) - 7.875*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 98.4375*(a*a)*(b*b*b)*(d*d*d*d)*p - 137.8125*(a*a)*b*(d*d)*(p*p) + 1.75*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 65.625*a*(b*b*b*b)*(d*d*d*d)*p + 275.625*a*(b*b)*(d*d)*(p*p) + 98.4375*a*(p*p*p) - 0.0625*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 9.84375*(b*b*b*b*b)*(d*d*d*d)*p - 114.84375*(b*b*b)*(d*d)*(p*p) - 172.265625*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU745(double a, double b, double p, double d, double s){
	return ((0.65625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.328125*(a*a*a*a)*(d*d*d*d)*p - 4.375*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 9.1875*(a*a*a)*b*(d*d*d*d)*p + 6.5625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 41.34375*(a*a)*(b*b)*(d*d*d*d)*p + 17.71875*(a*a)*(d*d)*(p*p) - 2.625*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 45.9375*a*(b*b*b)*(d*d*d*d)*p - 82.6875*a*b*(d*d)*(p*p) + 0.21875*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 11.484375*(b*b*b*b)*(d*d*d*d)*p + 62.015625*(b*b)*(d*d)*(p*p) + 27.0703125*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU746(double a, double b, double p, double d, double s){
	return (d*(-0.109375*(a*a*a*a)*b*(d*d*d*d) + 1.3125*(a*a*a)*(b*b)*(d*d*d*d) + 0.875*(a*a*a)*(d*d)*p - 3.28125*(a*a)*(b*b*b)*(d*d*d*d) - 9.1875*(a*a)*b*(d*d)*p + 2.1875*a*(b*b*b*b)*(d*d*d*d) + 18.375*a*(b*b)*(d*d)*p + 9.84375*a*(p*p) - 0.328125*(b*b*b*b*b)*(d*d*d*d) - 7.65625*(b*b*b)*(d*d)*p - 17.2265625*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU747(double a, double b, double p, double d, double s){
	return ((0.0078125*(a*a*a*a)*(d*d*d*d) - 0.21875*(a*a*a)*b*(d*d*d*d) + 0.984375*(a*a)*(b*b)*(d*d*d*d) + 0.84375*(a*a)*(d*d)*p - 1.09375*a*(b*b*b)*(d*d*d*d) - 3.9375*a*b*(d*d)*p + 0.2734375*(b*b*b*b)*(d*d*d*d) + 2.953125*(b*b)*(d*d)*p + 1.93359375*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU748(double a, double b, double p, double d, double s){
	return (d*(0.015625*(a*a*a)*(d*d) - 0.1640625*(a*a)*b*(d*d) + 0.328125*a*(b*b)*(d*d) + 0.3515625*a*p - 0.13671875*(b*b*b)*(d*d) - 0.615234375*b*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU749(double a, double b, double p, double d, double s){
	return ((0.01171875*(a*a)*(d*d) - 0.0546875*a*b*(d*d) + 0.041015625*(b*b)*(d*d) + 0.0537109375*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU7410(double a, double b, double p, double d, double s){
	return (d*(0.00390625*a - 0.0068359375*b)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU7411(double a, double b, double p, double d, double s){
	return (0.00048828125/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU750(double a, double b, double p, double d, double s){
	return ((-(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 10.5*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 26.25*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 13.125*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 17.5*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 131.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 196.875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 32.8125*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 5.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 157.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 656.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 459.375*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 52.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 656.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 1378.125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 295.3125*(a*a)*(d*d)*(p*p*p*p*p) - 3.75*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 196.875*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 1148.4375*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 1033.59375*a*b*(d*d)*(p*p*p*p*p) + 13.125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 229.6875*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 620.15625*(b*b)*(d*d)*(p*p*p*p*p) + 162.421875*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU751(double a, double b, double p, double d, double s){
	return (d*(3.5*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 26.25*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 39.375*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 6.5625*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 2.5*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 78.75*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 328.125*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 229.6875*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 52.5*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 656.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 1378.125*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 295.3125*(a*a*a)*(d*d)*(p*p*p*p) - 7.5*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 393.75*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 2296.875*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 2067.1875*(a*a)*b*(d*d)*(p*p*p*p) + 65.625*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 1148.4375*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 3100.78125*a*(b*b)*(d*d)*(p*p*p*p) + 812.109375*a*(p*p*p*p*p) - 1.875*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 137.8125*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 1033.59375*(b*b*b)*(d*d)*(p*p*p*p) - 1136.953125*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU752(double a, double b, double p, double d, double s){
	return ((-5.25*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 26.25*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 19.6875*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 8.75*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 131.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 295.3125*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 65.625*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 2.5*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 157.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 984.375*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 918.75*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 52.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 984.375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 2756.25*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 738.28125*(a*a)*(d*d)*(p*p*p*p) - 3.75*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 295.3125*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 2296.875*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 2583.984375*a*b*(d*d)*(p*p*p*p) + 19.6875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 459.375*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 1550.390625*(b*b)*(d*d)*(p*p*p*p) + 487.265625*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU753(double a, double b, double p, double d, double s){
	return (d*(4.375*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 13.125*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 3.28125*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 13.125*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 109.375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 114.84375*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 8.75*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 218.75*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 689.0625*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 196.875*(a*a*a)*(d*d)*(p*p*p) - 1.25*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 131.25*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 1148.4375*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 1378.125*(a*a)*b*(d*d)*(p*p*p) + 21.875*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 574.21875*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 2067.1875*a*(b*b)*(d*d)*(p*p*p) + 676.7578125*a*(p*p*p*p) - 0.625*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 68.90625*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 689.0625*(b*b*b)*(d*d)*(p*p*p) - 947.4609375*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU754(double a, double b, double p, double d, double s){
	return ((-2.1875*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 3.28125*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 10.9375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 49.21875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 16.40625*(a*a*a*a)*(d*d*d*d)*(p*p) - 13.125*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 164.0625*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 229.6875*(a*a*a)*b*(d*d*d*d)*(p*p) + 4.375*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 164.0625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 689.0625*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 246.09375*(a*a)*(d*d)*(p*p*p) - 0.3125*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 49.21875*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 574.21875*a*(b*b*b)*(d*d*d*d)*(p*p) - 861.328125*a*b*(d*d)*(p*p*p) + 3.28125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 114.84375*(b*b*b*b)*(d*d*d*d)*(p*p) + 516.796875*(b*b)*(d*d)*(p*p*p) + 203.02734375*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU755(double a, double b, double p, double d, double s){
	return (d*(0.65625*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.328125*(a*a*a*a*a)*(d*d*d*d)*p - 5.46875*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 11.484375*(a*a*a*a)*b*(d*d*d*d)*p + 10.9375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 68.90625*(a*a*a)*(b*b)*(d*d*d*d)*p + 29.53125*(a*a*a)*(d*d)*(p*p) - 6.5625*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 114.84375*(a*a)*(b*b*b)*(d*d*d*d)*p - 206.71875*(a*a)*b*(d*d)*(p*p) + 1.09375*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 57.421875*a*(b*b*b*b)*(d*d*d*d)*p + 310.078125*a*(b*b)*(d*d)*(p*p) + 135.3515625*a*(p*p*p) - 0.03125*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 6.890625*(b*b*b*b*b)*(d*d*d*d)*p - 103.359375*(b*b*b)*(d*d)*(p*p) - 189.4921875*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU756(double a, double b, double p, double d, double s){
	return ((-0.109375*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 1.640625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 1.09375*(a*a*a*a)*(d*d*d*d)*p - 5.46875*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 15.3125*(a*a*a)*b*(d*d*d*d)*p + 5.46875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 45.9375*(a*a)*(b*b)*(d*d*d*d)*p + 24.609375*(a*a)*(d*d)*(p*p) - 1.640625*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 38.28125*a*(b*b*b)*(d*d*d*d)*p - 86.1328125*a*b*(d*d)*(p*p) + 0.109375*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 7.65625*(b*b*b*b)*(d*d*d*d)*p + 51.6796875*(b*b)*(d*d)*(p*p) + 27.0703125*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU757(double a, double b, double p, double d, double s){
	return (d*(0.0078125*(a*a*a*a*a)*(d*d*d*d) - 0.2734375*(a*a*a*a)*b*(d*d*d*d) + 1.640625*(a*a*a)*(b*b)*(d*d*d*d) + 1.40625*(a*a*a)*(d*d)*p - 2.734375*(a*a)*(b*b*b)*(d*d*d*d) - 9.84375*(a*a)*b*(d*d)*p + 1.3671875*a*(b*b*b*b)*(d*d*d*d) + 14.765625*a*(b*b)*(d*d)*p + 9.66796875*a*(p*p) - 0.1640625*(b*b*b*b*b)*(d*d*d*d) - 4.921875*(b*b*b)*(d*d)*p - 13.53515625*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU758(double a, double b, double p, double d, double s){
	return ((0.01953125*(a*a*a*a)*(d*d*d*d) - 0.2734375*(a*a*a)*b*(d*d*d*d) + 0.8203125*(a*a)*(b*b)*(d*d*d*d) + 0.87890625*(a*a)*(d*d)*p - 0.68359375*a*(b*b*b)*(d*d*d*d) - 3.076171875*a*b*(d*d)*p + 0.13671875*(b*b*b*b)*(d*d*d*d) + 1.845703125*(b*b)*(d*d)*p + 1.4501953125*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU759(double a, double b, double p, double d, double s){
	return (d*(0.01953125*(a*a*a)*(d*d) - 0.13671875*(a*a)*b*(d*d) + 0.205078125*a*(b*b)*(d*d) + 0.2685546875*a*p - 0.068359375*(b*b*b)*(d*d) - 0.3759765625*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU7510(double a, double b, double p, double d, double s){
	return ((0.009765625*(a*a)*(d*d) - 0.0341796875*a*b*(d*d) + 0.0205078125*(b*b)*(d*d) + 0.0322265625*p)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU7511(double a, double b, double p, double d, double s){
	return (d*(0.00244140625*a - 0.00341796875*b)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU7512(double a, double b, double p, double d, double s){
	return (0.000244140625/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU760(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 10.5*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 26.25*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 13.125*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 21.0 *(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 157.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 236.25*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 39.375*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 7.5*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 236.25*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 984.375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 689.0625*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 105.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 1312.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 2756.25*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 590.625*(a*a*a)*(d*d)*(p*p*p*p*p) - 11.25*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 590.625*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 3445.3125*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 3100.78125*(a*a)*b*(d*d)*(p*p*p*p*p) + 78.75*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 1378.125*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 3720.9375*a*(b*b)*(d*d)*(p*p*p*p*p) + 974.53125*a*(p*p*p*p*p*p) - 1.875*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 137.8125*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p) - 1033.59375*(b*b*b)*(d*d)*(p*p*p*p*p) - 1136.953125*b*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU761(double a, double b, double p, double d, double s){
	return ((3.5*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 26.25*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 39.375*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 6.5625*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 3.0*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 94.5*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 393.75*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 275.625*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 78.75*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 984.375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 2067.1875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 442.96875*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 15.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 787.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 4593.75*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 4134.375*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 196.875*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 3445.3125*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 9302.34375*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 2436.328125*(a*a)*(d*d)*(p*p*p*p*p) - 11.25*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 826.875*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 6201.5625*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 6821.71875*a*b*(d*d)*(p*p*p*p*p) + 45.9375*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 1033.59375*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 3410.859375*(b*b)*(d*d)*(p*p*p*p*p) + 1055.7421875*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU762(double a, double b, double p, double d, double s){
	return (d*(-5.25*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 26.25*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 19.6875*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 10.5*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 157.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 354.375*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 78.75*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 3.75*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 236.25*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1476.5625*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 1378.125*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 105.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1968.75*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 5512.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 1476.5625*(a*a*a)*(d*d)*(p*p*p*p) - 11.25*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 885.9375*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 6890.625*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 7751.953125*(a*a)*b*(d*d)*(p*p*p*p) + 118.125*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 2756.25*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 9302.34375*a*(b*b)*(d*d)*(p*p*p*p) + 2923.59375*a*(p*p*p*p*p) - 2.8125*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 275.625*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 2583.984375*(b*b*b)*(d*d)*(p*p*p*p) - 3410.859375*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU763(double a, double b, double p, double d, double s){
	return ((4.375*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 13.125*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 3.28125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 15.75*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 131.25*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 137.8125*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 13.125*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 328.125*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1033.59375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 295.3125*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 2.5*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 262.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 2296.875*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 2756.25*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 65.625*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1722.65625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 6201.5625*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 2030.2734375*(a*a)*(d*d)*(p*p*p*p) - 3.75*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 413.4375*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 4134.375*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 5684.765625*a*b*(d*d)*(p*p*p*p) + 22.96875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 689.0625*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 2842.3828125*(b*b)*(d*d)*(p*p*p*p) + 1055.7421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU764(double a, double b, double p, double d, double s){
	return (d*(-2.1875*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 3.28125*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 13.125*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 59.0625*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 19.6875*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 19.6875*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 246.09375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 344.53125*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 8.75*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 328.125*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1378.125*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 492.1875*(a*a*a)*(d*d)*(p*p*p) - 0.9375*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 147.65625*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 1722.65625*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 2583.984375*(a*a)*b*(d*d)*(p*p*p) + 19.6875*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 689.0625*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 3100.78125*a*(b*b)*(d*d)*(p*p*p) + 1218.1640625*a*(p*p*p*p) - 0.46875*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 68.90625*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 861.328125*(b*b*b)*(d*d)*(p*p*p) - 1421.19140625*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU765(double a, double b, double p, double d, double s){
	return ((0.65625*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.328125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 6.5625*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 13.78125*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 16.40625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 103.359375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 44.296875*(a*a*a*a)*(d*d*d*d)*(p*p) - 13.125*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 229.6875*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 413.4375*(a*a*a)*b*(d*d*d*d)*(p*p) + 3.28125*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 172.265625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 930.234375*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 406.0546875*(a*a)*(d*d)*(p*p*p) - 0.1875*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 41.34375*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 620.15625*a*(b*b*b)*(d*d*d*d)*(p*p) - 1136.953125*a*b*(d*d)*(p*p*p) + 2.296875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 103.359375*(b*b*b*b)*(d*d*d*d)*(p*p) + 568.4765625*(b*b)*(d*d)*(p*p*p) + 263.935546875*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU766(double a, double b, double p, double d, double s){
	return (d*(-0.109375*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 1.96875*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 1.3125*(a*a*a*a*a)*(d*d*d*d)*p - 8.203125*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 22.96875*(a*a*a*a)*b*(d*d*d*d)*p + 10.9375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 91.875*(a*a*a)*(b*b)*(d*d*d*d)*p + 49.21875*(a*a*a)*(d*d)*(p*p) - 4.921875*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 114.84375*(a*a)*(b*b*b)*(d*d*d*d)*p - 258.3984375*(a*a)*b*(d*d)*(p*p) + 0.65625*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 45.9375*a*(b*b*b*b)*(d*d*d*d)*p + 310.078125*a*(b*b)*(d*d)*(p*p) + 162.421875*a*(p*p*p) - 0.015625*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 4.59375*(b*b*b*b*b)*(d*d*d*d)*p - 86.1328125*(b*b*b)*(d*d)*(p*p) - 189.4921875*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU767(double a, double b, double p, double d, double s){
	return ((0.0078125*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.328125*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 2.4609375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 2.109375*(a*a*a*a)*(d*d*d*d)*p - 5.46875*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 19.6875*(a*a*a)*b*(d*d*d*d)*p + 4.1015625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 44.296875*(a*a)*(b*b)*(d*d*d*d)*p + 29.00390625*(a*a)*(d*d)*(p*p) - 0.984375*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 29.53125*a*(b*b*b)*(d*d*d*d)*p - 81.2109375*a*b*(d*d)*(p*p) + 0.0546875*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 4.921875*(b*b*b*b)*(d*d*d*d)*p + 40.60546875*(b*b)*(d*d)*(p*p) + 25.13671875*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU768(double a, double b, double p, double d, double s){
	return (d*(0.0234375*(a*a*a*a*a)*(d*d*d*d) - 0.41015625*(a*a*a*a)*b*(d*d*d*d) + 1.640625*(a*a*a)*(b*b)*(d*d*d*d) + 1.7578125*(a*a*a)*(d*d)*p - 2.05078125*(a*a)*(b*b*b)*(d*d*d*d) - 9.228515625*(a*a)*b*(d*d)*p + 0.8203125*a*(b*b*b*b)*(d*d*d*d) + 11.07421875*a*(b*b)*(d*d)*p + 8.701171875*a*(p*p) - 0.08203125*(b*b*b*b*b)*(d*d*d*d) - 3.076171875*(b*b*b)*(d*d)*p - 10.1513671875*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU769(double a, double b, double p, double d, double s){
	return ((0.029296875*(a*a*a*a)*(d*d*d*d) - 0.2734375*(a*a*a)*b*(d*d*d*d) + 0.615234375*(a*a)*(b*b)*(d*d*d*d) + 0.8056640625*(a*a)*(d*d)*p - 0.41015625*a*(b*b*b)*(d*d*d*d) - 2.255859375*a*b*(d*d)*p + 0.068359375*(b*b*b*b)*(d*d*d*d) + 1.1279296875*(b*b)*(d*d)*p + 1.04736328125*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU7610(double a, double b, double p, double d, double s){
	return (d*(0.01953125*(a*a*a)*(d*d) - 0.1025390625*(a*a)*b*(d*d) + 0.123046875*a*(b*b)*(d*d) + 0.193359375*a*p - 0.0341796875*(b*b*b)*(d*d) - 0.2255859375*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU7611(double a, double b, double p, double d, double s){
	return ((0.00732421875*(a*a)*(d*d) - 0.0205078125*a*b*(d*d) + 0.01025390625*(b*b)*(d*d) + 0.01904296875*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU7612(double a, double b, double p, double d, double s){
	return (d*(0.00146484375*a - 0.001708984375*b)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU7613(double a, double b, double p, double d, double s){
	return (0.0001220703125/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU770(double a, double b, double p, double d, double s){
	return ((-(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) - 10.5*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 26.25*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 13.125*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p*p) + 24.5*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 183.75*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 275.625*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 45.9375*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p*p) - 10.5*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 330.75*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 1378.125*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 964.6875*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p*p) + 183.75*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 2296.875*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 4823.4375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 1033.59375*(a*a*a*a)*(d*d*d*d)*(p*p*p*p*p) - 26.25*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 1378.125*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 8039.0625*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 7235.15625*(a*a*a)*b*(d*d*d*d)*(p*p*p*p*p) + 275.625*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 4823.4375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 13023.28125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p*p) + 3410.859375*(a*a)*(d*d)*(p*p*p*p*p*p) - 13.125*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 964.6875*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 7235.15625*a*(b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 7958.671875*a*b*(d*d)*(p*p*p*p*p*p) + 45.9375*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 1033.59375*(b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) + 3410.859375*(b*b)*(d*d)*(p*p*p*p*p*p) + 1055.7421875*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU771(double a, double b, double p, double d, double s){
	return (d*(3.5*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 26.25*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 39.375*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 6.5625*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 3.5*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 110.25*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 459.375*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 321.5625*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 110.25*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 1378.125*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 2894.0625*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 620.15625*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 26.25*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 1378.125*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 8039.0625*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 7235.15625*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 459.375*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 8039.0625*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 21705.46875*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 5684.765625*(a*a*a)*(d*d)*(p*p*p*p*p) - 39.375*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 2894.0625*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 21705.46875*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 23876.015625*(a*a)*b*(d*d)*(p*p*p*p*p) + 321.5625*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 7235.15625*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 23876.015625*a*(b*b)*(d*d)*(p*p*p*p*p) + 7390.1953125*a*(p*p*p*p*p*p) - 6.5625*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 620.15625*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p) - 5684.765625*(b*b*b)*(d*d)*(p*p*p*p*p) - 7390.1953125*b*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU772(double a, double b, double p, double d, double s){
	return ((-5.25*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 26.25*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 19.6875*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 12.25*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 183.75*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 413.4375*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 91.875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 5.25*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 330.75*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 2067.1875*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 1929.375*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 183.75*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 3445.3125*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 9646.875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 2583.984375*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 26.25*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 2067.1875*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 16078.125*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 18087.890625*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 413.4375*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 9646.875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 32558.203125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 10232.578125*(a*a)*(d*d)*(p*p*p*p*p) - 19.6875*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 1929.375*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 18087.890625*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 23876.015625*a*b*(d*d)*(p*p*p*p*p) + 91.875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 2583.984375*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 10232.578125*(b*b)*(d*d)*(p*p*p*p*p) + 3695.09765625*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU773(double a, double b, double p, double d, double s){
	return (d*(4.375*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 13.125*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 3.28125*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 18.375*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 153.125*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 160.78125*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 18.375*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 459.375*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1447.03125*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 413.4375*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 4.375*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 459.375*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 4019.53125*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 4823.4375*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 153.125*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 4019.53125*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 14470.3125*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 4737.3046875*(a*a*a)*(d*d)*(p*p*p*p) - 13.125*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1447.03125*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 14470.3125*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 19896.6796875*(a*a)*b*(d*d)*(p*p*p*p) + 160.78125*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 4823.4375*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 19896.6796875*a*(b*b)*(d*d)*(p*p*p*p) + 7390.1953125*a*(p*p*p*p*p) - 3.28125*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 413.4375*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 4737.3046875*(b*b*b)*(d*d)*(p*p*p*p) - 7390.1953125*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU774(double a, double b, double p, double d, double s){
	return ((-2.1875*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 3.28125*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 15.3125*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 68.90625*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 22.96875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 27.5625*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 344.53125*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 482.34375*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 15.3125*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 574.21875*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 2411.71875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 861.328125*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 2.1875*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 344.53125*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 4019.53125*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 6029.296875*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 68.90625*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 2411.71875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 10852.734375*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 4263.57421875*(a*a)*(d*d)*(p*p*p*p) - 3.28125*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 482.34375*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 6029.296875*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 9948.33984375*a*b*(d*d)*(p*p*p*p) + 22.96875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 861.328125*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 4263.57421875*(b*b)*(d*d)*(p*p*p*p) + 1847.548828125*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU775(double a, double b, double p, double d, double s){
	return (d*(0.65625*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.328125*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 7.65625*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 16.078125*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 22.96875*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 144.703125*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 62.015625*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 22.96875*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 401.953125*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 723.515625*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 7.65625*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 401.953125*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 2170.546875*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 947.4609375*(a*a*a)*(d*d)*(p*p*p) - 0.65625*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 144.703125*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 2170.546875*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 3979.3359375*(a*a)*b*(d*d)*(p*p*p) + 16.078125*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 723.515625*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 3979.3359375*a*(b*b)*(d*d)*(p*p*p) + 1847.548828125*a*(p*p*p*p) - 0.328125*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 62.015625*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 947.4609375*(b*b*b)*(d*d)*(p*p*p) - 1847.548828125*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU776(double a, double b, double p, double d, double s){
	return ((-0.109375*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 2.296875*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 1.53125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 11.484375*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 32.15625*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 19.140625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 160.78125*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 86.1328125*(a*a*a*a)*(d*d*d*d)*(p*p) - 11.484375*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 267.96875*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 602.9296875*(a*a*a)*b*(d*d*d*d)*(p*p) + 2.296875*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 160.78125*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1085.2734375*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 568.4765625*(a*a)*(d*d)*(p*p*p) - 0.109375*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 32.15625*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 602.9296875*a*(b*b*b)*(d*d*d*d)*(p*p) - 1326.4453125*a*b*(d*d)*(p*p*p) + 1.53125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 86.1328125*(b*b*b*b)*(d*d*d*d)*(p*p) + 568.4765625*(b*b)*(d*d)*(p*p*p) + 307.9248046875*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU777(double a, double b, double p, double d, double s){
	return (d*(0.0078125*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.3828125*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 3.4453125*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 2.953125*(a*a*a*a*a)*(d*d*d*d)*p - 9.5703125*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 34.453125*(a*a*a*a)*b*(d*d*d*d)*p + 9.5703125*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 103.359375*(a*a*a)*(b*b)*(d*d*d*d)*p + 67.67578125*(a*a*a)*(d*d)*(p*p) - 3.4453125*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 103.359375*(a*a)*(b*b*b)*(d*d*d*d)*p - 284.23828125*(a*a)*b*(d*d)*(p*p) + 0.3828125*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 34.453125*a*(b*b*b*b)*(d*d*d*d)*p + 284.23828125*a*(b*b)*(d*d)*(p*p) + 175.95703125*a*(p*p*p) - 0.0078125*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 2.953125*(b*b*b*b*b)*(d*d*d*d)*p - 67.67578125*(b*b*b)*(d*d)*(p*p) - 175.95703125*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU778(double a, double b, double p, double d, double s){
	return ((0.02734375*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.57421875*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 2.87109375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 3.076171875*(a*a*a*a)*(d*d*d*d)*p - 4.78515625*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 21.533203125*(a*a*a)*b*(d*d*d*d)*p + 2.87109375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 38.759765625*(a*a)*(b*b)*(d*d*d*d)*p + 30.4541015625*(a*a)*(d*d)*(p*p) - 0.57421875*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 21.533203125*a*(b*b*b)*(d*d*d*d)*p - 71.0595703125*a*b*(d*d)*(p*p) + 0.02734375*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 3.076171875*(b*b*b*b)*(d*d*d*d)*p + 30.4541015625*(b*b)*(d*d)*(p*p) + 21.99462890625*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU779(double a, double b, double p, double d, double s){
	return (d*(0.041015625*(a*a*a*a*a)*(d*d*d*d) - 0.478515625*(a*a*a*a)*b*(d*d*d*d) + 1.435546875*(a*a*a)*(b*b)*(d*d*d*d) + 1.8798828125*(a*a*a)*(d*d)*p - 1.435546875*(a*a)*(b*b*b)*(d*d*d*d) - 7.8955078125*(a*a)*b*(d*d)*p + 0.478515625*a*(b*b*b*b)*(d*d*d*d) + 7.8955078125*a*(b*b)*(d*d)*p + 7.33154296875*a*(p*p) - 0.041015625*(b*b*b*b*b)*(d*d*d*d) - 1.8798828125*(b*b*b)*(d*d)*p - 7.33154296875*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU7710(double a, double b, double p, double d, double s){
	return ((0.0341796875*(a*a*a*a)*(d*d*d*d) - 0.2392578125*(a*a*a)*b*(d*d*d*d) + 0.4306640625*(a*a)*(b*b)*(d*d*d*d) + 0.6767578125*(a*a)*(d*d)*p - 0.2392578125*a*(b*b*b)*(d*d*d*d) - 1.5791015625*a*b*(d*d)*p + 0.0341796875*(b*b*b*b)*(d*d*d*d) + 0.6767578125*(b*b)*(d*d)*p + 0.733154296875*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU7711(double a, double b, double p, double d, double s){
	return (d*(0.01708984375*(a*a*a)*(d*d) - 0.07177734375*(a*a)*b*(d*d) + 0.07177734375*a*(b*b)*(d*d) + 0.13330078125*a*p - 0.01708984375*(b*b*b)*(d*d) - 0.13330078125*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU7712(double a, double b, double p, double d, double s){
	return ((0.005126953125*(a*a)*(d*d) - 0.011962890625*a*b*(d*d) + 0.005126953125*(b*b)*(d*d) + 0.0111083984375*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU7713(double a, double b, double p, double d, double s){
	return (0.0008544921875*d*(a - b)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU7714(double a, double b, double p, double d, double s){
	return (6.103515625e-5/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU780(double a, double b, double p, double d, double s){
	return (d*(-(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) - 10.5*(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 26.25*(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 13.125*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p*p) + 28.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 210.0*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 315.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 52.5*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p*p) - 14.0*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 441.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 1837.5*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 1286.25*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p*p) + 294.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 3675.0*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 7717.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 1653.75*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p*p) - 52.5*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 2756.25*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 16078.125*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 14470.3125*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p*p) + 735.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 12862.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 34728.75*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p*p) + 9095.625*(a*a*a)*(d*d)*(p*p*p*p*p*p) - 52.5*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 3858.75*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 28940.625*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 31834.6875*(a*a)*b*(d*d)*(p*p*p*p*p*p) + 367.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 8268.75*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) + 27286.875*a*(b*b)*(d*d)*(p*p*p*p*p*p) + 8445.9375*a*(p*p*p*p*p*p*p) - 6.5625*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 620.15625*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 5684.765625*(b*b*b)*(d*d)*(p*p*p*p*p*p) - 7390.1953125*b*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU781(double a, double b, double p, double d, double s){
	return ((3.5*(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) + 26.25*(a*a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 39.375*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 6.5625*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*(p*p*p) - 4.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) - 126.0*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 525.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 367.5*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p*p) + 147.0*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 1837.5*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 3858.75*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 826.875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p*p) - 42.0*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 2205.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 12862.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 11576.25*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p*p) + 918.75*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 16078.125*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 43410.9375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 11369.53125*(a*a*a*a)*(d*d*d*d)*(p*p*p*p*p) - 105.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 7717.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 57881.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 63669.375*(a*a*a)*b*(d*d*d*d)*(p*p*p*p*p) + 1286.25*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 28940.625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 95504.0625*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p*p) + 29560.78125*(a*a)*(d*d)*(p*p*p*p*p*p) - 52.5*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 4961.25*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 45478.125*a*(b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 59121.5625*a*b*(d*d)*(p*p*p*p*p*p) + 206.71875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 5684.765625*(b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) + 22170.5859375*(b*b)*(d*d)*(p*p*p*p*p*p) + 7918.06640625*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU782(double a, double b, double p, double d, double s){
	return (d*(-5.25*(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 26.25*(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 19.6875*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 14.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 210.0*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 472.5*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 105.0*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 7.0*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 441.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 2756.25*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 2572.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 294.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 5512.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 15435.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 4134.375*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 52.5*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 4134.375*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 32156.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 36175.78125*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 1102.5*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 25725.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 86821.875*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 27286.875*(a*a*a)*(d*d)*(p*p*p*p*p) - 78.75*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 7717.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 72351.5625*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 95504.0625*(a*a)*b*(d*d)*(p*p*p*p*p) + 735.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 20671.875*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 81860.625*a*(b*b)*(d*d)*(p*p*p*p*p) + 29560.78125*a*(p*p*p*p*p*p) - 13.125*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 1550.390625*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p) - 17054.296875*(b*b*b)*(d*d)*(p*p*p*p*p) - 25865.68359375*b*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU783(double a, double b, double p, double d, double s){
	return ((4.375*(a*a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 13.125*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 3.28125*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*(p*p) - 21.0 *(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 175.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 183.75*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 24.5*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 612.5*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 1929.375*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 551.25*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 7.0*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 735.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 6431.25*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 7717.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 306.25*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 8039.0625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 28940.625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 9474.609375*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 35.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 3858.75*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 38587.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 53057.8125*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 643.125*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 19293.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 79586.71875*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 29560.78125*(a*a)*(d*d)*(p*p*p*p*p) - 26.25*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 3307.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 37898.4375*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 59121.5625*a*b*(d*d)*(p*p*p*p*p) + 137.8125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 4737.3046875*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 22170.5859375*(b*b)*(d*d)*(p*p*p*p*p) + 9237.744140625*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU784(double a, double b, double p, double d, double s){
	return (d*(-2.1875*(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 3.28125*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 17.5*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 78.75*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 26.25*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 36.75*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 459.375*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 643.125*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 24.5*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 918.75*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 3858.75*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 1378.125*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 4.375*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 689.0625*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 8039.0625*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 12058.59375*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 183.75*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 6431.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 28940.625*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 11369.53125*(a*a*a)*(d*d)*(p*p*p*p) - 13.125*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1929.375*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 24117.1875*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 39793.359375*(a*a)*b*(d*d)*(p*p*p*p) + 183.75*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 6890.625*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 34108.59375*a*(b*b)*(d*d)*(p*p*p*p) + 14780.390625*a*(p*p*p*p*p) - 3.28125*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 516.796875*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 7105.95703125*(b*b*b)*(d*d)*(p*p*p*p) - 12932.841796875*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU785(double a, double b, double p, double d, double s){
	return ((0.65625*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d) + 0.328125*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*p - 8.75*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 18.375*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 30.625*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 192.9375*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 82.6875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 36.75*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 643.125*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 1157.625*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 15.3125*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 803.90625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 4341.09375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 1894.921875*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 1.75*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 385.875*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 5788.125*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 10611.5625*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 64.3125*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 2894.0625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 15917.34375*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 7390.1953125*(a*a)*(d*d)*(p*p*p*p) - 2.625*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 496.125*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 7579.6875*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 14780.390625*a*b*(d*d)*(p*p*p*p) + 20.671875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 947.4609375*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 5542.646484375*(b*b)*(d*d)*(p*p*p*p) + 2771.3232421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU786(double a, double b, double p, double d, double s){
	return (d*(-0.109375*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 2.625*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 1.75*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 15.3125*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 42.875*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 30.625*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 257.25*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 137.8125*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 22.96875*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 535.9375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 1205.859375*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 6.125*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 428.75*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 2894.0625*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 1515.9375*(a*a*a)*(d*d)*(p*p*p) - 0.4375*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 128.625*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 2411.71875*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 5305.78125*(a*a)*b*(d*d)*(p*p*p) + 12.25*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 689.0625*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 4547.8125*a*(b*b)*(d*d)*(p*p*p) + 2463.3984375*a*(p*p*p*p) - 0.21875*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 51.6796875*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 947.4609375*(b*b*b)*(d*d)*(p*p*p) - 2155.4736328125*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU787(double a, double b, double p, double d, double s){
	return ((0.0078125*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d) - 0.4375*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 4.59375*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 3.9375*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 15.3125*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 55.125*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 19.140625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 206.71875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 135.3515625*(a*a*a*a)*(d*d*d*d)*(p*p) - 9.1875*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 275.625*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 757.96875*(a*a*a)*b*(d*d*d*d)*(p*p) + 1.53125*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 137.8125*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1136.953125*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 703.828125*(a*a)*(d*d)*(p*p*p) - 0.0625*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 23.625*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 541.40625*a*(b*b*b)*(d*d*d*d)*(p*p) - 1407.65625*a*b*(d*d)*(p*p*p) + 0.984375*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 67.67578125*(b*b*b*b)*(d*d*d*d)*(p*p) + 527.87109375*(b*b)*(d*d)*(p*p*p) + 329.91943359375*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU788(double a, double b, double p, double d, double s){
	return (d*(0.03125*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.765625*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 4.59375*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 4.921875*(a*a*a*a*a)*(d*d*d*d)*p - 9.5703125*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 43.06640625*(a*a*a*a)*b*(d*d*d*d)*p + 7.65625*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 103.359375*(a*a*a)*(b*b)*(d*d*d*d)*p + 81.2109375*(a*a*a)*(d*d)*(p*p) - 2.296875*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 86.1328125*(a*a)*(b*b*b)*(d*d*d*d)*p - 284.23828125*(a*a)*b*(d*d)*(p*p) + 0.21875*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 24.609375*a*(b*b*b*b)*(d*d*d*d)*p + 243.6328125*a*(b*b)*(d*d)*(p*p) + 175.95703125*a*(p*p*p) - 0.00390625*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 1.845703125*(b*b*b*b*b)*(d*d*d*d)*p - 50.7568359375*(b*b*b)*(d*d)*(p*p) - 153.96240234375*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU789(double a, double b, double p, double d, double s){
	return ((0.0546875*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.765625*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 2.87109375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 3.759765625*(a*a*a*a)*(d*d*d*d)*p - 3.828125*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 21.0546875*(a*a*a)*b*(d*d*d*d)*p + 1.9140625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 31.58203125*(a*a)*(b*b)*(d*d*d*d)*p + 29.326171875*(a*a)*(d*d)*(p*p) - 0.328125*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 15.0390625*a*(b*b*b)*(d*d*d*d)*p - 58.65234375*a*b*(d*d)*(p*p) + 0.013671875*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 1.8798828125*(b*b*b*b)*(d*d*d*d)*p + 21.99462890625*(b*b)*(d*d)*(p*p) + 18.328857421875*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU7810(double a, double b, double p, double d, double s){
	return (d*(0.0546875*(a*a*a*a*a)*(d*d*d*d) - 0.478515625*(a*a*a*a)*b*(d*d*d*d) + 1.1484375*(a*a*a)*(b*b)*(d*d*d*d) + 1.8046875*(a*a*a)*(d*d)*p - 0.95703125*(a*a)*(b*b*b)*(d*d*d*d) - 6.31640625*(a*a)*b*(d*d)*p + 0.2734375*a*(b*b*b*b)*(d*d*d*d) + 5.4140625*a*(b*b)*(d*d)*p + 5.865234375*a*(p*p) - 0.0205078125*(b*b*b*b*b)*(d*d*d*d) - 1.1279296875*(b*b*b)*(d*d)*p - 5.132080078125*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU7811(double a, double b, double p, double d, double s){
	return ((0.0341796875*(a*a*a*a)*(d*d*d*d) - 0.19140625*(a*a*a)*b*(d*d*d*d) + 0.287109375*(a*a)*(b*b)*(d*d*d*d) + 0.533203125*(a*a)*(d*d)*p - 0.13671875*a*(b*b*b)*(d*d*d*d) - 1.06640625*a*b*(d*d)*p + 0.01708984375*(b*b*b*b)*(d*d*d*d) + 0.39990234375*(b*b)*(d*d)*p + 0.4998779296875*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU7812(double a, double b, double p, double d, double s){
	return (d*(0.013671875*(a*a*a)*(d*d) - 0.0478515625*(a*a)*b*(d*d) + 0.041015625*a*(b*b)*(d*d) + 0.0888671875*a*p - 0.008544921875*(b*b*b)*(d*d) - 0.0777587890625*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU7813(double a, double b, double p, double d, double s){
	return ((0.00341796875*(a*a)*(d*d) - 0.0068359375*a*b*(d*d) + 0.0025634765625*(b*b)*(d*d) + 0.00640869140625*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU7814(double a, double b, double p, double d, double s){
	return (d*(0.00048828125*a - 0.00042724609375*b)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU7815(double a, double b, double p, double d, double s){
	return (3.0517578125e-5/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU800(double a, double b, double p, double d, double s){
	return (((b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 14.0*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 52.5*(b*b*b*b)*(d*d*d*d)*(p*p) + 52.5*(b*b)*(d*d)*(p*p*p) + 6.5625*(p*p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU801(double a, double b, double p, double d, double s){
	return (b*d*(-4.0*(b*b*b*b*b*b)*(d*d*d*d*d*d) - 42.0*(b*b*b*b)*(d*d*d*d)*p - 105.0*(b*b)*(d*d)*(p*p) - 52.5*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU802(double a, double b, double p, double d, double s){
	return ((7.0*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 52.5*(b*b*b*b)*(d*d*d*d)*p + 78.75*(b*b)*(d*d)*(p*p) + 13.125*(p*p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU803(double a, double b, double p, double d, double s){
	return (b*d*(-7.0*(b*b*b*b)*(d*d*d*d) - 35.0*(b*b)*(d*d)*p - 26.25*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU804(double a, double b, double p, double d, double s){
	return ((4.375*(b*b*b*b)*(d*d*d*d) + 13.125*(b*b)*(d*d)*p + 3.28125*(p*p))/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU805(double a, double b, double p, double d, double s){
	return (b*d*(-1.75*(b*b)*(d*d) - 2.625*p)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU806(double a, double b, double p, double d, double s){
	return ((0.4375*(b*b)*(d*d) + 0.21875*p)/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU807(double a, double b, double p, double d, double s){
	return (-0.0625*b*d/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU808(double a, double b, double p, double d, double s){
	return (0.00390625/(p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU810(double a, double b, double p, double d, double s){
	return (d*(a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 14.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 52.5*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 52.5*a*(b*b)*(d*d)*(p*p*p) + 6.5625*a*(p*p*p*p) - 4.0*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 42.0*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 105.0*(b*b*b)*(d*d)*(p*p*p) - 52.5*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU811(double a, double b, double p, double d, double s){
	return ((-4.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 42.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 105.0*a*(b*b*b)*(d*d*d*d)*(p*p) - 52.5*a*b*(d*d)*(p*p*p) + 0.5*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 21.0 *(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 131.25*(b*b*b*b)*(d*d*d*d)*(p*p) + 183.75*(b*b)*(d*d)*(p*p*p) + 29.53125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU812(double a, double b, double p, double d, double s){
	return (d*(7.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 52.5*a*(b*b*b*b)*(d*d*d*d)*p + 78.75*a*(b*b)*(d*d)*(p*p) + 13.125*a*(p*p*p) - 2.0*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 42.0*(b*b*b*b*b)*(d*d*d*d)*p - 157.5*(b*b*b)*(d*d)*(p*p) - 105.0*b*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU813(double a, double b, double p, double d, double s){
	return ((-7.0*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 35.0*a*(b*b*b)*(d*d*d*d)*p - 26.25*a*b*(d*d)*(p*p) + 3.5*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 43.75*(b*b*b*b)*(d*d*d*d)*p + 91.875*(b*b)*(d*d)*(p*p) + 19.6875*(p*p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU814(double a, double b, double p, double d, double s){
	return (d*(4.375*a*(b*b*b*b)*(d*d*d*d) + 13.125*a*(b*b)*(d*d)*p + 3.28125*a*(p*p) - 3.5*(b*b*b*b*b)*(d*d*d*d) - 26.25*(b*b*b)*(d*d)*p - 26.25*b*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU815(double a, double b, double p, double d, double s){
	return ((-1.75*a*(b*b*b)*(d*d*d*d) - 2.625*a*b*(d*d)*p + 2.1875*(b*b*b*b)*(d*d*d*d) + 9.1875*(b*b)*(d*d)*p + 2.953125*(p*p))/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU816(double a, double b, double p, double d, double s){
	return (d*(0.4375*a*(b*b)*(d*d) + 0.21875*a*p - 0.875*(b*b*b)*(d*d) - 1.75*b*p)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU817(double a, double b, double p, double d, double s){
	return ((-0.0625*a*b*(d*d) + 0.21875*(b*b)*(d*d) + 0.140625*p)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU818(double a, double b, double p, double d, double s){
	return (d*(0.00390625*a - 0.03125*b)/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU819(double a, double b, double p, double d, double s){
	return (0.001953125/(p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU820(double a, double b, double p, double d, double s){
	return (((a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 14.0*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 52.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 52.5*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 6.5625*(a*a)*(d*d)*(p*p*p*p) - 8.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 84.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 210.0*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 105.0*a*b*(d*d)*(p*p*p*p) + 0.5*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 21.0 *(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 131.25*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 183.75*(b*b)*(d*d)*(p*p*p*p) + 29.53125*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU821(double a, double b, double p, double d, double s){
	return (d*(-4.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 42.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 105.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 52.5*(a*a)*b*(d*d)*(p*p*p) + a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 42.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 262.5*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 367.5*a*(b*b)*(d*d)*(p*p*p) + 59.0625*a*(p*p*p*p) - 6.0*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 105.0*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 367.5*(b*b*b)*(d*d)*(p*p*p) - 236.25*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU822(double a, double b, double p, double d, double s){
	return ((7.0*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 52.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 78.75*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 13.125*(a*a)*(d*d)*(p*p*p) - 4.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 84.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 315.0*a*(b*b*b)*(d*d*d*d)*(p*p) - 210.0*a*b*(d*d)*(p*p*p) + 0.25*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 21.0 *(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 196.875*(b*b*b*b)*(d*d*d*d)*(p*p) + 367.5*(b*b)*(d*d)*(p*p*p) + 73.828125*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU823(double a, double b, double p, double d, double s){
	return (d*(-7.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 35.0*(a*a)*(b*b*b)*(d*d*d*d)*p - 26.25*(a*a)*b*(d*d)*(p*p) + 7.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 87.5*a*(b*b*b*b)*(d*d*d*d)*p + 183.75*a*(b*b)*(d*d)*(p*p) + 39.375*a*(p*p*p) - (b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 35.0*(b*b*b*b*b)*(d*d*d*d)*p - 183.75*(b*b*b)*(d*d)*(p*p) - 157.5*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU824(double a, double b, double p, double d, double s){
	return ((4.375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 13.125*(a*a)*(b*b)*(d*d*d*d)*p + 3.28125*(a*a)*(d*d)*(p*p) - 7.0*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 52.5*a*(b*b*b)*(d*d*d*d)*p - 52.5*a*b*(d*d)*(p*p) + 1.75*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 32.8125*(b*b*b*b)*(d*d*d*d)*p + 91.875*(b*b)*(d*d)*(p*p) + 24.609375*(p*p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU825(double a, double b, double p, double d, double s){
	return (d*(-1.75*(a*a)*(b*b*b)*(d*d*d*d) - 2.625*(a*a)*b*(d*d)*p + 4.375*a*(b*b*b*b)*(d*d*d*d) + 18.375*a*(b*b)*(d*d)*p + 5.90625*a*(p*p) - 1.75*(b*b*b*b*b)*(d*d*d*d) - 18.375*(b*b*b)*(d*d)*p - 23.625*b*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU826(double a, double b, double p, double d, double s){
	return ((0.4375*(a*a)*(b*b)*(d*d*d*d) + 0.21875*(a*a)*(d*d)*p - 1.75*a*(b*b*b)*(d*d*d*d) - 3.5*a*b*(d*d)*p + 1.09375*(b*b*b*b)*(d*d*d*d) + 6.125*(b*b)*(d*d)*p + 2.4609375*(p*p))/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU827(double a, double b, double p, double d, double s){
	return (d*(-0.0625*(a*a)*b*(d*d) + 0.4375*a*(b*b)*(d*d) + 0.28125*a*p - 0.4375*(b*b*b)*(d*d) - 1.125*b*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU828(double a, double b, double p, double d, double s){
	return ((0.00390625*(a*a)*(d*d) - 0.0625*a*b*(d*d) + 0.109375*(b*b)*(d*d) + 0.087890625*p)/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU829(double a, double b, double p, double d, double s){
	return (d*(0.00390625*a - 0.015625*b)/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8210(double a, double b, double p, double d, double s){
	return (0.0009765625/(p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU830(double a, double b, double p, double d, double s){
	return (d*((a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 14.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 52.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 52.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 6.5625*(a*a*a)*(d*d)*(p*p*p*p) - 12.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 126.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 315.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 157.5*(a*a)*b*(d*d)*(p*p*p*p) + 1.5*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 63.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 393.75*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 551.25*a*(b*b)*(d*d)*(p*p*p*p) + 88.59375*a*(p*p*p*p*p) - 6.0*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 105.0*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 367.5*(b*b*b)*(d*d)*(p*p*p*p) - 236.25*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU831(double a, double b, double p, double d, double s){
	return ((-4.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 42.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 105.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 52.5*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 1.5*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 63.0*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 393.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 551.25*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 88.59375*(a*a)*(d*d)*(p*p*p*p) - 18.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 315.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 1102.5*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 708.75*a*b*(d*d)*(p*p*p*p) + 0.75*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 52.5*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 459.375*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 826.875*(b*b)*(d*d)*(p*p*p*p) + 162.421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU832(double a, double b, double p, double d, double s){
	return (d*(7.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 52.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 78.75*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 13.125*(a*a*a)*(d*d)*(p*p*p) - 6.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 126.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 472.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 315.0*(a*a)*b*(d*d)*(p*p*p) + 0.75*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 63.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 590.625*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 1102.5*a*(b*b)*(d*d)*(p*p*p) + 221.484375*a*(p*p*p*p) - 6.0*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 157.5*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 735.0*(b*b*b)*(d*d)*(p*p*p) - 590.625*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU833(double a, double b, double p, double d, double s){
	return ((-7.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 35.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 26.25*(a*a*a)*b*(d*d*d*d)*(p*p) + 10.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 131.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 275.625*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 59.0625*(a*a)*(d*d)*(p*p*p) - 3.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 105.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 551.25*a*(b*b*b)*(d*d*d*d)*(p*p) - 472.5*a*b*(d*d)*(p*p*p) + 0.125*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 17.5*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 229.6875*(b*b*b*b)*(d*d*d*d)*(p*p) + 551.25*(b*b)*(d*d)*(p*p*p) + 135.3515625*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU834(double a, double b, double p, double d, double s){
	return (d*(4.375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 13.125*(a*a*a)*(b*b)*(d*d*d*d)*p + 3.28125*(a*a*a)*(d*d)*(p*p) - 10.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 78.75*(a*a)*(b*b*b)*(d*d*d*d)*p - 78.75*(a*a)*b*(d*d)*(p*p) + 5.25*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 98.4375*a*(b*b*b*b)*(d*d*d*d)*p + 275.625*a*(b*b)*(d*d)*(p*p) + 73.828125*a*(p*p*p) - 0.5*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 26.25*(b*b*b*b*b)*(d*d*d*d)*p - 183.75*(b*b*b)*(d*d)*(p*p) - 196.875*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU835(double a, double b, double p, double d, double s){
	return ((-1.75*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 2.625*(a*a*a)*b*(d*d*d*d)*p + 6.5625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 27.5625*(a*a)*(b*b)*(d*d*d*d)*p + 8.859375*(a*a)*(d*d)*(p*p) - 5.25*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 55.125*a*(b*b*b)*(d*d*d*d)*p - 70.875*a*b*(d*d)*(p*p) + 0.875*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 22.96875*(b*b*b*b)*(d*d*d*d)*p + 82.6875*(b*b)*(d*d)*(p*p) + 27.0703125*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU836(double a, double b, double p, double d, double s){
	return (d*(0.4375*(a*a*a)*(b*b)*(d*d*d*d) + 0.21875*(a*a*a)*(d*d)*p - 2.625*(a*a)*(b*b*b)*(d*d*d*d) - 5.25*(a*a)*b*(d*d)*p + 3.28125*a*(b*b*b*b)*(d*d*d*d) + 18.375*a*(b*b)*(d*d)*p + 7.3828125*a*(p*p) - 0.875*(b*b*b*b*b)*(d*d*d*d) - 12.25*(b*b*b)*(d*d)*p - 19.6875*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU837(double a, double b, double p, double d, double s){
	return ((-0.0625*(a*a*a)*b*(d*d*d*d) + 0.65625*(a*a)*(b*b)*(d*d*d*d) + 0.421875*(a*a)*(d*d)*p - 1.3125*a*(b*b*b)*(d*d*d*d) - 3.375*a*b*(d*d)*p + 0.546875*(b*b*b*b)*(d*d*d*d) + 3.9375*(b*b)*(d*d)*p + 1.93359375*(p*p))/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU838(double a, double b, double p, double d, double s){
	return (d*(0.00390625*(a*a*a)*(d*d) - 0.09375*(a*a)*b*(d*d) + 0.328125*a*(b*b)*(d*d) + 0.263671875*a*p - 0.21875*(b*b*b)*(d*d) - 0.703125*b*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU839(double a, double b, double p, double d, double s){
	return ((0.005859375*(a*a)*(d*d) - 0.046875*a*b*(d*d) + 0.0546875*(b*b)*(d*d) + 0.0537109375*p)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8310(double a, double b, double p, double d, double s){
	return (d*(0.0029296875*a - 0.0078125*b)/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8311(double a, double b, double p, double d, double s){
	return (0.00048828125/(p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU840(double a, double b, double p, double d, double s){
	return (((a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 14.0*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 52.5*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 52.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 6.5625*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 16.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 168.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 420.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 210.0*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 3.0*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 126.0*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 787.5*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 1102.5*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 177.1875*(a*a)*(d*d)*(p*p*p*p*p) - 24.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 420.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 1470.0*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 945.0*a*b*(d*d)*(p*p*p*p*p) + 0.75*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 52.5*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 459.375*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 826.875*(b*b)*(d*d)*(p*p*p*p*p) + 162.421875*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU841(double a, double b, double p, double d, double s){
	return (d*(-4.0*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 42.0*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 105.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 52.5*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 2.0*(a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 84.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 525.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 735.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 118.125*(a*a*a)*(d*d)*(p*p*p*p) - 36.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 630.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 2205.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 1417.5*(a*a)*b*(d*d)*(p*p*p*p) + 3.0*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 210.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 1837.5*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 3307.5*a*(b*b)*(d*d)*(p*p*p*p) + 649.6875*a*(p*p*p*p*p) - 15.0*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 367.5*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 1653.75*(b*b*b)*(d*d)*(p*p*p*p) - 1299.375*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU842(double a, double b, double p, double d, double s){
	return ((7.0*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 52.5*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 78.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 13.125*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 8.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 168.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 630.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 420.0*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 1.5*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 126.0*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1181.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 2205.0*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 442.96875*(a*a)*(d*d)*(p*p*p*p) - 24.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 630.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 2940.0*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 2362.5*a*b*(d*d)*(p*p*p*p) + 0.75*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 78.75*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 918.75*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 2067.1875*(b*b)*(d*d)*(p*p*p*p) + 487.265625*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU843(double a, double b, double p, double d, double s){
	return (d*(-7.0*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 35.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 26.25*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 14.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 175.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 367.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 78.75*(a*a*a)*(d*d)*(p*p*p) - 6.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 210.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 1102.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 945.0*(a*a)*b*(d*d)*(p*p*p) + 0.5*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 70.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 918.75*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 2205.0*a*(b*b)*(d*d)*(p*p*p) + 541.40625*a*(p*p*p*p) - 5.0*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 183.75*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 1102.5*(b*b*b)*(d*d)*(p*p*p) - 1082.8125*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU844(double a, double b, double p, double d, double s){
	return ((4.375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 13.125*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 3.28125*(a*a*a*a)*(d*d*d*d)*(p*p) - 14.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 105.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 105.0*(a*a*a)*b*(d*d*d*d)*(p*p) + 10.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 196.875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 551.25*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 147.65625*(a*a)*(d*d)*(p*p*p) - 2.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 105.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 735.0*a*(b*b*b)*(d*d*d*d)*(p*p) - 787.5*a*b*(d*d)*(p*p*p) + 0.0625*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 13.125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 229.6875*(b*b*b*b)*(d*d*d*d)*(p*p) + 689.0625*(b*b)*(d*d)*(p*p*p) + 203.02734375*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU845(double a, double b, double p, double d, double s){
	return (d*(-1.75*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 2.625*(a*a*a*a)*b*(d*d*d*d)*p + 8.75*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 36.75*(a*a*a)*(b*b)*(d*d*d*d)*p + 11.8125*(a*a*a)*(d*d)*(p*p) - 10.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 110.25*(a*a)*(b*b*b)*(d*d*d*d)*p - 141.75*(a*a)*b*(d*d)*(p*p) + 3.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 91.875*a*(b*b*b*b)*(d*d*d*d)*p + 330.75*a*(b*b)*(d*d)*(p*p) + 108.28125*a*(p*p*p) - 0.25*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 18.375*(b*b*b*b*b)*(d*d*d*d)*p - 165.375*(b*b*b)*(d*d)*(p*p) - 216.5625*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU846(double a, double b, double p, double d, double s){
	return ((0.4375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.21875*(a*a*a*a)*(d*d*d*d)*p - 3.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 7.0*(a*a*a)*b*(d*d*d*d)*p + 6.5625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 36.75*(a*a)*(b*b)*(d*d*d*d)*p + 14.765625*(a*a)*(d*d)*(p*p) - 3.5*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 49.0*a*(b*b*b)*(d*d*d*d)*p - 78.75*a*b*(d*d)*(p*p) + 0.4375*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 15.3125*(b*b*b*b)*(d*d*d*d)*p + 68.90625*(b*b)*(d*d)*(p*p) + 27.0703125*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU847(double a, double b, double p, double d, double s){
	return (d*(-0.0625*(a*a*a*a)*b*(d*d*d*d) + 0.875*(a*a*a)*(b*b)*(d*d*d*d) + 0.5625*(a*a*a)*(d*d)*p - 2.625*(a*a)*(b*b*b)*(d*d*d*d) - 6.75*(a*a)*b*(d*d)*p + 2.1875*a*(b*b*b*b)*(d*d*d*d) + 15.75*a*(b*b)*(d*d)*p + 7.734375*a*(p*p) - 0.4375*(b*b*b*b*b)*(d*d*d*d) - 7.875*(b*b*b)*(d*d)*p - 15.46875*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU848(double a, double b, double p, double d, double s){
	return ((0.00390625*(a*a*a*a)*(d*d*d*d) - 0.125*(a*a*a)*b*(d*d*d*d) + 0.65625*(a*a)*(b*b)*(d*d*d*d) + 0.52734375*(a*a)*(d*d)*p - 0.875*a*(b*b*b)*(d*d*d*d) - 2.8125*a*b*(d*d)*p + 0.2734375*(b*b*b*b)*(d*d*d*d) + 2.4609375*(b*b)*(d*d)*p + 1.4501953125*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU849(double a, double b, double p, double d, double s){
	return (d*(0.0078125*(a*a*a)*(d*d) - 0.09375*(a*a)*b*(d*d) + 0.21875*a*(b*b)*(d*d) + 0.21484375*a*p - 0.109375*(b*b*b)*(d*d) - 0.4296875*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8410(double a, double b, double p, double d, double s){
	return ((0.005859375*(a*a)*(d*d) - 0.03125*a*b*(d*d) + 0.02734375*(b*b)*(d*d) + 0.0322265625*p)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8411(double a, double b, double p, double d, double s){
	return (d*(0.001953125*a - 0.00390625*b)/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8412(double a, double b, double p, double d, double s){
	return (0.000244140625/(p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU850(double a, double b, double p, double d, double s){
	return (d*((a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 14.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 52.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 52.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 6.5625*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 20.0*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 210.0*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 525.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 262.5*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 5.0*(a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 210.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 1312.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 1837.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 295.3125*(a*a*a)*(d*d)*(p*p*p*p*p) - 60.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 1050.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 3675.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 2362.5*(a*a)*b*(d*d)*(p*p*p*p*p) + 3.75*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 262.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 2296.875*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 4134.375*a*(b*b)*(d*d)*(p*p*p*p*p) + 812.109375*a*(p*p*p*p*p*p) - 15.0*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 367.5*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p) - 1653.75*(b*b*b)*(d*d)*(p*p*p*p*p) - 1299.375*b*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU851(double a, double b, double p, double d, double s){
	return ((-4.0*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 42.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 105.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 52.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 2.5*(a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 105.0*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 656.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 918.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 147.65625*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 60.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 1050.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 3675.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 2362.5*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 7.5*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 525.0*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 4593.75*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 8268.75*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 1624.21875*(a*a)*(d*d)*(p*p*p*p*p) - 75.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 1837.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 8268.75*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 6496.875*a*b*(d*d)*(p*p*p*p*p) + 1.875*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 183.75*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 2067.1875*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 4547.8125*(b*b)*(d*d)*(p*p*p*p*p) + 1055.7421875*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU852(double a, double b, double p, double d, double s){
	return (d*(7.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 52.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 78.75*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 13.125*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 10.0*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 210.0*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 787.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 525.0*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 2.5*(a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 210.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1968.75*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 3675.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 738.28125*(a*a*a)*(d*d)*(p*p*p*p) - 60.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1575.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 7350.0*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 5906.25*(a*a)*b*(d*d)*(p*p*p*p) + 3.75*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 393.75*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 4593.75*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 10335.9375*a*(b*b)*(d*d)*(p*p*p*p) + 2436.328125*a*(p*p*p*p*p) - 22.5*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 735.0*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 4134.375*(b*b*b)*(d*d)*(p*p*p*p) - 3898.125*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU853(double a, double b, double p, double d, double s){
	return ((-7.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 35.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 26.25*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 17.5*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 218.75*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 459.375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 98.4375*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 10.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 350.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1837.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 1575.0*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 1.25*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 175.0*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 2296.875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 5512.5*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 1353.515625*(a*a)*(d*d)*(p*p*p*p) - 25.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 918.75*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 5512.5*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 5414.0625*a*b*(d*d)*(p*p*p*p) + 0.625*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 91.875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 1378.125*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 3789.84375*(b*b)*(d*d)*(p*p*p*p) + 1055.7421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU854(double a, double b, double p, double d, double s){
	return (d*(4.375*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 13.125*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 3.28125*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 17.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 131.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 131.25*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 17.5*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 328.125*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 918.75*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 246.09375*(a*a*a)*(d*d)*(p*p*p) - 5.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 262.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 1837.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 1968.75*(a*a)*b*(d*d)*(p*p*p) + 0.3125*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 65.625*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 1148.4375*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 3445.3125*a*(b*b)*(d*d)*(p*p*p) + 1015.13671875*a*(p*p*p*p) - 3.75*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 183.75*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 1378.125*(b*b*b)*(d*d)*(p*p*p) - 1624.21875*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU855(double a, double b, double p, double d, double s){
	return ((-1.75*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 2.625*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 10.9375*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 45.9375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 14.765625*(a*a*a*a)*(d*d*d*d)*(p*p) - 17.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 183.75*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 236.25*(a*a*a)*b*(d*d*d*d)*(p*p) + 8.75*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 229.6875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 826.875*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 270.703125*(a*a)*(d*d)*(p*p*p) - 1.25*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 91.875*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 826.875*a*(b*b*b)*(d*d*d*d)*(p*p) - 1082.8125*a*b*(d*d)*(p*p*p) + 0.03125*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 9.1875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 206.71875*(b*b*b*b)*(d*d*d*d)*(p*p) + 757.96875*(b*b)*(d*d)*(p*p*p) + 263.935546875*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU856(double a, double b, double p, double d, double s){
	return (d*(0.4375*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.21875*(a*a*a*a*a)*(d*d*d*d)*p - 4.375*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 8.75*(a*a*a*a)*b*(d*d*d*d)*p + 10.9375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 61.25*(a*a*a)*(b*b)*(d*d*d*d)*p + 24.609375*(a*a*a)*(d*d)*(p*p) - 8.75*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 122.5*(a*a)*(b*b*b)*(d*d*d*d)*p - 196.875*(a*a)*b*(d*d)*(p*p) + 2.1875*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 76.5625*a*(b*b*b*b)*(d*d*d*d)*p + 344.53125*a*(b*b)*(d*d)*(p*p) + 135.3515625*a*(p*p*p) - 0.125*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 12.25*(b*b*b*b*b)*(d*d*d*d)*p - 137.8125*(b*b*b)*(d*d)*(p*p) - 216.5625*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU857(double a, double b, double p, double d, double s){
	return ((-0.0625*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 1.09375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.703125*(a*a*a*a)*(d*d*d*d)*p - 4.375*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 11.25*(a*a*a)*b*(d*d*d*d)*p + 5.46875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 39.375*(a*a)*(b*b)*(d*d*d*d)*p + 19.3359375*(a*a)*(d*d)*(p*p) - 2.1875*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 39.375*a*(b*b*b)*(d*d*d*d)*p - 77.34375*a*b*(d*d)*(p*p) + 0.21875*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 9.84375*(b*b*b*b)*(d*d*d*d)*p + 54.140625*(b*b)*(d*d)*(p*p) + 25.13671875*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU858(double a, double b, double p, double d, double s){
	return (d*(0.00390625*(a*a*a*a*a)*(d*d*d*d) - 0.15625*(a*a*a*a)*b*(d*d*d*d) + 1.09375*(a*a*a)*(b*b)*(d*d*d*d) + 0.87890625*(a*a*a)*(d*d)*p - 2.1875*(a*a)*(b*b*b)*(d*d*d*d) - 7.03125*(a*a)*b*(d*d)*p + 1.3671875*a*(b*b*b*b)*(d*d*d*d) + 12.3046875*a*(b*b)*(d*d)*p + 7.2509765625*a*(p*p) - 0.21875*(b*b*b*b*b)*(d*d*d*d) - 4.921875*(b*b*b)*(d*d)*p - 11.6015625*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU859(double a, double b, double p, double d, double s){
	return ((0.009765625*(a*a*a*a)*(d*d*d*d) - 0.15625*(a*a*a)*b*(d*d*d*d) + 0.546875*(a*a)*(b*b)*(d*d*d*d) + 0.537109375*(a*a)*(d*d)*p - 0.546875*a*(b*b*b)*(d*d*d*d) - 2.1484375*a*b*(d*d)*p + 0.13671875*(b*b*b*b)*(d*d*d*d) + 1.50390625*(b*b)*(d*d)*p + 1.04736328125*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8510(double a, double b, double p, double d, double s){
	return (d*(0.009765625*(a*a*a)*(d*d) - 0.078125*(a*a)*b*(d*d) + 0.13671875*a*(b*b)*(d*d) + 0.1611328125*a*p - 0.0546875*(b*b*b)*(d*d) - 0.2578125*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8511(double a, double b, double p, double d, double s){
	return ((0.0048828125*(a*a)*(d*d) - 0.01953125*a*b*(d*d) + 0.013671875*(b*b)*(d*d) + 0.01904296875*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8512(double a, double b, double p, double d, double s){
	return (d*(0.001220703125*a - 0.001953125*b)/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8513(double a, double b, double p, double d, double s){
	return (0.0001220703125/(p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU860(double a, double b, double p, double d, double s){
	return (((a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) + 14.0*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 52.5*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 52.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 6.5625*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p*p) - 24.0*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 252.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 630.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 315.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p*p) + 7.5*(a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 315.0*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 1968.75*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 2756.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 442.96875*(a*a*a*a)*(d*d*d*d)*(p*p*p*p*p) - 120.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 2100.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 7350.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 4725.0*(a*a*a)*b*(d*d*d*d)*(p*p*p*p*p) + 11.25*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 787.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 6890.625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 12403.125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p*p) + 2436.328125*(a*a)*(d*d)*(p*p*p*p*p*p) - 90.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 2205.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 9922.5*a*(b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 7796.25*a*b*(d*d)*(p*p*p*p*p*p) + 1.875*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 183.75*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 2067.1875*(b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) + 4547.8125*(b*b)*(d*d)*(p*p*p*p*p*p) + 1055.7421875*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU861(double a, double b, double p, double d, double s){
	return (d*(-4.0*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 42.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 105.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 52.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 3.0*(a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 126.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 787.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 1102.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 177.1875*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 90.0*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 1575.0*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 5512.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 3543.75*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 15.0*(a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 1050.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 9187.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 16537.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 3248.4375*(a*a*a)*(d*d)*(p*p*p*p*p) - 225.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 5512.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 24806.25*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 19490.625*(a*a)*b*(d*d)*(p*p*p*p*p) + 11.25*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 1102.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 12403.125*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 27286.875*a*(b*b)*(d*d)*(p*p*p*p*p) + 6334.453125*a*(p*p*p*p*p*p) - 52.5*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 1653.75*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p) - 9095.625*(b*b*b)*(d*d)*(p*p*p*p*p) - 8445.9375*b*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU862(double a, double b, double p, double d, double s){
	return ((7.0*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 52.5*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 78.75*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 13.125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 12.0*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 252.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 945.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 630.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 3.75*(a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 315.0*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 2953.125*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 5512.5*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 1107.421875*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 120.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 3150.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 14700.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 11812.5*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 11.25*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 1181.25*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 13781.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 31007.8125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 7308.984375*(a*a)*(d*d)*(p*p*p*p*p) - 135.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 4410.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 24806.25*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 23388.75*a*b*(d*d)*(p*p*p*p*p) + 2.8125*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 367.5*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 5167.96875*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 13643.4375*(b*b)*(d*d)*(p*p*p*p*p) + 3695.09765625*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU863(double a, double b, double p, double d, double s){
	return (d*(-7.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 35.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 26.25*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 21.0 *(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 262.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 551.25*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 118.125*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 15.0*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 525.0*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 2756.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 2362.5*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 2.5*(a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 350.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 4593.75*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 11025.0*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 2707.03125*(a*a*a)*(d*d)*(p*p*p*p) - 75.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 2756.25*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 16537.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 16242.1875*(a*a)*b*(d*d)*(p*p*p*p) + 3.75*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 551.25*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 8268.75*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 22739.0625*a*(b*b)*(d*d)*(p*p*p*p) + 6334.453125*a*(p*p*p*p*p) - 26.25*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 1102.5*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 7579.6875*(b*b*b)*(d*d)*(p*p*p*p) - 8445.9375*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU864(double a, double b, double p, double d, double s){
	return ((4.375*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 13.125*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 3.28125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 21.0 *(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 157.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 157.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 26.25*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 492.1875*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1378.125*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 369.140625*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 10.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 525.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 3675.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 3937.5*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 0.9375*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 196.875*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 3445.3125*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 10335.9375*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 3045.41015625*(a*a)*(d*d)*(p*p*p*p) - 22.5*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1102.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 8268.75*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 9745.3125*a*b*(d*d)*(p*p*p*p) + 0.46875*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 91.875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 1722.65625*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 5684.765625*(b*b)*(d*d)*(p*p*p*p) + 1847.548828125*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU865(double a, double b, double p, double d, double s){
	return (d*(-1.75*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 2.625*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 13.125*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 55.125*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 17.71875*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 26.25*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 275.625*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 354.375*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 17.5*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 459.375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1653.75*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 541.40625*(a*a*a)*(d*d)*(p*p*p) - 3.75*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 275.625*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 2480.625*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 3248.4375*(a*a)*b*(d*d)*(p*p*p) + 0.1875*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 55.125*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 1240.3125*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 4547.8125*a*(b*b)*(d*d)*(p*p*p) + 1583.61328125*a*(p*p*p*p) - 2.625*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 165.375*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 1515.9375*(b*b*b)*(d*d)*(p*p*p) - 2111.484375*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU866(double a, double b, double p, double d, double s){
	return ((0.4375*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.21875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 5.25*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 10.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 16.40625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 91.875*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 36.9140625*(a*a*a*a)*(d*d*d*d)*(p*p) - 17.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 245.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 393.75*(a*a*a)*b*(d*d*d*d)*(p*p) + 6.5625*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 229.6875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1033.59375*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 406.0546875*(a*a)*(d*d)*(p*p*p) - 0.75*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 73.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 826.875*a*(b*b*b)*(d*d*d*d)*(p*p) - 1299.375*a*b*(d*d)*(p*p*p) + 0.015625*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 6.125*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 172.265625*(b*b*b*b)*(d*d*d*d)*(p*p) + 757.96875*(b*b)*(d*d)*(p*p*p) + 307.9248046875*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU867(double a, double b, double p, double d, double s){
	return (d*(-0.0625*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 1.3125*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 0.84375*(a*a*a*a*a)*(d*d*d*d)*p - 6.5625*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 16.875*(a*a*a*a)*b*(d*d*d*d)*p + 10.9375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 78.75*(a*a*a)*(b*b)*(d*d*d*d)*p + 38.671875*(a*a*a)*(d*d)*(p*p) - 6.5625*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 118.125*(a*a)*(b*b*b)*(d*d*d*d)*p - 232.03125*(a*a)*b*(d*d)*(p*p) + 1.3125*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 59.0625*a*(b*b*b*b)*(d*d*d*d)*p + 324.84375*a*(b*b)*(d*d)*(p*p) + 150.8203125*a*(p*p*p) - 0.0625*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 7.875*(b*b*b*b*b)*(d*d*d*d)*p - 108.28125*(b*b*b)*(d*d)*(p*p) - 201.09375*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU868(double a, double b, double p, double d, double s){
	return ((0.00390625*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.1875*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 1.640625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 1.318359375*(a*a*a*a)*(d*d*d*d)*p - 4.375*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 14.0625*(a*a*a)*b*(d*d*d*d)*p + 4.1015625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 36.9140625*(a*a)*(b*b)*(d*d*d*d)*p + 21.7529296875*(a*a)*(d*d)*(p*p) - 1.3125*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 29.53125*a*(b*b*b)*(d*d*d*d)*p - 69.609375*a*b*(d*d)*(p*p) + 0.109375*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 6.15234375*(b*b*b*b)*(d*d*d*d)*p + 40.60546875*(b*b)*(d*d)*(p*p) + 21.99462890625*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU869(double a, double b, double p, double d, double s){
	return (d*(0.01171875*(a*a*a*a*a)*(d*d*d*d) - 0.234375*(a*a*a*a)*b*(d*d*d*d) + 1.09375*(a*a*a)*(b*b)*(d*d*d*d) + 1.07421875*(a*a*a)*(d*d)*p - 1.640625*(a*a)*(b*b*b)*(d*d*d*d) - 6.4453125*(a*a)*b*(d*d)*p + 0.8203125*a*(b*b*b*b)*(d*d*d*d) + 9.0234375*a*(b*b)*(d*d)*p + 6.2841796875*a*(p*p) - 0.109375*(b*b*b*b*b)*(d*d*d*d) - 3.0078125*(b*b*b)*(d*d)*p - 8.37890625*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8610(double a, double b, double p, double d, double s){
	return ((0.0146484375*(a*a*a*a)*(d*d*d*d) - 0.15625*(a*a*a)*b*(d*d*d*d) + 0.41015625*(a*a)*(b*b)*(d*d*d*d) + 0.4833984375*(a*a)*(d*d)*p - 0.328125*a*(b*b*b)*(d*d*d*d) - 1.546875*a*b*(d*d)*p + 0.068359375*(b*b*b*b)*(d*d*d*d) + 0.90234375*(b*b)*(d*d)*p + 0.733154296875*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8611(double a, double b, double p, double d, double s){
	return (d*(0.009765625*(a*a*a)*(d*d) - 0.05859375*(a*a)*b*(d*d) + 0.08203125*a*(b*b)*(d*d) + 0.1142578125*a*p - 0.02734375*(b*b*b)*(d*d) - 0.15234375*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8612(double a, double b, double p, double d, double s){
	return ((0.003662109375*(a*a)*(d*d) - 0.01171875*a*b*(d*d) + 0.0068359375*(b*b)*(d*d) + 0.0111083984375*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8613(double a, double b, double p, double d, double s){
	return (d*(0.000732421875*a - 0.0009765625*b)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8614(double a, double b, double p, double d, double s){
	return (6.103515625e-5/(p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU870(double a, double b, double p, double d, double s){
	return (d*((a*a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) + 14.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 52.5*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 52.5*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 6.5625*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p*p) - 28.0*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 294.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 735.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 367.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p*p) + 10.5*(a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 441.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 2756.25*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 3858.75*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 620.15625*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p*p) - 210.0*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 3675.0*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 12862.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 8268.75*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p*p) + 26.25*(a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 1837.5*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 16078.125*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 28940.625*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p*p) + 5684.765625*(a*a*a)*(d*d)*(p*p*p*p*p*p) - 315.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 7717.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 34728.75*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 27286.875*(a*a)*b*(d*d)*(p*p*p*p*p*p) + 13.125*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 1286.25*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 14470.3125*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) + 31834.6875*a*(b*b)*(d*d)*(p*p*p*p*p*p) + 7390.1953125*a*(p*p*p*p*p*p*p) - 52.5*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 1653.75*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 9095.625*(b*b*b)*(d*d)*(p*p*p*p*p*p) - 8445.9375*b*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU871(double a, double b, double p, double d, double s){
	return ((-4.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) - 42.0*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 105.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 52.5*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p*p) + 3.5*(a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) + 147.0*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 918.75*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 1286.25*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 206.71875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p*p) - 126.0*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 2205.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 7717.5*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 4961.25*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p*p) + 26.25*(a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 1837.5*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 16078.125*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 28940.625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 5684.765625*(a*a*a*a)*(d*d*d*d)*(p*p*p*p*p) - 525.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 12862.5*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 57881.25*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 45478.125*(a*a*a)*b*(d*d*d*d)*(p*p*p*p*p) + 39.375*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 3858.75*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 43410.9375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 95504.0625*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p*p) + 22170.5859375*(a*a)*(d*d)*(p*p*p*p*p*p) - 367.5*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 11576.25*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 63669.375*a*(b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 59121.5625*a*b*(d*d)*(p*p*p*p*p*p) + 6.5625*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 826.875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 11369.53125*(b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) + 29560.78125*(b*b)*(d*d)*(p*p*p*p*p*p) + 7918.06640625*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU872(double a, double b, double p, double d, double s){
	return (d*(7.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 52.5*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 78.75*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 13.125*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 14.0*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 294.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 1102.5*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 735.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 5.25*(a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 441.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 4134.375*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 7717.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 1550.390625*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 210.0*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 5512.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 25725.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 20671.875*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 26.25*(a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 2756.25*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 32156.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 72351.5625*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 17054.296875*(a*a*a)*(d*d)*(p*p*p*p*p) - 472.5*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 15435.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 86821.875*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 81860.625*(a*a)*b*(d*d)*(p*p*p*p*p) + 19.6875*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 2572.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 36175.78125*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 95504.0625*a*(b*b)*(d*d)*(p*p*p*p*p) + 25865.68359375*a*(p*p*p*p*p*p) - 105.0*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 4134.375*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p) - 27286.875*(b*b*b)*(d*d)*(p*p*p*p*p) - 29560.78125*b*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU873(double a, double b, double p, double d, double s){
	return ((-7.0*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 35.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 26.25*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 24.5*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 306.25*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 643.125*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 137.8125*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 21.0 *(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 735.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 3858.75*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 3307.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 4.375*(a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 612.5*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 8039.0625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 19293.75*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 4737.3046875*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 175.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 6431.25*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 38587.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 37898.4375*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 13.125*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 1929.375*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 28940.625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 79586.71875*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 22170.5859375*(a*a)*(d*d)*(p*p*p*p*p) - 183.75*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 7717.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 53057.8125*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 59121.5625*a*b*(d*d)*(p*p*p*p*p) + 3.28125*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 551.25*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 9474.609375*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 29560.78125*(b*b)*(d*d)*(p*p*p*p*p) + 9237.744140625*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU874(double a, double b, double p, double d, double s){
	return (d*(4.375*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 13.125*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 3.28125*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 24.5*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 183.75*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 183.75*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 36.75*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 689.0625*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 1929.375*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 516.796875*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 17.5*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 918.75*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 6431.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 6890.625*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 2.1875*(a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 459.375*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 8039.0625*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 24117.1875*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 7105.95703125*(a*a*a)*(d*d)*(p*p*p*p) - 78.75*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 3858.75*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 28940.625*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 34108.59375*(a*a)*b*(d*d)*(p*p*p*p) + 3.28125*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 643.125*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 12058.59375*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 39793.359375*a*(b*b)*(d*d)*(p*p*p*p) + 12932.841796875*a*(p*p*p*p*p) - 26.25*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 1378.125*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 11369.53125*(b*b*b)*(d*d)*(p*p*p*p) - 14780.390625*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU875(double a, double b, double p, double d, double s){
	return ((-1.75*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 2.625*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 15.3125*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 64.3125*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 20.671875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 36.75*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 385.875*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 496.125*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 30.625*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 803.90625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 2894.0625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 947.4609375*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 8.75*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 643.125*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 5788.125*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 7579.6875*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 0.65625*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 192.9375*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 4341.09375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 15917.34375*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 5542.646484375*(a*a)*(d*d)*(p*p*p*p) - 18.375*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1157.625*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 10611.5625*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 14780.390625*a*b*(d*d)*(p*p*p*p) + 0.328125*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 82.6875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 1894.921875*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 7390.1953125*(b*b)*(d*d)*(p*p*p*p) + 2771.3232421875*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU876(double a, double b, double p, double d, double s){
	return (d*(0.4375*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.21875*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 6.125*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 12.25*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 22.96875*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 128.625*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 51.6796875*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 30.625*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 428.75*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 689.0625*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 15.3125*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 535.9375*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 2411.71875*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 947.4609375*(a*a*a)*(d*d)*(p*p*p) - 2.625*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 257.25*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 2894.0625*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 4547.8125*(a*a)*b*(d*d)*(p*p*p) + 0.109375*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 42.875*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 1205.859375*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 5305.78125*a*(b*b)*(d*d)*(p*p*p) + 2155.4736328125*a*(p*p*p*p) - 1.75*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 137.8125*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 1515.9375*(b*b*b)*(d*d)*(p*p*p) - 2463.3984375*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU877(double a, double b, double p, double d, double s){
	return ((-0.0625*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 1.53125*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 0.984375*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 9.1875*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 23.625*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 19.140625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 137.8125*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 67.67578125*(a*a*a*a)*(d*d*d*d)*(p*p) - 15.3125*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 275.625*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 541.40625*(a*a*a)*b*(d*d*d*d)*(p*p) + 4.59375*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 206.71875*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1136.953125*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 527.87109375*(a*a)*(d*d)*(p*p*p) - 0.4375*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 55.125*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 757.96875*a*(b*b*b)*(d*d*d*d)*(p*p) - 1407.65625*a*b*(d*d)*(p*p*p) + 0.0078125*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 3.9375*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 135.3515625*(b*b*b*b)*(d*d*d*d)*(p*p) + 703.828125*(b*b)*(d*d)*(p*p*p) + 329.91943359375*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU878(double a, double b, double p, double d, double s){
	return (d*(0.00390625*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.21875*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 2.296875*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 1.845703125*(a*a*a*a*a)*(d*d*d*d)*p - 7.65625*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 24.609375*(a*a*a*a)*b*(d*d*d*d)*p + 9.5703125*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 86.1328125*(a*a*a)*(b*b)*(d*d*d*d)*p + 50.7568359375*(a*a*a)*(d*d)*(p*p) - 4.59375*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 103.359375*(a*a)*(b*b*b)*(d*d*d*d)*p - 243.6328125*(a*a)*b*(d*d)*(p*p) + 0.765625*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 43.06640625*a*(b*b*b*b)*(d*d*d*d)*p + 284.23828125*a*(b*b)*(d*d)*(p*p) + 153.96240234375*a*(p*p*p) - 0.03125*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 4.921875*(b*b*b*b*b)*(d*d*d*d)*p - 81.2109375*(b*b*b)*(d*d)*(p*p) - 175.95703125*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU879(double a, double b, double p, double d, double s){
	return ((0.013671875*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.328125*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 1.9140625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 1.8798828125*(a*a*a*a)*(d*d*d*d)*p - 3.828125*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 15.0390625*(a*a*a)*b*(d*d*d*d)*p + 2.87109375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 31.58203125*(a*a)*(b*b)*(d*d*d*d)*p + 21.99462890625*(a*a)*(d*d)*(p*p) - 0.765625*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 21.0546875*a*(b*b*b)*(d*d*d*d)*p - 58.65234375*a*b*(d*d)*(p*p) + 0.0546875*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 3.759765625*(b*b*b*b)*(d*d*d*d)*p + 29.326171875*(b*b)*(d*d)*(p*p) + 18.328857421875*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8710(double a, double b, double p, double d, double s){
	return (d*(0.0205078125*(a*a*a*a*a)*(d*d*d*d) - 0.2734375*(a*a*a*a)*b*(d*d*d*d) + 0.95703125*(a*a*a)*(b*b)*(d*d*d*d) + 1.1279296875*(a*a*a)*(d*d)*p - 1.1484375*(a*a)*(b*b*b)*(d*d*d*d) - 5.4140625*(a*a)*b*(d*d)*p + 0.478515625*a*(b*b*b*b)*(d*d*d*d) + 6.31640625*a*(b*b)*(d*d)*p + 5.132080078125*a*(p*p) - 0.0546875*(b*b*b*b*b)*(d*d*d*d) - 1.8046875*(b*b*b)*(d*d)*p - 5.865234375*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8711(double a, double b, double p, double d, double s){
	return ((0.01708984375*(a*a*a*a)*(d*d*d*d) - 0.13671875*(a*a*a)*b*(d*d*d*d) + 0.287109375*(a*a)*(b*b)*(d*d*d*d) + 0.39990234375*(a*a)*(d*d)*p - 0.19140625*a*(b*b*b)*(d*d*d*d) - 1.06640625*a*b*(d*d)*p + 0.0341796875*(b*b*b*b)*(d*d*d*d) + 0.533203125*(b*b)*(d*d)*p + 0.4998779296875*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8712(double a, double b, double p, double d, double s){
	return (d*(0.008544921875*(a*a*a)*(d*d) - 0.041015625*(a*a)*b*(d*d) + 0.0478515625*a*(b*b)*(d*d) + 0.0777587890625*a*p - 0.013671875*(b*b*b)*(d*d) - 0.0888671875*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8713(double a, double b, double p, double d, double s){
	return ((0.0025634765625*(a*a)*(d*d) - 0.0068359375*a*b*(d*d) + 0.00341796875*(b*b)*(d*d) + 0.00640869140625*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8714(double a, double b, double p, double d, double s){
	return (d*(0.00042724609375*a - 0.00048828125*b)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8715(double a, double b, double p, double d, double s){
	return (3.0517578125e-5/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU880(double a, double b, double p, double d, double s){
	return (((a*a*a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d*d*d) + 14.0*(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d)*p + 52.5*(a*a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*(p*p) + 52.5*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p*p) + 6.5625*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*(p*p*p*p) - 32.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d)*p - 336.0*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*(p*p) - 840.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p*p) - 420.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p*p*p) + 14.0*(a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d)*p + 588.0*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*(p*p) + 3675.0*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p*p) + 5145.0*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p*p*p) + 826.875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p*p*p) - 336.0*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*(p*p) - 5880.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p*p) - 20580.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p*p) - 13230.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p*p*p) + 52.5*(a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*(p*p) + 3675.0*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p*p) + 32156.25*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p*p) + 57881.25*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p*p*p) + 11369.53125*(a*a*a*a)*(d*d*d*d)*(p*p*p*p*p*p) - 840.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p*p) - 20580.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p*p) - 92610.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p*p*p) - 72765.0*(a*a*a)*b*(d*d*d*d)*(p*p*p*p*p*p) + 52.5*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p*p) + 5145.0*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p*p) + 57881.25*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p*p) + 127338.75*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p*p*p) + 29560.78125*(a*a)*(d*d)*(p*p*p*p*p*p*p) - 420.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p*p) - 13230.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p*p) - 72765.0*a*(b*b*b)*(d*d*d*d)*(p*p*p*p*p*p) - 67567.5*a*b*(d*d)*(p*p*p*p*p*p*p) + 6.5625*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p*p) + 826.875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p*p) + 11369.53125*(b*b*b*b)*(d*d*d*d)*(p*p*p*p*p*p) + 29560.78125*(b*b)*(d*d)*(p*p*p*p*p*p*p) + 7918.06640625*(p*p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU881(double a, double b, double p, double d, double s){
	return (d*(-4.0*(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) - 42.0*(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 105.0*(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 52.5*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p*p) + 4.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) + 168.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 1050.0*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 1470.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 236.25*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p*p) - 168.0*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 2940.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 10290.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 6615.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p*p) + 42.0*(a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 2940.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 25725.0*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 46305.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 9095.625*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p*p) - 1050.0*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 25725.0*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 115762.5*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 90956.25*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p*p) + 105.0*(a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 10290.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 115762.5*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 254677.5*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p*p) + 59121.5625*(a*a*a)*(d*d)*(p*p*p*p*p*p) - 1470.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 46305.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 254677.5*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 236486.25*(a*a)*b*(d*d)*(p*p*p*p*p*p) + 52.5*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 6615.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 90956.25*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) + 236486.25*a*(b*b)*(d*d)*(p*p*p*p*p*p) + 63344.53125*a*(p*p*p*p*p*p*p) - 236.25*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 9095.625*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 59121.5625*(b*b*b)*(d*d)*(p*p*p*p*p*p) - 63344.53125*b*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU882(double a, double b, double p, double d, double s){
	return ((7.0*(a*a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) + 52.5*(a*a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 78.75*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 13.125*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*(p*p*p) - 16.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) - 336.0*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 1260.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 840.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p*p) + 7.0*(a*a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d*d*d) + 588.0*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 5512.5*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 10290.0*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 2067.1875*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p*p) - 336.0*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p - 8820.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 41160.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 33075.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p*p) + 52.5*(a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d)*p + 5512.5*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 64312.5*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 144703.125*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 34108.59375*(a*a*a*a)*(d*d*d*d)*(p*p*p*p*p) - 1260.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) - 41160.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 231525.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 218295.0*(a*a*a)*b*(d*d*d*d)*(p*p*p*p*p) + 78.75*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*(p*p) + 10290.0*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 144703.125*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 382016.25*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p*p) + 103462.734375*(a*a)*(d*d)*(p*p*p*p*p*p) - 840.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) - 33075.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) - 218295.0*a*(b*b*b)*(d*d*d*d)*(p*p*p*p*p) - 236486.25*a*b*(d*d)*(p*p*p*p*p*p) + 13.125*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p*p) + 2067.1875*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p*p) + 34108.59375*(b*b*b*b)*(d*d*d*d)*(p*p*p*p*p) + 103462.734375*(b*b)*(d*d)*(p*p*p*p*p*p) + 31672.265625*(p*p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU883(double a, double b, double p, double d, double s){
	return (d*(-7.0*(a*a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 35.0*(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 26.25*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 28.0*(a*a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 350.0*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 735.0*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 157.5*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 28.0*(a*a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 980.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 5145.0*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 4410.0*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 7.0*(a*a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 980.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 12862.5*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 30870.0*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 7579.6875*(a*a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 350.0*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 12862.5*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 77175.0*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 75796.875*(a*a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 35.0*(a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 5145.0*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 77175.0*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 212231.25*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 59121.5625*(a*a*a)*(d*d)*(p*p*p*p*p) - 735.0*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 30870.0*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 212231.25*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 236486.25*(a*a)*b*(d*d)*(p*p*p*p*p) + 26.25*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 4410.0*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 75796.875*a*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 236486.25*a*(b*b)*(d*d)*(p*p*p*p*p) + 73901.953125*a*(p*p*p*p*p*p) - 157.5*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 7579.6875*(b*b*b*b*b)*(d*d*d*d)*(p*p*p*p) - 59121.5625*(b*b*b)*(d*d)*(p*p*p*p*p) - 73901.953125*b*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU884(double a, double b, double p, double d, double s){
	return ((4.375*(a*a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 13.125*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 3.28125*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*(p*p) - 28.0*(a*a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 210.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 210.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*(p*p) + 49.0*(a*a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 918.75*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 2572.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 689.0625*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p*p) - 28.0*(a*a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) - 1470.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 10290.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 11025.0*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p*p) + 4.375*(a*a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d*d*d) + 918.75*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 16078.125*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 48234.375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p*p) + 14211.9140625*(a*a*a*a)*(d*d*d*d)*(p*p*p*p) - 210.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p - 10290.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 77175.0*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 90956.25*(a*a*a)*b*(d*d*d*d)*(p*p*p*p) + 13.125*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d)*p + 2572.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 48234.375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 159173.4375*(a*a)*(b*b)*(d*d*d*d)*(p*p*p*p) + 51731.3671875*(a*a)*(d*d)*(p*p*p*p*p) - 210.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) - 11025.0*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) - 90956.25*a*(b*b*b)*(d*d*d*d)*(p*p*p*p) - 118243.125*a*b*(d*d)*(p*p*p*p*p) + 3.28125*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*(p*p) + 689.0625*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p*p) + 14211.9140625*(b*b*b*b)*(d*d*d*d)*(p*p*p*p) + 51731.3671875*(b*b)*(d*d)*(p*p*p*p*p) + 18475.48828125*(p*p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU885(double a, double b, double p, double d, double s){
	return (d*(-1.75*(a*a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 2.625*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 17.5*(a*a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 73.5*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 23.625*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 49.0*(a*a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 514.5*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 661.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 49.0*(a*a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 1286.25*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 4630.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 1515.9375*(a*a*a*a*a)*(d*d*d*d)*(p*p*p) - 17.5*(a*a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 1286.25*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 11576.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 15159.375*(a*a*a*a)*b*(d*d*d*d)*(p*p*p) + 1.75*(a*a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 514.5*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 11576.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 42446.25*(a*a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 14780.390625*(a*a*a)*(d*d)*(p*p*p*p) - 73.5*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 4630.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 42446.25*(a*a)*(b*b*b)*(d*d*d*d)*(p*p*p) - 59121.5625*(a*a)*b*(d*d)*(p*p*p*p) + 2.625*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 661.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 15159.375*a*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 59121.5625*a*(b*b)*(d*d)*(p*p*p*p) + 22170.5859375*a*(p*p*p*p*p) - 23.625*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 1515.9375*(b*b*b*b*b)*(d*d*d*d)*(p*p*p) - 14780.390625*(b*b*b)*(d*d)*(p*p*p*p) - 22170.5859375*b*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU886(double a, double b, double p, double d, double s){
	return ((0.4375*(a*a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d*d*d) + 0.21875*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d)*p - 7.0*(a*a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 14.0*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d)*p + 30.625*(a*a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 171.5*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d)*p + 68.90625*(a*a*a*a*a*a)*(d*d*d*d*d*d)*(p*p) - 49.0*(a*a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 686.0*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d)*p - 1102.5*(a*a*a*a*a)*b*(d*d*d*d*d*d)*(p*p) + 30.625*(a*a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 1071.875*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 4823.4375*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*(p*p) + 1894.921875*(a*a*a*a)*(d*d*d*d)*(p*p*p) - 7.0*(a*a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) - 686.0*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 7717.5*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*(p*p) - 12127.5*(a*a*a)*b*(d*d*d*d)*(p*p*p) + 0.4375*(a*a)*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d*d*d) + 171.5*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 4823.4375*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 21223.125*(a*a)*(b*b)*(d*d*d*d)*(p*p*p) + 8621.89453125*(a*a)*(d*d)*(p*p*p*p) - 14.0*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p - 1102.5*a*(b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) - 12127.5*a*(b*b*b)*(d*d*d*d)*(p*p*p) - 19707.1875*a*b*(d*d)*(p*p*p*p) + 0.21875*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d)*p + 68.90625*(b*b*b*b*b*b)*(d*d*d*d*d*d)*(p*p) + 1894.921875*(b*b*b*b)*(d*d*d*d)*(p*p*p) + 8621.89453125*(b*b)*(d*d)*(p*p*p*p) + 3695.09765625*(p*p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU887(double a, double b, double p, double d, double s){
	return (d*(-0.0625*(a*a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 1.75*(a*a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 1.125*(a*a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 12.25*(a*a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 31.5*(a*a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 30.625*(a*a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 220.5*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 108.28125*(a*a*a*a*a)*(d*d*d*d)*(p*p) - 30.625*(a*a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 551.25*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 1082.8125*(a*a*a*a)*b*(d*d*d*d)*(p*p) + 12.25*(a*a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 551.25*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 3031.875*(a*a*a)*(b*b)*(d*d*d*d)*(p*p) + 1407.65625*(a*a*a)*(d*d)*(p*p*p) - 1.75*(a*a)*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 220.5*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 3031.875*(a*a)*(b*b*b)*(d*d*d*d)*(p*p) - 5630.625*(a*a)*b*(d*d)*(p*p*p) + 0.0625*a*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 31.5*a*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 1082.8125*a*(b*b*b*b)*(d*d*d*d)*(p*p) + 5630.625*a*(b*b)*(d*d)*(p*p*p) + 2639.35546875*a*(p*p*p*p) - 1.125*(b*b*b*b*b*b*b)*(d*d*d*d*d*d)*p - 108.28125*(b*b*b*b*b)*(d*d*d*d)*(p*p) - 1407.65625*(b*b*b)*(d*d)*(p*p*p) - 2639.35546875*b*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU888(double a, double b, double p, double d, double s){
	return ((0.00390625*(a*a*a*a*a*a*a*a)*(d*d*d*d*d*d*d*d) - 0.25*(a*a*a*a*a*a*a)*b*(d*d*d*d*d*d*d*d) + 3.0625*(a*a*a*a*a*a)*(b*b)*(d*d*d*d*d*d*d*d) + 2.4609375*(a*a*a*a*a*a)*(d*d*d*d*d*d)*p - 12.25*(a*a*a*a*a)*(b*b*b)*(d*d*d*d*d*d*d*d) - 39.375*(a*a*a*a*a)*b*(d*d*d*d*d*d)*p + 19.140625*(a*a*a*a)*(b*b*b*b)*(d*d*d*d*d*d*d*d) + 172.265625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d)*p + 101.513671875*(a*a*a*a)*(d*d*d*d)*(p*p) - 12.25*(a*a*a)*(b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 275.625*(a*a*a)*(b*b*b)*(d*d*d*d*d*d)*p - 649.6875*(a*a*a)*b*(d*d*d*d)*(p*p) + 3.0625*(a*a)*(b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 172.265625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d)*p + 1136.953125*(a*a)*(b*b)*(d*d*d*d)*(p*p) + 615.849609375*(a*a)*(d*d)*(p*p*p) - 0.25*a*(b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) - 39.375*a*(b*b*b*b*b)*(d*d*d*d*d*d)*p - 649.6875*a*(b*b*b)*(d*d*d*d)*(p*p) - 1407.65625*a*b*(d*d)*(p*p*p) + 0.00390625*(b*b*b*b*b*b*b*b)*(d*d*d*d*d*d*d*d) + 2.4609375*(b*b*b*b*b*b)*(d*d*d*d*d*d)*p + 101.513671875*(b*b*b*b)*(d*d*d*d)*(p*p) + 615.849609375*(b*b)*(d*d)*(p*p*p) + 329.91943359375*(p*p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU889(double a, double b, double p, double d, double s){
	return (d*(0.015625*(a*a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.4375*(a*a*a*a*a*a)*b*(d*d*d*d*d*d) + 3.0625*(a*a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 3.0078125*(a*a*a*a*a)*(d*d*d*d)*p - 7.65625*(a*a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 30.078125*(a*a*a*a)*b*(d*d*d*d)*p + 7.65625*(a*a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 84.21875*(a*a*a)*(b*b)*(d*d*d*d)*p + 58.65234375*(a*a*a)*(d*d)*(p*p) - 3.0625*(a*a)*(b*b*b*b*b)*(d*d*d*d*d*d) - 84.21875*(a*a)*(b*b*b)*(d*d*d*d)*p - 234.609375*(a*a)*b*(d*d)*(p*p) + 0.4375*a*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 30.078125*a*(b*b*b*b)*(d*d*d*d)*p + 234.609375*a*(b*b)*(d*d)*(p*p) + 146.630859375*a*(p*p*p) - 0.015625*(b*b*b*b*b*b*b)*(d*d*d*d*d*d) - 3.0078125*(b*b*b*b*b)*(d*d*d*d)*p - 58.65234375*(b*b*b)*(d*d)*(p*p) - 146.630859375*b*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8810(double a, double b, double p, double d, double s){
	return ((0.02734375*(a*a*a*a*a*a)*(d*d*d*d*d*d) - 0.4375*(a*a*a*a*a)*b*(d*d*d*d*d*d) + 1.9140625*(a*a*a*a)*(b*b)*(d*d*d*d*d*d) + 2.255859375*(a*a*a*a)*(d*d*d*d)*p - 3.0625*(a*a*a)*(b*b*b)*(d*d*d*d*d*d) - 14.4375*(a*a*a)*b*(d*d*d*d)*p + 1.9140625*(a*a)*(b*b*b*b)*(d*d*d*d*d*d) + 25.265625*(a*a)*(b*b)*(d*d*d*d)*p + 20.5283203125*(a*a)*(d*d)*(p*p) - 0.4375*a*(b*b*b*b*b)*(d*d*d*d*d*d) - 14.4375*a*(b*b*b)*(d*d*d*d)*p - 46.921875*a*b*(d*d)*(p*p) + 0.02734375*(b*b*b*b*b*b)*(d*d*d*d*d*d) + 2.255859375*(b*b*b*b)*(d*d*d*d)*p + 20.5283203125*(b*b)*(d*d)*(p*p) + 14.6630859375*(p*p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8811(double a, double b, double p, double d, double s){
	return (d*(0.02734375*(a*a*a*a*a)*(d*d*d*d) - 0.2734375*(a*a*a*a)*b*(d*d*d*d) + 0.765625*(a*a*a)*(b*b)*(d*d*d*d) + 1.06640625*(a*a*a)*(d*d)*p - 0.765625*(a*a)*(b*b*b)*(d*d*d*d) - 4.265625*(a*a)*b*(d*d)*p + 0.2734375*a*(b*b*b*b)*(d*d*d*d) + 4.265625*a*(b*b)*(d*d)*p + 3.9990234375*a*(p*p) - 0.02734375*(b*b*b*b*b)*(d*d*d*d) - 1.06640625*(b*b*b)*(d*d)*p - 3.9990234375*b*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8812(double a, double b, double p, double d, double s){
	return ((0.01708984375*(a*a*a*a)*(d*d*d*d) - 0.109375*(a*a*a)*b*(d*d*d*d) + 0.19140625*(a*a)*(b*b)*(d*d*d*d) + 0.31103515625*(a*a)*(d*d)*p - 0.109375*a*(b*b*b)*(d*d*d*d) - 0.7109375*a*b*(d*d)*p + 0.01708984375*(b*b*b*b)*(d*d*d*d) + 0.31103515625*(b*b)*(d*d)*p + 0.333251953125*(p*p))/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8813(double a, double b, double p, double d, double s){
	return (d*(0.0068359375*(a*a*a)*(d*d) - 0.02734375*(a*a)*b*(d*d) + 0.02734375*a*(b*b)*(d*d) + 0.05126953125*a*p - 0.0068359375*(b*b*b)*(d*d) - 0.05126953125*b*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8814(double a, double b, double p, double d, double s){
	return ((0.001708984375*(a*a)*(d*d) - 0.00390625*a*b*(d*d) + 0.001708984375*(b*b)*(d*d) + 0.003662109375*p)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8815(double a, double b, double p, double d, double s){
	return (0.000244140625*d*(a - b)/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ inline double MD_Et_GPU8816(double a, double b, double p, double d, double s){
	return (1.52587890625e-5/(p*p*p*p*p*p*p*p*p*p*p*p*p*p*p*p))*s;
}

__device__ double (*MD_EtArray[])(double, double, double, double, double) = {
    MD_Et_GPU000,MD_Et_GPU010,MD_Et_GPU011,MD_Et_GPU020,MD_Et_GPU021,MD_Et_GPU022,MD_Et_GPU030,MD_Et_GPU031,MD_Et_GPU032,MD_Et_GPU033,MD_Et_GPU040,MD_Et_GPU041,MD_Et_GPU042,MD_Et_GPU043,MD_Et_GPU044,MD_Et_GPU050,
    MD_Et_GPU051,MD_Et_GPU052,MD_Et_GPU053,MD_Et_GPU054,MD_Et_GPU055,MD_Et_GPU060,MD_Et_GPU061,MD_Et_GPU062,MD_Et_GPU063,MD_Et_GPU064,MD_Et_GPU065,MD_Et_GPU066,MD_Et_GPU070,MD_Et_GPU071,MD_Et_GPU072,MD_Et_GPU073,
    MD_Et_GPU074,MD_Et_GPU075,MD_Et_GPU076,MD_Et_GPU077,MD_Et_GPU080,MD_Et_GPU081,MD_Et_GPU082,MD_Et_GPU083,MD_Et_GPU084,MD_Et_GPU085,MD_Et_GPU086,MD_Et_GPU087,MD_Et_GPU088,MD_Et_GPU100,MD_Et_GPU101,MD_Et_GPU110,
    MD_Et_GPU111,MD_Et_GPU112,MD_Et_GPU120,MD_Et_GPU121,MD_Et_GPU122,MD_Et_GPU123,MD_Et_GPU130,MD_Et_GPU131,MD_Et_GPU132,MD_Et_GPU133,MD_Et_GPU134,MD_Et_GPU140,MD_Et_GPU141,MD_Et_GPU142,MD_Et_GPU143,MD_Et_GPU144,
    MD_Et_GPU145,MD_Et_GPU150,MD_Et_GPU151,MD_Et_GPU152,MD_Et_GPU153,MD_Et_GPU154,MD_Et_GPU155,MD_Et_GPU156,MD_Et_GPU160,MD_Et_GPU161,MD_Et_GPU162,MD_Et_GPU163,MD_Et_GPU164,MD_Et_GPU165,MD_Et_GPU166,MD_Et_GPU167,
    MD_Et_GPU170,MD_Et_GPU171,MD_Et_GPU172,MD_Et_GPU173,MD_Et_GPU174,MD_Et_GPU175,MD_Et_GPU176,MD_Et_GPU177,MD_Et_GPU178,MD_Et_GPU180,MD_Et_GPU181,MD_Et_GPU182,MD_Et_GPU183,MD_Et_GPU184,MD_Et_GPU185,MD_Et_GPU186,
    MD_Et_GPU187,MD_Et_GPU188,MD_Et_GPU189,MD_Et_GPU200,MD_Et_GPU201,MD_Et_GPU202,MD_Et_GPU210,MD_Et_GPU211,MD_Et_GPU212,MD_Et_GPU213,MD_Et_GPU220,MD_Et_GPU221,MD_Et_GPU222,MD_Et_GPU223,MD_Et_GPU224,MD_Et_GPU230,
    MD_Et_GPU231,MD_Et_GPU232,MD_Et_GPU233,MD_Et_GPU234,MD_Et_GPU235,MD_Et_GPU240,MD_Et_GPU241,MD_Et_GPU242,MD_Et_GPU243,MD_Et_GPU244,MD_Et_GPU245,MD_Et_GPU246,MD_Et_GPU250,MD_Et_GPU251,MD_Et_GPU252,MD_Et_GPU253,
    MD_Et_GPU254,MD_Et_GPU255,MD_Et_GPU256,MD_Et_GPU257,MD_Et_GPU260,MD_Et_GPU261,MD_Et_GPU262,MD_Et_GPU263,MD_Et_GPU264,MD_Et_GPU265,MD_Et_GPU266,MD_Et_GPU267,MD_Et_GPU268,MD_Et_GPU270,MD_Et_GPU271,MD_Et_GPU272,
    MD_Et_GPU273,MD_Et_GPU274,MD_Et_GPU275,MD_Et_GPU276,MD_Et_GPU277,MD_Et_GPU278,MD_Et_GPU279,MD_Et_GPU280,MD_Et_GPU281,MD_Et_GPU282,MD_Et_GPU283,MD_Et_GPU284,MD_Et_GPU285,MD_Et_GPU286,MD_Et_GPU287,MD_Et_GPU288,
    MD_Et_GPU289,MD_Et_GPU2810,MD_Et_GPU300,MD_Et_GPU301,MD_Et_GPU302,MD_Et_GPU303,MD_Et_GPU310,MD_Et_GPU311,MD_Et_GPU312,MD_Et_GPU313,MD_Et_GPU314,MD_Et_GPU320,MD_Et_GPU321,MD_Et_GPU322,MD_Et_GPU323,MD_Et_GPU324,
    MD_Et_GPU325,MD_Et_GPU330,MD_Et_GPU331,MD_Et_GPU332,MD_Et_GPU333,MD_Et_GPU334,MD_Et_GPU335,MD_Et_GPU336,MD_Et_GPU340,MD_Et_GPU341,MD_Et_GPU342,MD_Et_GPU343,MD_Et_GPU344,MD_Et_GPU345,MD_Et_GPU346,MD_Et_GPU347,
    MD_Et_GPU350,MD_Et_GPU351,MD_Et_GPU352,MD_Et_GPU353,MD_Et_GPU354,MD_Et_GPU355,MD_Et_GPU356,MD_Et_GPU357,MD_Et_GPU358,MD_Et_GPU360,MD_Et_GPU361,MD_Et_GPU362,MD_Et_GPU363,MD_Et_GPU364,MD_Et_GPU365,MD_Et_GPU366,
    MD_Et_GPU367,MD_Et_GPU368,MD_Et_GPU369,MD_Et_GPU370,MD_Et_GPU371,MD_Et_GPU372,MD_Et_GPU373,MD_Et_GPU374,MD_Et_GPU375,MD_Et_GPU376,MD_Et_GPU377,MD_Et_GPU378,MD_Et_GPU379,MD_Et_GPU3710,MD_Et_GPU380,MD_Et_GPU381,
    MD_Et_GPU382,MD_Et_GPU383,MD_Et_GPU384,MD_Et_GPU385,MD_Et_GPU386,MD_Et_GPU387,MD_Et_GPU388,MD_Et_GPU389,MD_Et_GPU3810,MD_Et_GPU3811,MD_Et_GPU400,MD_Et_GPU401,MD_Et_GPU402,MD_Et_GPU403,MD_Et_GPU404,MD_Et_GPU410,
    MD_Et_GPU411,MD_Et_GPU412,MD_Et_GPU413,MD_Et_GPU414,MD_Et_GPU415,MD_Et_GPU420,MD_Et_GPU421,MD_Et_GPU422,MD_Et_GPU423,MD_Et_GPU424,MD_Et_GPU425,MD_Et_GPU426,MD_Et_GPU430,MD_Et_GPU431,MD_Et_GPU432,MD_Et_GPU433,
    MD_Et_GPU434,MD_Et_GPU435,MD_Et_GPU436,MD_Et_GPU437,MD_Et_GPU440,MD_Et_GPU441,MD_Et_GPU442,MD_Et_GPU443,MD_Et_GPU444,MD_Et_GPU445,MD_Et_GPU446,MD_Et_GPU447,MD_Et_GPU448,MD_Et_GPU450,MD_Et_GPU451,MD_Et_GPU452,
    MD_Et_GPU453,MD_Et_GPU454,MD_Et_GPU455,MD_Et_GPU456,MD_Et_GPU457,MD_Et_GPU458,MD_Et_GPU459,MD_Et_GPU460,MD_Et_GPU461,MD_Et_GPU462,MD_Et_GPU463,MD_Et_GPU464,MD_Et_GPU465,MD_Et_GPU466,MD_Et_GPU467,MD_Et_GPU468,
    MD_Et_GPU469,MD_Et_GPU4610,MD_Et_GPU470,MD_Et_GPU471,MD_Et_GPU472,MD_Et_GPU473,MD_Et_GPU474,MD_Et_GPU475,MD_Et_GPU476,MD_Et_GPU477,MD_Et_GPU478,MD_Et_GPU479,MD_Et_GPU4710,MD_Et_GPU4711,MD_Et_GPU480,MD_Et_GPU481,
    MD_Et_GPU482,MD_Et_GPU483,MD_Et_GPU484,MD_Et_GPU485,MD_Et_GPU486,MD_Et_GPU487,MD_Et_GPU488,MD_Et_GPU489,MD_Et_GPU4810,MD_Et_GPU4811,MD_Et_GPU4812,MD_Et_GPU500,MD_Et_GPU501,MD_Et_GPU502,MD_Et_GPU503,MD_Et_GPU504,
    MD_Et_GPU505,MD_Et_GPU510,MD_Et_GPU511,MD_Et_GPU512,MD_Et_GPU513,MD_Et_GPU514,MD_Et_GPU515,MD_Et_GPU516,MD_Et_GPU520,MD_Et_GPU521,MD_Et_GPU522,MD_Et_GPU523,MD_Et_GPU524,MD_Et_GPU525,MD_Et_GPU526,MD_Et_GPU527,
    MD_Et_GPU530,MD_Et_GPU531,MD_Et_GPU532,MD_Et_GPU533,MD_Et_GPU534,MD_Et_GPU535,MD_Et_GPU536,MD_Et_GPU537,MD_Et_GPU538,MD_Et_GPU540,MD_Et_GPU541,MD_Et_GPU542,MD_Et_GPU543,MD_Et_GPU544,MD_Et_GPU545,MD_Et_GPU546,
    MD_Et_GPU547,MD_Et_GPU548,MD_Et_GPU549,MD_Et_GPU550,MD_Et_GPU551,MD_Et_GPU552,MD_Et_GPU553,MD_Et_GPU554,MD_Et_GPU555,MD_Et_GPU556,MD_Et_GPU557,MD_Et_GPU558,MD_Et_GPU559,MD_Et_GPU5510,MD_Et_GPU560,MD_Et_GPU561,
    MD_Et_GPU562,MD_Et_GPU563,MD_Et_GPU564,MD_Et_GPU565,MD_Et_GPU566,MD_Et_GPU567,MD_Et_GPU568,MD_Et_GPU569,MD_Et_GPU5610,MD_Et_GPU5611,MD_Et_GPU570,MD_Et_GPU571,MD_Et_GPU572,MD_Et_GPU573,MD_Et_GPU574,MD_Et_GPU575,
    MD_Et_GPU576,MD_Et_GPU577,MD_Et_GPU578,MD_Et_GPU579,MD_Et_GPU5710,MD_Et_GPU5711,MD_Et_GPU5712,MD_Et_GPU580,MD_Et_GPU581,MD_Et_GPU582,MD_Et_GPU583,MD_Et_GPU584,MD_Et_GPU585,MD_Et_GPU586,MD_Et_GPU587,MD_Et_GPU588,
    MD_Et_GPU589,MD_Et_GPU5810,MD_Et_GPU5811,MD_Et_GPU5812,MD_Et_GPU5813,MD_Et_GPU600,MD_Et_GPU601,MD_Et_GPU602,MD_Et_GPU603,MD_Et_GPU604,MD_Et_GPU605,MD_Et_GPU606,MD_Et_GPU610,MD_Et_GPU611,MD_Et_GPU612,MD_Et_GPU613,
    MD_Et_GPU614,MD_Et_GPU615,MD_Et_GPU616,MD_Et_GPU617,MD_Et_GPU620,MD_Et_GPU621,MD_Et_GPU622,MD_Et_GPU623,MD_Et_GPU624,MD_Et_GPU625,MD_Et_GPU626,MD_Et_GPU627,MD_Et_GPU628,MD_Et_GPU630,MD_Et_GPU631,MD_Et_GPU632,
    MD_Et_GPU633,MD_Et_GPU634,MD_Et_GPU635,MD_Et_GPU636,MD_Et_GPU637,MD_Et_GPU638,MD_Et_GPU639,MD_Et_GPU640,MD_Et_GPU641,MD_Et_GPU642,MD_Et_GPU643,MD_Et_GPU644,MD_Et_GPU645,MD_Et_GPU646,MD_Et_GPU647,MD_Et_GPU648,
    MD_Et_GPU649,MD_Et_GPU6410,MD_Et_GPU650,MD_Et_GPU651,MD_Et_GPU652,MD_Et_GPU653,MD_Et_GPU654,MD_Et_GPU655,MD_Et_GPU656,MD_Et_GPU657,MD_Et_GPU658,MD_Et_GPU659,MD_Et_GPU6510,MD_Et_GPU6511,MD_Et_GPU660,MD_Et_GPU661,
    MD_Et_GPU662,MD_Et_GPU663,MD_Et_GPU664,MD_Et_GPU665,MD_Et_GPU666,MD_Et_GPU667,MD_Et_GPU668,MD_Et_GPU669,MD_Et_GPU6610,MD_Et_GPU6611,MD_Et_GPU6612,MD_Et_GPU670,MD_Et_GPU671,MD_Et_GPU672,MD_Et_GPU673,MD_Et_GPU674,
    MD_Et_GPU675,MD_Et_GPU676,MD_Et_GPU677,MD_Et_GPU678,MD_Et_GPU679,MD_Et_GPU6710,MD_Et_GPU6711,MD_Et_GPU6712,MD_Et_GPU6713,MD_Et_GPU680,MD_Et_GPU681,MD_Et_GPU682,MD_Et_GPU683,MD_Et_GPU684,MD_Et_GPU685,MD_Et_GPU686,
    MD_Et_GPU687,MD_Et_GPU688,MD_Et_GPU689,MD_Et_GPU6810,MD_Et_GPU6811,MD_Et_GPU6812,MD_Et_GPU6813,MD_Et_GPU6814,MD_Et_GPU700,MD_Et_GPU701,MD_Et_GPU702,MD_Et_GPU703,MD_Et_GPU704,MD_Et_GPU705,MD_Et_GPU706,MD_Et_GPU707,
    MD_Et_GPU710,MD_Et_GPU711,MD_Et_GPU712,MD_Et_GPU713,MD_Et_GPU714,MD_Et_GPU715,MD_Et_GPU716,MD_Et_GPU717,MD_Et_GPU718,MD_Et_GPU720,MD_Et_GPU721,MD_Et_GPU722,MD_Et_GPU723,MD_Et_GPU724,MD_Et_GPU725,MD_Et_GPU726,
    MD_Et_GPU727,MD_Et_GPU728,MD_Et_GPU729,MD_Et_GPU730,MD_Et_GPU731,MD_Et_GPU732,MD_Et_GPU733,MD_Et_GPU734,MD_Et_GPU735,MD_Et_GPU736,MD_Et_GPU737,MD_Et_GPU738,MD_Et_GPU739,MD_Et_GPU7310,MD_Et_GPU740,MD_Et_GPU741,
    MD_Et_GPU742,MD_Et_GPU743,MD_Et_GPU744,MD_Et_GPU745,MD_Et_GPU746,MD_Et_GPU747,MD_Et_GPU748,MD_Et_GPU749,MD_Et_GPU7410,MD_Et_GPU7411,MD_Et_GPU750,MD_Et_GPU751,MD_Et_GPU752,MD_Et_GPU753,MD_Et_GPU754,MD_Et_GPU755,
    MD_Et_GPU756,MD_Et_GPU757,MD_Et_GPU758,MD_Et_GPU759,MD_Et_GPU7510,MD_Et_GPU7511,MD_Et_GPU7512,MD_Et_GPU760,MD_Et_GPU761,MD_Et_GPU762,MD_Et_GPU763,MD_Et_GPU764,MD_Et_GPU765,MD_Et_GPU766,MD_Et_GPU767,MD_Et_GPU768,
    MD_Et_GPU769,MD_Et_GPU7610,MD_Et_GPU7611,MD_Et_GPU7612,MD_Et_GPU7613,MD_Et_GPU770,MD_Et_GPU771,MD_Et_GPU772,MD_Et_GPU773,MD_Et_GPU774,MD_Et_GPU775,MD_Et_GPU776,MD_Et_GPU777,MD_Et_GPU778,MD_Et_GPU779,MD_Et_GPU7710,
    MD_Et_GPU7711,MD_Et_GPU7712,MD_Et_GPU7713,MD_Et_GPU7714,MD_Et_GPU780,MD_Et_GPU781,MD_Et_GPU782,MD_Et_GPU783,MD_Et_GPU784,MD_Et_GPU785,MD_Et_GPU786,MD_Et_GPU787,MD_Et_GPU788,MD_Et_GPU789,MD_Et_GPU7810,MD_Et_GPU7811,
    MD_Et_GPU7812,MD_Et_GPU7813,MD_Et_GPU7814,MD_Et_GPU7815,MD_Et_GPU800,MD_Et_GPU801,MD_Et_GPU802,MD_Et_GPU803,MD_Et_GPU804,MD_Et_GPU805,MD_Et_GPU806,MD_Et_GPU807,MD_Et_GPU808,MD_Et_GPU810,MD_Et_GPU811,MD_Et_GPU812,
    MD_Et_GPU813,MD_Et_GPU814,MD_Et_GPU815,MD_Et_GPU816,MD_Et_GPU817,MD_Et_GPU818,MD_Et_GPU819,MD_Et_GPU820,MD_Et_GPU821,MD_Et_GPU822,MD_Et_GPU823,MD_Et_GPU824,MD_Et_GPU825,MD_Et_GPU826,MD_Et_GPU827,MD_Et_GPU828,
    MD_Et_GPU829,MD_Et_GPU8210,MD_Et_GPU830,MD_Et_GPU831,MD_Et_GPU832,MD_Et_GPU833,MD_Et_GPU834,MD_Et_GPU835,MD_Et_GPU836,MD_Et_GPU837,MD_Et_GPU838,MD_Et_GPU839,MD_Et_GPU8310,MD_Et_GPU8311,MD_Et_GPU840,MD_Et_GPU841,
    MD_Et_GPU842,MD_Et_GPU843,MD_Et_GPU844,MD_Et_GPU845,MD_Et_GPU846,MD_Et_GPU847,MD_Et_GPU848,MD_Et_GPU849,MD_Et_GPU8410,MD_Et_GPU8411,MD_Et_GPU8412,MD_Et_GPU850,MD_Et_GPU851,MD_Et_GPU852,MD_Et_GPU853,MD_Et_GPU854,
    MD_Et_GPU855,MD_Et_GPU856,MD_Et_GPU857,MD_Et_GPU858,MD_Et_GPU859,MD_Et_GPU8510,MD_Et_GPU8511,MD_Et_GPU8512,MD_Et_GPU8513,MD_Et_GPU860,MD_Et_GPU861,MD_Et_GPU862,MD_Et_GPU863,MD_Et_GPU864,MD_Et_GPU865,MD_Et_GPU866,
    MD_Et_GPU867,MD_Et_GPU868,MD_Et_GPU869,MD_Et_GPU8610,MD_Et_GPU8611,MD_Et_GPU8612,MD_Et_GPU8613,MD_Et_GPU8614,MD_Et_GPU870,MD_Et_GPU871,MD_Et_GPU872,MD_Et_GPU873,MD_Et_GPU874,MD_Et_GPU875,MD_Et_GPU876,MD_Et_GPU877,
    MD_Et_GPU878,MD_Et_GPU879,MD_Et_GPU8710,MD_Et_GPU8711,MD_Et_GPU8712,MD_Et_GPU8713,MD_Et_GPU8714,MD_Et_GPU8715,MD_Et_GPU880,MD_Et_GPU881,MD_Et_GPU882,MD_Et_GPU883,MD_Et_GPU884,MD_Et_GPU885,MD_Et_GPU886,MD_Et_GPU887,
    MD_Et_GPU888,MD_Et_GPU889,MD_Et_GPU8810,MD_Et_GPU8811,MD_Et_GPU8812,MD_Et_GPU8813,MD_Et_GPU8814,MD_Et_GPU8815,MD_Et_GPU8816

};


// MD法におけるEに関して，(i,l,t)の対応するデバイス関数を選択
__device__ inline double MD_Et_NonRecursion(int i, int l, int t, double alpha, double beta, double dist){
    return MD_EtArray[4*i*(10+i) + (i+l)*(i+l+1)/2 + t](alpha, beta, alpha+beta, dist, exp(-alpha*beta/(alpha+beta)*dist*dist));
}


// 1電子の計算で使用
inline __device__ double Et_GPU(int i, int l, int t, double alpha, double beta, double dist){
	if( i<0 || l<0 || t<0 || i+l<t){ // 範囲外の処理
        return 0.0;
    }else{
		return MD_EtArray[4*i*(10+i) + (i+l)*(i+l+1)/2 + t](alpha, beta, alpha+beta, dist, 1.0);
	}
}

}