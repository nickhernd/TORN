#ifndef __khrplatform_h_
#define __khrplatform_h_

/*
** Copyright (c) 2008-2018 The Khronos Group Inc.
**
** Permission is hereby granted, free of charge, to any person obtaining a
** copy of this software and/or associated documentation files (the
** "Materials"), to deal in the Materials without restriction, including
** without limitation the rights to use, copy, modify, merge, publish,
** distribute, sublicense, and/or sell copies of the Materials, and to
** permit persons to whom the Materials are furnished to do so, subject to
** the following conditions:
**
** The above copyright notice and this permission notice shall be included
** in all copies or substantial portions of the Materials.
*/

/* Khronos platform-specific types and definitions.
 *
 * The master copy of khrplatform.h is maintained in the Khronos EGL
 * Registry repository at https://github.com/KhronosGroup/EGL-Registry
 */

#include <stdint.h>

typedef int32_t                 khronos_int32_t;
typedef uint32_t                khronos_uint32_t;
typedef int64_t                 khronos_int64_t;
typedef uint64_t                khronos_uint64_t;
typedef signed   char           khronos_int8_t;
typedef unsigned char           khronos_uint8_t;
typedef signed   short int      khronos_int16_t;
typedef unsigned short int      khronos_uint16_t;
typedef intptr_t                khronos_intptr_t;
typedef uintptr_t               khronos_uintptr_t;
typedef signed   long  long int khronos_ssize_t;
typedef unsigned long  long int khronos_usize_t;
typedef float                   khronos_float_t;
typedef khronos_uint64_t        khronos_utime_nanoseconds_t;
typedef khronos_int64_t         khronos_stime_nanoseconds_t;

#define KHRONOS_MAX_ENUM 0x7FFFFFFF

#ifdef __cplusplus
extern "C" {
#endif

#ifndef KHRONOS_APICALL
#  if (defined(_WIN32) || defined(__CYGWIN__)) && !defined(KHRONOS_STATIC)
#    define KHRONOS_APICALL __declspec(dllimport)
#  elif defined (__GNUC__) && __GNUC__ >= 4
#    define KHRONOS_APICALL __attribute__((visibility("default")))
#  else
#    define KHRONOS_APICALL
#  endif
#endif

#ifndef KHRONOS_APIENTRY
#  if defined(_WIN32) && !defined(_WIN32_WCE) && !defined(__SCITECH_SNAP__)
#    define KHRONOS_APIENTRY __stdcall
#  else
#    define KHRONOS_APIENTRY
#  endif
#endif

#ifndef KHRONOS_APIATTRIBUTES
#define KHRONOS_APIATTRIBUTES
#endif

#ifdef __cplusplus
}
#endif

#endif
