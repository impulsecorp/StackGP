#ifndef _RANDOMNESS_HEADER_H
#define _RANDOMNESS_HEADER_H

#define FLOAT_TYPE double

#ifdef USE_BOOST_RANDOM
    #include <boost/random.hpp>
#else
    #include <stdlib.h>
#endif

#include <vector>
#include <limits>

/*
//clamps the first argument between the second two
inline void Clamp(double &a_Arg, const double a_Min, const double a_Max)
{
    //ASSERT(a_Min <= a_Max);

    if (a_Arg < a_Min)
    {
        a_Arg = a_Min;
        return;
    }

    if (a_Arg > a_Max)
    {
        a_Arg = a_Max;
        return;
    }
}*/

//clamps the first argument between the second two
inline void Clamp(FLOAT_TYPE &a_Arg, const FLOAT_TYPE a_Min, const FLOAT_TYPE a_Max)
{
    //ASSERT(a_Min <= a_Max);

    if (a_Arg < a_Min)
    {
        a_Arg = a_Min;
        return;
    }

    if (a_Arg > a_Max)
    {
        a_Arg = a_Max;
        return;
    }
}

//clamps the first argument between the second two
inline void Clamp(int &a_Arg, const int a_Min, const int a_Max)
{
    //ASSERT(a_Min <= a_Max);

    if (a_Arg < a_Min)
    {
        a_Arg = a_Min;
        return;
    }

    if (a_Arg > a_Max)
    {
        a_Arg = a_Max;
        return;
    }
}

//rounds a double up or down depending on its value
inline int Rounded(const double a_Val)
{
    const int t_Integral = static_cast<int>(a_Val);
    const double t_Mantissa = a_Val - t_Integral;

    if (t_Mantissa < 0.5)
    {
        return t_Integral;
    }

    else
    {
        return t_Integral + 1;
    }
}

//rounds a double up or down depending on whether its
//mantissa is higher or lower than offset
inline int RoundUnderOffset(const double a_Val, const double a_Offset)
{
    //ASSERT(a_Offset < 1 && a_Offset > -1); ???!? Should this be a test for the offset
    const int t_Integral = static_cast<int>(a_Val);
    const double t_Mantissa = a_Val - t_Integral;

    if (t_Mantissa < a_Offset)
    {
        return t_Integral;
    }
    else
    {
        return t_Integral + 1;
    }
}


// Scales the value "a", that is in range [a_min .. a_max] into its relative value in the range [tr_min .. tr_max]
// Example: A=2, in the range [0 .. 4] .. we want to scale it to the range [-12 .. 12] .. we get 0..
/*inline void Scale(    double& a,
                      const double a_min,
                      const double a_max,
                      const double a_tr_min,
                      const double a_tr_max)
{
//        ASSERT((a >= a_min) && (a <= a_max));
//        ASSERT(a_min <= a_max);
//        ASSERT(a_tr_min <= a_tr_max);

    const double t_a_r = a_max - a_min;
    const double t_r = a_tr_max - a_tr_min;
    const double rel_a = (a - a_min) / t_a_r;
    a = a_tr_min + t_r * rel_a;
}*/

// Scales the value "a", that is in range [a_min .. a_max] into its relative value in the range [tr_min .. tr_max]
// Example: A=2, in the range [0 .. 4] .. we want to scale it to the range [-12 .. 12] .. we get 0..
inline void Scale(    FLOAT_TYPE& a,
                      const double a_min,
                      const double a_max,
                      const double a_tr_min,
                      const double a_tr_max)
{
//        ASSERT((a >= a_min) && (a <= a_max));
//        ASSERT(a_min <= a_max);
//        ASSERT(a_tr_min <= a_tr_max);

    const double t_a_r = a_max - a_min;
    const double t_r = a_tr_max - a_tr_min;
    const double rel_a = (a - a_min) / t_a_r;
    a = a_tr_min + t_r * rel_a;
}

inline double Abs(double x)
{
    if (x<0)
    {
        return -x;
    }
    else
    {
        return x;
    }
}


class RNG
{
    
#ifdef USE_BOOST_RANDOM
    boost::random::mt19937 gen;
#endif

public:
    // Seeds the random number generator with this value
    void Seed(long seed);

    // Seeds the random number generator with time
    void TimeSeed();

    // Returns randomly either 1 or -1
    int RandPosNeg();

    // Returns a random integer between X and Y
    int RandInt(int x, int y);

    // Returns a random number from a uniform distribution in the range of [0 .. 1]
    double RandFloat();

    // Returns a random number from a uniform distribution in the range of [-1 .. 1]
    double RandFloatSigned();

    // Returns a random number from a gaussian (normal) distribution in the range of [-1 .. 1]
    double RandGaussSigned();

    // Returns an index given a vector of probabilities
    int Roulette(std::vector<double>& a_probs);

	double Uniform(double min, double max)
	{
		double x = RandFloat();
		Scale(x, 0.0, 1.0, min, max);
		return x;
	}
};


#endif
