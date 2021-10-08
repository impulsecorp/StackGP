#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <typeinfo>

#define USE_BOOST_RANDOM

#include "random.h"
#include <cmath>
#include <time.h>
#include <algorithm>
#include <boost/thread.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;





float sigmoid(float x)
{
    return 1.0/(1.0+exp(-x));
}


float rev_sigmoid(float x)
{
    return 1.0-(1.0/(1.0+exp(-x)));
}


float sqr(float x)
{
	return x * x;
}


float rounded(float x)
{
    if (x > 0.5)
        return 1.0;
    else
        return 0.0;
}


float step(float x)
{
	if (x > 0)
		return 1.0;
	else
		return 0.0;
}

float clip(float x, float min, float max)
{
	if (x > max)
	{
		return max;
	}
	else if (x < min)
	{
		return min;
	}
	else
	{
		return x;
	}
}



float mini(float x, float y)
{
    if (x<y)
    {
        return x;
    }
    else
    {
        return y;
    }
}


float maxi(float x, float y)
{
    if (x>y)
    {
        return x;
    }
    else
    {
        return y;
    }
}



int read_x(const char* fname, std::vector< std::vector<float> >& tit_x)
{
    std::ifstream data(fname);
    if (!data.is_open())
        throw std::runtime_error("File couldn't be open");

	// read in the number of columns first
	int columns;
	data >> columns;

    while(!data.eof())
    {
        std::vector<float> row;
        for(int c=0; c<columns; c++)
        {
			float a=0;
            data >> a;
            row.push_back(a);
        }
        tit_x.push_back(row);
    }
    data.close();

	return columns;
}


void read_y(const char* fname, std::vector<float>& tit_y)
{
    std::ifstream data(fname);
    if (!data.is_open())
		throw std::runtime_error("File couldn't be open");

    while(!data.eof())
    {
		float a=0;
        data >> a;
		//if (a == 0) a = -1;
        tit_y.push_back(a);
    }
    data.close();
}


// all tree outputs pass through here
void process_output(std::vector<float>& out)
{
	for (int i = 0; i < out.size(); i++)
	{
		if (1)
		{
			out[i] = sigmoid(out[i]);
		}

		/*if (1)
		{
			out[i] = rounded(out[i]);
		}*/
	}
}


void clip_output(std::vector<float>& out, float min, float max)
{
	for (int i = 0; i < out.size(); i++)
	{
		out[i] = clip(out[i], min, max);
	}
}


float acc_score(std::vector<float>& out, std::vector< std::vector<float> >& x, std::vector<float>& y)
{
	//process_output(output);
	//clip_output(output, 0, 1);

	/*for (int i = 0; i < x.size(); i++)
	{
		out[i] = sigmoid(out[i]);
	}*/

	float hits = 0;
	for (int i = 0; i < x.size(); i++)
	{
		//hits += 2.0 - abs(out[i] - y[i]);
		if (out[i] == y[i])
			hits++;
	}

	return hits / (float)(x.size());
}



bool in(int val, std::vector<int>& arr)
{
	for (int i : arr)
	{
		if (val == i)
			return true;
	}

	return false;
}

std::vector<int> get_subset_idx(RNG& rng, std::vector< std::vector<float> >& x, float fraction = 0.5)
{
	std::vector<int> subs;
	int fr = (int)(fraction*x.size());

	if (fr >= x.size()) fr = x.size() - 1;
	if (fr <= 1) fr = 1;

	// the first entry
	int c = rng.RandInt(0, x.size() - 1);;
	subs.push_back(c);

	for (int i = 1; i < fr; i++)
	{
		// always pick non-repeating index
		do
		{
			c = rng.RandInt(0, x.size()-1);
		} while (in(c, subs));

		subs.push_back(c);
	}

	return subs;
}


void train_test_split(RNG& rng, std::vector< std::vector<float> >& x, std::vector<float>& y,
	std::vector< std::vector<float> >& train_x, std::vector<float>& train_y,
	std::vector< std::vector<float> >& test_x, std::vector<float>& test_y,
	float fraction = 0.8)
{
	std::vector<int> idx = get_subset_idx(rng, x, fraction = fraction);
	std::vector<int> refl_idx;

	train_x.clear();
	train_y.clear();
	test_x.clear();
	test_y.clear();

	// create the other indices
	for (int i = 0; i < x.size(); i++)
	{
		if (!(in(i, idx)))
			refl_idx.push_back(i);
	}

	for (int i = 0; i < idx.size(); i++)
	{
		train_x.push_back(x[idx[i]]);
		train_y.push_back(y[idx[i]]);
	}

	for (int i = 0; i < refl_idx.size(); i++)
	{
		test_x.push_back(x[refl_idx[i]]);
		test_y.push_back(y[refl_idx[i]]);
	}
}


// helper to determine uniqueness of a score
bool score_is_unique(float acc, std::vector<float>& past_scores, float delta)
{
	for (int i = 0; i < past_scores.size(); i++)
	{
		if (abs(acc - past_scores[i]) < delta)
		{
			return false;
		}
	}

	return true;
}



// arity means the stack must be at least this big or the instruction will be equivalent to NOP (won't do anything)
// all instructions with arity>0 (except POP, which doesn't push back a result) will pop the values from the stack, do the operation, and push the result back on the stack
enum Instructions
{
	NOP = 0, // arity 0, no operand
	PUSHV,   // arity 0, 1 operand
	PUSHC,   // arity 0, 1 operand
//	POP,     // arity 1, no operand
	ADD,     // arity 2, no operand
	MUL,     // arity 2, no operand
	DIV,     // arity 2, no operand
	NEG,     // arity 1, no operand
	MIN,     // arity 2, no operand
	MAX,     // arity 2, no operand
	GREATER, // arity 2, no operand
	LESS,    // arity 2, no operand
	EQUAL,   // arity 2, no operand
	SIN,     // arity 1, no operand 
	COS,     // arity 1, no operand 
	EXP,     // arity 1, no operand
	LOG,     // arity 1, no operand
	SQR,     // arity 1, no operand
	SQRT,    // arity 1, no operand
	TANH     // arity 1, no operand
};


void random_inst(float& inst, float& operand, int num_inputs, RNG& rng)
{
	operand = 0;
	inst = (float)(rng.RandInt(0, (int)Instructions::TANH));
	if (inst == (float)(Instructions::PUSHV))
	{
		operand = (float)(rng.RandInt(0, num_inputs - 1));
	}
	if (inst == (float)(Instructions::PUSHC))
	{
		operand = rng.RandFloatSigned() * 5;
	}
}


float eval_prog(std::vector<float>& prog, std::vector< std::vector<float> >& data_x, std::vector<float>& data_y)
{
	// the stack
	std::vector<float> stack;

	// let's run the program
	std::vector<float> out;

	// execute the program
	for (int i = 0; i < data_x.size(); i++)
	{
		stack.clear();
		//for(int k=0; k<data_x[i].size(); k++)
		//	stack.push_back(data_x[i][k]);
		//stack.push_back(0);

		for (int ip = 0; ip < prog.size(); ip += 2)
		{
			// fetch instruction
			float inst = prog[ip];
			float operand = prog[ip + 1];

			if (inst == NOP)
			{
				continue;
			}

			if (inst == PUSHV)
			{
				stack.push_back(data_x[i][(int)operand]);
			}

			if (inst == PUSHC)
			{
				stack.push_back(operand);
			}

			//if ((inst == POP) && (stack.size() > 0))
			//	stack.pop_back();

			if ((inst == ADD) && (stack.size() > 1))
			{
				float x = stack[stack.size() - 1];
				stack.pop_back();
				float y = stack[stack.size() - 1];
				stack.pop_back();
				stack.push_back(x + y);
			}

			if ((inst == MUL) && (stack.size() > 1))
			{
				float x = stack[stack.size() - 1];
				stack.pop_back();
				float y = stack[stack.size() - 1];
				stack.pop_back();
				stack.push_back(x * y);
			}

			if ((inst == DIV) && (stack.size() > 1))
			{
				float x = stack[stack.size() - 1];
				stack.pop_back();
				float y = stack[stack.size() - 1];
				stack.pop_back();
				stack.push_back(x / y);
			}

			if ((inst == NEG) && (stack.size() > 0))
			{
				float x = stack[stack.size() - 1];
				stack.pop_back();
				stack.push_back(-x);
			}

			if ((inst == MIN) && (stack.size() > 1))
			{
				float x = stack[stack.size() - 1];
				stack.pop_back();
				float y = stack[stack.size() - 1];
				stack.pop_back();
				stack.push_back(x > y ? y : x);
			}

			if ((inst == MAX) && (stack.size() > 1))
			{
				float x = stack[stack.size() - 1];
				stack.pop_back();
				float y = stack[stack.size() - 1];
				stack.pop_back();
				stack.push_back(x > y ? x : y);
			}

			if ((inst == GREATER) && (stack.size() > 1))
			{
				float x = stack[stack.size() - 1];
				stack.pop_back();
				float y = stack[stack.size() - 1];
				stack.pop_back();
				stack.push_back((float)(x > y));
			}

			if ((inst == LESS) && (stack.size() > 1))
			{
				float x = stack[stack.size() - 1];
				stack.pop_back();
				float y = stack[stack.size() - 1];
				stack.pop_back();
				stack.push_back((float)(x < y));
			}

			if ((inst == EQUAL) && (stack.size() > 1))
			{
				float x = stack[stack.size() - 1];
				stack.pop_back();
				float y = stack[stack.size() - 1];
				stack.pop_back();
				stack.push_back((float)(x == y));
			}

			if ((inst == SIN) && (stack.size() > 0))
			{
				float x = stack[stack.size() - 1];
				stack.pop_back();
				stack.push_back(sin(x));
			}

			if ((inst == COS) && (stack.size() > 0))
			{
				float x = stack[stack.size() - 1];
				stack.pop_back();
				stack.push_back(cos(x));
			}

			if ((inst == EXP) && (stack.size() > 0))
			{
				float x = stack[stack.size() - 1];
				stack.pop_back();
				stack.push_back(exp(x));
			}

			if ((inst == LOG) && (stack.size() > 0))
			{
				float x = stack[stack.size() - 1];
				stack.pop_back();
				stack.push_back(log(x));
			}

			if ((inst == SQR) && (stack.size() > 0))
			{
				float x = stack[stack.size() - 1];
				stack.pop_back();
				stack.push_back(x*x);
			}

			if ((inst == SQRT) && (stack.size() > 0))
			{
				float x = stack[stack.size() - 1];
				stack.pop_back();
				stack.push_back(sqrt(x));
			}

			if ((inst == TANH) && (stack.size() > 0))
			{
				float x = stack[stack.size() - 1];
				stack.pop_back();
				stack.push_back(tanh(x));
			}
		}

		if (stack.size() == 0)
		{
			// can't possibly have any output
			break;
		}
		else
		{
			float x = sigmoid(stack[stack.size() - 1]);
			/*if (x > 0.5)
			{
				x = 1;
			}
			else
			{
				x = 0;
			}*/
			out.push_back(x);
		}
	}

	if (out.size() == 0)
		return -999;

	float sum = 0;
	for (int i = 0; i < data_x.size(); i++)
	{
		sum += 1 - abs(out[i] - data_y[i]);
	}

	float r = sum / data_x.size();
	if (isnan(r) || isinf(r))
		r = -999;

	return r;
}



bool genome_better(std::pair< std::vector<float>, float >& ls, std::pair< std::vector<float>, float >& rs)
{
	return (ls.second > rs.second);
}


std::vector<float> crossover(std::vector<float>& mom, std::vector<float>& dad, RNG& rng)
{
	/*
	
	// multipoint gene merger - didn't work well
	
	std::vector<float> baby = mom;
	for (int i = 0; i < dad.size(); i += 2)
	{
		if (rng.RandFloat() < 0.5)
		{
			baby[i] = dad[i];
			baby[i + 1] = dad[i + 1];
		}
	}*/

	// two-point crossover

	std::vector<float> baby;
	int p1 = rng.RandInt(0, mom.size()/2 - 1) * 2;
	int p2 = rng.RandInt(0, dad.size()/2 - 1) * 2;
	while (p1 == p2)
	{
		p1 = rng.RandInt(0, mom.size()/2 - 1) * 2;
		p2 = rng.RandInt(0, dad.size()/2 - 1) * 2;
	}

	if (p1 > p2)
	{
		int tmp = p1;
		p1 = p2;
		p2 = tmp;
	}

	for (int i = 0; i < p1; i += 2)
	{
		baby.push_back(mom[i]);
		baby.push_back(mom[i+1]);
	}
	for (int i = p1; i < p2; i += 2)
	{
		baby.push_back(dad[i]);
		baby.push_back(dad[i + 1]);
	}
	for (int i = p2; i < mom.size(); i += 2)
	{
		baby.push_back(mom[i]);
		baby.push_back(mom[i + 1]);
	}

	return baby;
}

void mutate(std::vector<float>& baby, int num_inputs, RNG& rng)
{
	for (int i = 0; i < baby.size(); i += 2)
	{
		if (rng.RandFloat() < 0.1)
		{
			if ((baby[i] == PUSHV) && (rng.RandFloat() < 0.8))
			{
				baby[i + 1] = rng.RandInt(0, num_inputs - 1);
			}
			else if ((baby[i] == PUSHC) && (rng.RandFloat() < 0.8))
			{
				baby[i + 1] = rng.RandFloatSigned() * 5;
			}
			else
			{
				random_inst(baby[i], baby[i + 1], num_inputs, rng);
			}
		}
	}
}


int main(int argc, char *argv[])
{
	po::options_description desc{ "Peter's Genetic Programming Tool - Options" };
	desc.add_options()
		////////////////////////////
		// Startup parameters
		////////////////////////////
		("help,h", "This help screen")
		("train-x,x", po::value<std::string>()->default_value("tit_x.txt"), "Train data file - X")
		("train-y,y", po::value<std::string>()->default_value("tit_y.txt"), "Train data file - Y")
		("ext-validation,v", "Will use validation data specified with parameters -a and -b (--val-x or --val-y)")
		("val-x,a", po::value<std::string>()->default_value("tit_x_test.txt"), "Validation data file - X")
		("val-y,b", po::value<std::string>()->default_value("tit_y_test.txt"), "Validation data file - Y")
		;
	po::variables_map vm;
	try
	{
		store(parse_command_line(argc, argv, desc), vm);
		notify(vm);
	}
	catch (const po::error &ex)
	{
		std::cerr << ex.what() << '\n';
		return 1;
	}


	std::vector< std::vector<float> > data_x;
	std::vector< float > data_y;

	std::vector< std::vector<float> > data_x_val;
	std::vector< float > data_y_val;


	int num_inputs = 0;
	int num_inputs_ext_val = 0;
	bool has_ext_val = false; // todo: be a parameter
	bool has_val = false;

	if (vm.count("train-x") && vm.count("train-y"))
	{
		std::cout << "Reading train data files: " << vm["train-x"].as<std::string>() << " and " << vm["train-y"].as<std::string>() << '\n';

		num_inputs = read_x(vm["train-x"].as<std::string>().c_str(), data_x);
		read_y(vm["train-y"].as<std::string>().c_str(), data_y);
	}

	if (vm.count("ext-validation"))
	{
		if (vm.count("val-x") && vm.count("val-y"))
		{
			std::cout << "Reading validation data files: " << vm["val-x"].as<std::string>() << " and " << vm["val-y"].as<std::string>() << '\n';

			num_inputs_ext_val = read_x(vm["val-x"].as<std::string>().c_str(), data_x_val);
			read_y(vm["val-y"].as<std::string>().c_str(), data_y_val);

			has_ext_val = true;
		}
	}

	if ((has_ext_val) && (num_inputs != num_inputs_ext_val))
		throw std::runtime_error("Number of inputs in training file doesn't match inputs in validation file");

	std::cout << "\n\n";

	RNG rng;
	rng.TimeSeed();






	// the program
	int proglen = 16;
	int popsize = 15000;

	std::vector< std::pair< std::vector<float>, float > > pop;

	for (int p = 0; p < popsize; p++)
	{
		std::vector<float> prog;
		prog.resize(proglen * 2);
		for (int i = 0; i < prog.size(); i += 2)
		{
			random_inst(prog[i], prog[i + 1], num_inputs, rng);
		}

		pop.push_back(std::make_pair(prog, 0));
	}


	// GA
	for (int gen = 0; gen < 100000000; gen++)
	{

		// evaluate all individuals and display their scores
		std::cout << "Generation: " << gen << "\n";
		float best = 0;
		int besti = 0;
		for (int i = 0; i < pop.size(); i++)
		{
			float b = eval_prog(pop[i].first, data_x, data_y);
			pop[i].second = b;
			if (b > best)
			{
				best = b;
				besti = i;
			}
		}
		std::cout << "Best: " << best << "\n";
		for (int i = 0; i < pop[besti].first.size(); i+=2)
		{
			float ins = pop[besti].first[i];
			float op = pop[besti].first[i+1];
			if (ins == NOP)
				std::cout << "NOP"; 
			if (ins == PUSHV)
				std::cout << "PUSHV " << op;
			if (ins == PUSHC)
				std::cout << "PUSHC " << op;
			if (ins == ADD)
				std::cout << "ADD";
			if (ins == MUL)
				std::cout << "MUL";
			if (ins == DIV)
				std::cout << "DIV";
			if (ins == NEG)
				std::cout << "NEG";
			if (ins == MIN)
				std::cout << "MIN";
			if (ins == MAX)
				std::cout << "MAX";
			if (ins == GREATER)
				std::cout << "GREATER";
			if (ins == LESS)
				std::cout << "LESS";
			if (ins == EQUAL)
				std::cout << "EQUAL";
			if (ins == SIN)
				std::cout << "SIN";
			if (ins == COS)
				std::cout << "COS";
			if (ins == EXP)
				std::cout << "EXP";
			if (ins == LOG)
				std::cout << "LOG";
			if (ins == SQR)
				std::cout << "SQR";
			if (ins == SQRT)
				std::cout << "SQRT";
			if (ins == TANH)
				std::cout << "TANH";

			std::cout << " | ";
		}
		std::cout << "\n\n";

		// reproduction

		std::sort(pop.begin(), pop.end(), genome_better);

		std::vector< std::pair< std::vector<float>, float > > newpop;
		pop.resize((int)((float)(pop.size()) * 0.2));

		// make new pop
		for (int i = 0; i < popsize; i++)
		{
			// decide how to make a baby 
			std::vector<float> baby;

			if (i == 0) // elitism
			{
				baby = pop[0].first;
			}
			else
			{
				// crossover? 
				if (rng.RandFloat() < 0.8)
				{
					int idx1 = rng.RandInt(0, pop.size() - 1);
					int idx2 = rng.RandInt(0, pop.size() - 1);
					baby = crossover(pop[idx1].first, pop[idx2].first, rng);
				}
				else
				{
					int idx = rng.RandInt(0, pop.size() - 1);
					baby = pop[idx].first;
					mutate(baby, num_inputs, rng);
				}

				// mutate?
				if (rng.RandFloat() < 0.5)
				{
					mutate(baby, num_inputs, rng);
				}
			}

			newpop.push_back(std::make_pair(baby, 0));
		}
		pop = newpop;
	}











	return 0;
}








