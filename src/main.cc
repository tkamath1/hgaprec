#include "env.hh"
#include "hgaprec.hh"
#include "ratings.hh"

#include <stdlib.h>
#include <string>
#include <sstream>
#include <signal.h>

string Env::prefix = "";
Logger::Level Env::level = Logger::DEBUG;
FILE *Env::_plogf = NULL;
void usage();
void test();

Env *env_global = NULL;
volatile sig_atomic_t sig_handler_active = 0;

void term_handler(int sig)
{
	if (env_global)
	{
		printf("Got signal. Saving model state.\n");
		fflush(stdout);
		env_global->save_state_now = 1;
	}
	else
	{
		signal(sig, SIG_DFL);
		raise(sig);
	}
}

int main(int argc, char **argv)
{
	signal(SIGTERM, term_handler);
	if (argc <= 1)
	{
		printf("hgaprec -dir <netflix-dataset-dir> -n <users>\n"
			   "-m <movies> -k <dims> -a <hyperparameter-(hypp)-a> -b <hypp-b> -c <hypp-c> -d <hypp-d>"
                           " -label <out-dir-tag>\n");
		exit(0);
	}

	string fname;
	uint32_t n = 0, m = 0;
	uint32_t k = 0;
	uint32_t rfreq = 10;
	uint32_t max_iterations = 1000;
	double rand_seed = 0;
	double a = 0.3, b = 0.3, c = 0.3, d = 0.3;
	bool beta_precomputed = false;


	uint32_t i = 0;
	while (i <= argc - 1)
	{
		if (strcmp(argv[i], "-dir") == 0)
		{
			fname = string(argv[++i]);
			fprintf(stdout, "+ dir = %s\n", fname.c_str());
		}
		else if (strcmp(argv[i], "-n") == 0)
		{
			n = atoi(argv[++i]);
			fprintf(stdout, "+ n = %d\n", n);
		}
		else if (strcmp(argv[i], "-m") == 0)
		{
			m = atoi(argv[++i]);
			fprintf(stdout, "+ m = %d\n", m);
		}
		else if (strcmp(argv[i], "-k") == 0)
		{
			k = atoi(argv[++i]);
			fprintf(stdout, "+ k = %d\n", k);
		}
		else if (strcmp(argv[i], "-rfreq") == 0)
		{
			rfreq = atoi(argv[++i]);
			fprintf(stdout, "+ rfreq = %d\n", rfreq);
		}
		else if (strcmp(argv[i], "-max-iterations") == 0)
		{
			max_iterations = atoi(argv[++i]);
			fprintf(stdout, "+ max iterations %d\n", max_iterations);
		}
		else if (strcmp(argv[i], "-a") == 0)
		{
			a = atof(argv[++i]);
		}
		else if (strcmp(argv[i], "-b") == 0)
		{
			b = atof(argv[++i]);
		}
		else if (strcmp(argv[i], "-c") == 0)
		{
			c = atof(argv[++i]);
		}
		else if (strcmp(argv[i], "-d") == 0)
		{
			d = atof(argv[++i]);
		}
		else if (strcmp(argv[i], "-beta-precomputed") == 0)
		{
			beta_precomputed = true;
		}
		++i;
	};

	Env env(n, m, k, fname, false, "", rfreq,
			false, "", false, 0, max_iterations,
			false, "",
			false, a, b, c, d, Env::MENDELEY,
			true, false, false, true,
			false, true, false, false, false, false,
			false, 0,
			false, false, 0, 0,
			false, false, false,
			false, false, false, false, beta_precomputed);
	env_global = &env;

	Ratings ratings(env);
	if (ratings.read(fname.c_str()) < 0)
	{
		fprintf(stderr, "error reading dataset from dir %s; quitting\n", fname.c_str());
		return -1;
	}

	HGAPRec hgaprec(env, ratings);
	if (beta_precomputed)
	{
		hgaprec.load_beta();
	}
	hgaprec.vb_hier();

}
