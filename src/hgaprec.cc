#include "hgaprec.hh"

#ifdef HAVE_NMFLIB
#include "./nmflib/include/common.h"
#include "./nmflib/include/nmfdriver.h"
#endif

HGAPRec::HGAPRec(Env &env, Ratings &ratings)
	: _env(env), _ratings(ratings),
	  _n(env.n), _m(env.m), _k(env.k),
	  _iter(0),
	  _start_time(time(0)),
	  _theta("theta", 0.3, 0.3, _n, _k, &_r),
	  _beta("beta", 0.3, 0.3, _m, _k, &_r),
	  _thetabias("thetabias", 0.3, 0.3, _n, 1, &_r),
	  _betabias("betabias", 0.3, 0.3, _m, 1, &_r),
	  _htheta("htheta", 0.3, 0.3, _n, _k, &_r),
	  _hbeta("hbeta", 0.3, 0.3, _m, _k, &_r),
	  _thetarate("thetarate", env.a, env.a/env.b, _n, &_r),
	  _betarate("betarate", env.c, env.c/env.d, _m, &_r),
	  _theta_mle(_n, _k),
	  _beta_mle(_m, _k),
	  _old_theta_mle(_n, _k),
	  _old_beta_mle(_m, _k),
	  _lda_gamma(NULL), _lda_beta(NULL),
	  _nmf_theta(NULL), _nmf_beta(NULL),
	  _ctr_theta(NULL), _ctr_beta(NULL),
	  _prev_h(.0), _nh(.0),
	  _save_ranking_file(false),
	  _use_rate_as_score(true),
	  _topN_by_user(100),
	  _maxval(0), _minval(65536)
{
	gsl_rng_env_setup();
	const gsl_rng_type *T = gsl_rng_default;
	_r = gsl_rng_alloc(T);
	if (_env.seed)
		gsl_rng_set(_r, _env.seed);
	Env::plog("infer n:", _n);

	_hf = fopen(Env::file_str("/heldout.txt").c_str(), "w");
	if (!_hf)
	{
		printf("cannot open heldout file:%s\n", strerror(errno));
		exit(-1);
	}
	_vf = fopen(Env::file_str("/validation.txt").c_str(), "w");
	if (!_vf)
	{
		printf("cannot open heldout file:%s\n", strerror(errno));
		exit(-1);
	}
	_tf = fopen(Env::file_str("/test.txt").c_str(), "w");
	if (!_tf)
	{
		printf("cannot open heldout file:%s\n", strerror(errno));
		exit(-1);
	}
	_af = fopen(Env::file_str("/logl.txt").c_str(), "w");
	if (!_af)
	{
		printf("cannot open logl file:%s\n", strerror(errno));
		exit(-1);
	}
	_pf = fopen(Env::file_str("/precision.txt").c_str(), "w");
	if (!_pf)
	{
		printf("cannot open logl file:%s\n", strerror(errno));
		exit(-1);
	}
	_df = fopen(Env::file_str("/ndcg.txt").c_str(), "w");
	if (!_df)
	{
		printf("cannot open logl file:%s\n", strerror(errno));
		exit(-1);
	}
	_rf = fopen(Env::file_str("/rmse.txt").c_str(), "w");
	if (!_rf)
	{
		printf("cannot open logl file:%s\n", strerror(errno));
		exit(-1);
	}

	if (!_env.write_training)
		load_validation_and_test_sets();

	Env::plog("htheta shape:", _htheta.sprior());
	Env::plog("htheta rate:", _htheta.rprior());

	Env::plog("hbeta shape:", _hbeta.sprior());
	Env::plog("hbeta rate:", _hbeta.rprior());

	Env::plog("thetarate shape:", _thetarate.sprior());
	Env::plog("thetarate rate:", _thetarate.rprior());

	Env::plog("betarate shape:", _betarate.sprior());
	Env::plog("betarate rate:", _betarate.rprior());
}

HGAPRec::~HGAPRec()
{
	fclose(_hf);
	fclose(_vf);
	fclose(_af);
	fclose(_pf);
	fclose(_tf);
	fclose(_rf);
}

void HGAPRec::load_validation_and_test_sets()
{
	char buf[4096];
	sprintf(buf, "%s/validation.tsv", _env.datfname.c_str());
	FILE *validf = fopen(buf, "r");
	assert(validf);	
	_ratings.read_generic(validf, &_validation_map);
	fclose(validf);

	for (CountMap::const_iterator i = _validation_map.begin();
		 i != _validation_map.end(); ++i)
	{
		const Rating &r = i->first;
		_validation_users_of_movie[r.second]++;
	}

	sprintf(buf, "%s/test.tsv", _env.datfname.c_str());
	FILE *testf = fopen(buf, "r");
	assert(testf);
       	_ratings.read_generic(testf, &_test_map);
	fclose(testf);

	// XXX: keeps one heldout test item for each user
	// assumes leave-one-out
	for (CountMap::const_iterator i = _test_map.begin();
		 i != _test_map.end(); ++i)
	{
		const Rating &r = i->first;
		_leave_one_out[r.first] = r.second;
		debug("adding %d -> %d to leave one out", r.first, r.second);
	}

	printf("+ loaded validation and test sets from %s\n", _env.datfname.c_str());
	fflush(stdout);
	Env::plog("test ratings", _test_map.size());
	Env::plog("validation ratings", _validation_map.size());
}

void HGAPRec::initialize()
{
	_thetarate.initialize2(_k);
	_thetarate.compute_expectations();

	if (!_env.beta_precomputed) {
		_betarate.initialize2(_k);
		_hbeta.initialize();
		_hbeta.initialize_exp();
	}
	_betarate.compute_expectations();

	_htheta.initialize();
	_htheta.initialize_exp();
}

void HGAPRec::get_phi(GPBase<Matrix> &a, uint32_t ai,
					  GPBase<Matrix> &b, uint32_t bi,
					  Array &phi)
{
	assert(phi.size() == a.k() && phi.size() == b.k());
	assert(ai < a.n() && bi < b.n());
	const double **eloga = a.expected_logv().const_data();
	const double **elogb = b.expected_logv().const_data();
	phi.zero();
	for (uint32_t k = 0; k < _k; ++k)
		phi[k] = eloga[ai][k] + elogb[bi][k];
	phi.lognormalize();
}


void HGAPRec::vb_hier()
{
	initialize();

	//lerr("htheta = %s", _htheta.rate_next().s().c_str());

	uint32_t x;
	if (_env.bias)
		x = _k + 2;
	else
		x = _k;

	Array phi(x);

	while (1)
	{
		if (_iter > _env.max_iterations)
		{
			exit(0);
		}
		for (uint32_t n = 0; n < _n; ++n)
		{
			const vector<uint32_t> *movies = _ratings.get_movies(n);
			for (uint32_t j = 0; movies && j < movies->size(); ++j)
			{
				uint32_t m = (*movies)[j];
				yval_t y = _ratings.r(n, m);

				get_phi(_htheta, n, _hbeta, m, phi);

				if (y > 1) {
					phi.scale(y);
				}

				_htheta.update_shape_next1(n, phi);
				if (!_env.beta_precomputed)
				{
					_hbeta.update_shape_next1(m, phi);
				}
			}
		}

		debug("htheta = %s", _htheta.expected_v().s().c_str());
		debug("hbeta = %s", _hbeta.expected_v().s().c_str());
		Array betarowsum(_k);
		_hbeta.sum_rows(betarowsum);
		_htheta.set_prior_rate(_thetarate.expected_v(), _thetarate.expected_logv());
		debug("adding %s to theta rate", _thetarate.expected_v().s().c_str());
		debug("betarowsum %s", betarowsum.s().c_str());
		_htheta.update_rate_next(betarowsum);
		_htheta.swap();
		_htheta.compute_expectations();

		if (!_env.beta_precomputed) {
			Array thetarowsum(_k);
			_htheta.sum_rows(thetarowsum);
			_hbeta.set_prior_rate(_betarate.expected_v(), _betarate.expected_logv());
			_hbeta.update_rate_next(thetarowsum);
			_hbeta.swap();
		}
		_hbeta.compute_expectations();

		Array thetacolsum(_n);
		_htheta.sum_cols(thetacolsum);
		_thetarate.update_shape_next(_k * _thetarate.sprior());
		_thetarate.update_rate_next(thetacolsum);
		debug("thetacolsum = %s", thetacolsum.s().c_str());

		_thetarate.swap();
		_thetarate.compute_expectations();

		if (!_env.beta_precomputed) {
			Array betacolsum(_m);
			_hbeta.sum_cols(betacolsum);
			_betarate.update_shape_next(_k * _betarate.sprior());
			_betarate.update_rate_next(betacolsum);
			debug("betacolsum = %s", betacolsum.s().c_str());

			_betarate.swap();
		}
		_betarate.compute_expectations();

		printf("\r iteration %d", _iter);
		fflush(stdout);
		if (_iter % _env.reportfreq == 0)
		{
			compute_likelihood(true);
			compute_likelihood(false);
			//compute_rmse();
			save_model();
			//compute_precision(false);
			//compute_itemrank(false);
			//gen_ranking_for_users(false);
		}

		if (_env.save_state_now)
		{
			lerr("Saving state at iteration %d duration %d secs", _iter, duration());
			do_on_stop();
		}
		_iter++;
	}
}

void HGAPRec::compute_likelihood(bool validation)
{
	uint32_t k = 0, kzeros = 0, kones = 0;
	double s = .0, szeros = 0, sones = 0;

	CountMap *mp = NULL;
	FILE *ff = NULL;
	if (validation)
	{
		mp = &_validation_map;
		ff = _vf;
	}
	else
	{
		mp = &_test_map;
		ff = _tf;
	}

	for (CountMap::const_iterator i = mp->begin();
		 i != mp->end(); ++i)
	{
		const Rating &e = i->first;
		uint32_t n = e.first;
		uint32_t m = e.second;

		yval_t r = i->second;
		double u = rating_likelihood_hier(n, m, r);
		s += u;
		k += 1;
	}

	double a = .0;
	info("s = %.5f\n", s);
	fprintf(ff, "%d\t%d\t%.9f\t%d\n", _iter, duration(), s / k, k);
	fflush(ff);
	a = s / k;

	if (!validation)
		return;

	bool stop = false;
	int why = -1;
	if (_iter > 30)
	{
		if (a > _prev_h && _prev_h != 0 && fabs((a - _prev_h) / _prev_h) < 0.000001)
		{
			stop = true;
			why = 0;
		}
		else if (a < _prev_h)
			_nh++;
		else if (a > _prev_h)
			_nh = 0;

		if (_nh > 2)
		{ // be robust to small fluctuations in predictive likelihood
			why = 1;
			stop = true;
		}
	}
	_prev_h = a;
	FILE *f = fopen(Env::file_str("/max.txt").c_str(), "w");
	fprintf(f, "%d\t%d\t%.5f\t%d\n",
			_iter, duration(), a, why);
	fclose(f);
	if (stop && _iter >= 250)
	{
		do_on_stop();
		exit(0);
	}
}

double
HGAPRec::rating_likelihood_hier(uint32_t p, uint32_t q, yval_t y) const
{
	const double **etheta = _htheta.expected_v().const_data();
	const double **ebeta = _hbeta.expected_v().const_data();

	double s = .0;
	for (uint32_t k = 0; k < _k; ++k)
		s += etheta[p][k] * ebeta[q][k];

	if (_env.bias)
	{
		const double **ethetabias = _thetabias.expected_v().const_data();
		const double **ebetabias = _betabias.expected_v().const_data();
		s += ethetabias[p][0] + ebetabias[q][0];
	}

	if (s < 1e-30)
		s = 1e-30;

	if (_env.binary_data)
		return y == 0 ? -s : log(1 - exp(-s));
	return y * log(s) - s - log_factorial(y);
}

double
HGAPRec::log_factorial(uint32_t n) const
{
	double v = log(1);
	for (uint32_t i = 2; i <= n; ++i)
		v += log(i);
	return v;
}

void HGAPRec::do_on_stop()
{
	save_model();
	//gen_ranking_for_users(false);
}

void HGAPRec::load_beta()
{
	_betarate.load();
	_hbeta.load();
}

void HGAPRec::save_model()
{
	if (_env.hier)
	{
		_hbeta.save_state(_ratings.seq2movie());
		_betarate.save_state(_ratings.seq2movie());
		_htheta.save_state(_ratings.seq2user());
		_thetarate.save_state(_ratings.seq2user());
	}
	else
	{
		_beta.save_state(_ratings.seq2movie());
		_theta.save_state(_ratings.seq2user());
	}

	//	if (_env.bias)
	//{
	//	_betabias.save_state(_ratings.seq2movie());
	//	_thetabias.save_state(_ratings.seq2user());
	//}
	//if (_env.canny || _env.mle_user || _env.mle_item)
	//{
	//	_theta_mle.save(Env::file_str("/theta_mle.tsv"), _ratings.seq2user());
	//	_beta_mle.save(Env::file_str("/beta_mle.tsv"), _ratings.seq2movie());
	//}
}
