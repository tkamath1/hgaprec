#include "ratings.hh"
#include "log.hh"
#include <wchar.h>

int
Ratings::read(string s)
{
  fprintf(stdout, "+ reading ratings dataset from %s\n", s.c_str());
  fflush(stdout);

  if (_env.mode == Env::CREATE_TRAIN_TEST_SETS) {
    if (_env.dataset == Env::NETFLIX) {
      for (uint32_t i = 0; i < _env.m; ++i) {
	if (read_netflix_movie(s,i+1) < 0) {
	  lerr("error adding movie %d\n", i);
	  return -1;
	}
      }
    } else if (_env.dataset == Env::MOVIELENS)
      read_movielens(s);
    else  if (_env.dataset == Env::MENDELEY)
      read_mendeley(s);
    else if (_env.dataset == Env::ECHONEST)
      read_echonest(s);
    else if (_env.dataset == Env::NYT)
      read_nyt(s);
  } else {
    read_generic_train(s);
    write_marginal_distributions();
  }
    
  char st[1024];
  sprintf(st, "read %d users, %d movies, %d ratings", 
	  _curr_user_seq, _curr_movie_seq, _nratings);
  _env.n = _curr_user_seq;
  _env.m = _curr_movie_seq;
  Env::plog("statistics", string(st));

  return 0;
}

int
Ratings::read_generic_train(string dir)
{
  char buf[1024];
  sprintf(buf, "%s/train.tsv", dir.c_str());
  FILE *f = fopen(buf, "r");
  if (!f) {
    fprintf(stderr, "error: cannot open file %s:%s", buf, strerror(errno));
    fclose(f);
    exit(-1);
  }

  //if (_env.dataset == Env::NYT) {
  //  read_nyt_titles(dir);
  //  read_nyt_train(f, NULL);
  //} else
  read_generic(f, NULL);
  fclose(f);
  Env::plog("training ratings", _nratings);
  return 0;
}

int
Ratings::read_generic(FILE *f, CountMap *cmap)
{
  assert(f);
  char b[128];
  uint32_t mid = 0, uid = 0, rating = 0;
  while (!feof(f)) {
    if (fscanf(f, "%u\t%u\t%u\n", &uid, &mid, &rating) < 0) {
      printf("error: unexpected lines in file\n");
      fclose(f);
      exit(-1);
    }

    IDMap::iterator it = _user2seq.find(uid);
    IDMap::iterator mt = _movie2seq.find(mid);

    if ((it == _user2seq.end() && _curr_user_seq >= _env.n) ||
	(mt == _movie2seq.end() && _curr_movie_seq >= _env.m))
      continue;

    if (input_rating_class(rating) == 0)
      continue;
    
    if (it == _user2seq.end())
      assert(add_user(uid));
    
    if (mt == _movie2seq.end())
      assert(add_movie(mid));

    uint32_t m = _movie2seq[mid];
    uint32_t n = _user2seq[uid];
    
    if (input_rating_class(rating) > 0) {
      if (!cmap) {
	_nratings++;
	RatingMap *rm = _users2rating[n];
	if (_env.binary_data)
	  (*rm)[m] = 1;
	else {
	  assert (rating > 0);
	  (*rm)[m] = rating;
	}
	_users[n]->push_back(m);
	_movies[m]->push_back(n);
      } else {
	debug("adding test or validation entry for user %d, item %d", n, m);
	Rating r(n,m);
	assert(cmap);
	if (_env.binary_data)
	  (*cmap)[r] = 1;
	else
	  (*cmap)[r] = rating;
      }
    }
  }
  return 0;
}

int
Ratings::read_nyt_titles(string dir)
{
  char buf[1024];
  sprintf(buf, "%s/nyt-titles.tsv", dir.c_str());
  FILE *f = fopen(buf, "r");
  if (!f) {
    fprintf(stderr, "error: cannot open file %s:%s", buf, strerror(errno));
    fclose(f);
    exit(-1);
  }
  char title[512];
  uint32_t id;
  uint32_t c = 0;
  char *line = (char *)malloc(4096);
  while (!feof(f)) {
    if (fgets(line, 4096, f) == NULL)
      break;
    char *p = line;
    const char r[3]="|";
    char *q = NULL;
    char *d=strtok_r(p, r, &q);
    id = atoi(d);
    IDMap::iterator mt = _movie2seq.find(id);    
    if (mt == _movie2seq.end()) {
      add_movie(id);
      c++;
    }
  }
  fclose(f);
  Env::plog("read titles", c);
  return 0;
}

int
Ratings::read_nyt_train(FILE *f, CountMap *cmap)
{
  assert(f);
  char b[128];
  uint32_t mid = 0, uid = 0, rating = 0;
  while (!feof(f)) {
    if (fscanf(f, "%u\t%u\t%u\n", &uid, &mid, &rating) < 0) {
      printf("error: unexpected lines in file\n");
      fclose(f);
      exit(-1);
    }

    IDMap::iterator it = _user2seq.find(uid);
    IDMap::iterator mt = _movie2seq.find(mid);

    if ((it == _user2seq.end() && _curr_user_seq >= _env.n) ||
	(mt == _movie2seq.end()))
      continue;

    if (input_rating_class(rating) == 0)
      continue;
    
    if (it == _user2seq.end())
      assert(add_user(uid));
    
    assert (mt != _movie2seq.end());

    uint32_t m = _movie2seq[mid];
    uint32_t n = _user2seq[uid];
    
    if (input_rating_class(rating) > 0) {
      if (!cmap) {
	_nratings++;
	RatingMap *rm = _users2rating[n];
	if (_env.binary_data)
	  (*rm)[m] = 1;
	else {
	  assert (rating > 0);
	  (*rm)[m] = rating;
	}
	_users[n]->push_back(m);
	_movies[m]->push_back(n);
      } else {
	debug("adding test or validation entry for user %d, item %d", n, m);
	Rating r(n,m);
	assert(cmap);
	if (_env.binary_data)
	  (*cmap)[r] = 1;
	else
	  (*cmap)[r] = rating;
      }
    }
    if (_nratings % 1000 == 0) {
      printf("\r+ read %d users, %d movies, %d ratings", 
	     _curr_user_seq, _curr_movie_seq, _nratings);
      fflush(stdout);
    }
  }
  return 0;
}


int
Ratings::write_marginal_distributions()
{
  FILE *f = fopen(Env::file_str("/byusers.tsv").c_str(), "w");
  uint32_t x = 0;
  uint32_t nusers = 0;
  for (uint32_t n = 0; n < _env.n; ++n) {
    const vector<uint32_t> *movies = get_movies(n);
    IDMap::const_iterator it = seq2user().find(n);
    if (!movies || movies->size() == 0) {
      debug("0 movies for user %d (%d)", n, it->second);
      x++;
      continue;
    }
    uint32_t t = 0;
    for (uint32_t m = 0; m < movies->size(); m++) {
      uint32_t mov = (*movies)[m];
      yval_t y = r(n,mov);
      t += y;
    }
    x = 0;
    fprintf(f, "%d\t%d\t%lu\t%d\n", n, it->second, movies->size(), t);
    nusers++;
  }
  fclose(f);
  //_env.n = nusers;
  lerr("longest sequence of users with no movies: %d", x);

  f = fopen(Env::file_str("/byitems.tsv").c_str(), "w");
  x = 0;
  uint32_t nitems = 0;
  for (uint32_t n = 0; n < _env.m; ++n) {
    const vector<uint32_t> *users = get_users(n);
    IDMap::const_iterator it = seq2movie().find(n);
    if (!users || users->size() == 0) {
      lerr("0 users for movie %d (%d)", n, it->second);
      x++;
      continue;
    }
    uint32_t t = 0;
    for (uint32_t m = 0; m < users->size(); m++) {
      uint32_t u = (*users)[m];
      yval_t y = r(u,n);
      t += y;
    }
    x = 0;
    fprintf(f, "%d\t%d\t%lu\t%d\n", n, it->second, users->size(), t);
    nitems++;
  }
  fclose(f);
  //_env.m = nitems;
  lerr("longest sequence of items with no users: %d", x);
  Env::plog("post pruning nusers:", _env.n);
  Env::plog("post pruning nitems:", _env.m);
  return 0;
}

int
Ratings::read_test_users(FILE *f, UserMap *bmap)
{
  assert (bmap);
  uint32_t uid = 0;
  while (!feof(f)) {
    if (fscanf(f, "%u\n", &uid) < 0) {
      printf("error: unexpected lines in file\n");
      exit(-1);
    }

    IDMap::iterator it = _user2seq.find(uid);
    if (it == _user2seq.end())
      continue;
    uint32_t n = _user2seq[uid];
    (*bmap)[n] = true;
  }
  Env::plog("read %d test users", bmap->size());
  return 0;
}

int
Ratings::read_echonest(string dir)
{
  printf("reading echo nest dataset...\n");
  fflush(stdout);
  uint32_t mcurr = 1, scurr = 1;
  char buf[1024];
  sprintf(buf, "%s/train_triplets.txt", dir.c_str());

  FILE *f = fopen(buf, "r");
  if (!f) {
    fprintf(stderr, "error: cannot open file %s:%s", buf, strerror(errno));
    fclose(f);
    exit(-1);
  }
  uint32_t mid = 0, uid = 0, rating = 0;
  char mids[512], uids[512];
  char b[128];
  while (!feof(f)) {
    if (fscanf(f, "%s\t%s\t%u\n", uids, mids, &rating) < 0) {
      printf("error: unexpected lines in file\n");
      fclose(f);
      exit(-1);
    }

    StrMap::iterator uiditr = _str2id.find(uids);
    if (uiditr == _str2id.end()) {
      _str2id[uids] = scurr;
      scurr++;
    }
    uid = _str2id[uids];
    
    StrMap::iterator miditr = _str2id.find(mids);
    if (miditr == _str2id.end()) {
      _str2id[mids] = mcurr;
      mcurr++;
    }
    mid = _str2id[mids];

    IDMap::iterator it = _user2seq.find(uid);
    if (it == _user2seq.end() && !add_user(uid)) {
      printf("error: exceeded user limit %d, %d, %d\n",
	     uid, mid, rating);
      fflush(stdout);
      continue;
    }
    
    IDMap::iterator mt = _movie2seq.find(mid);
    if (mt == _movie2seq.end() && !add_movie(mid)) {
      printf("error: exceeded movie limit %d, %d, %d\n",
	     uid, mid, rating);
      fflush(stdout);
      continue;
    }
    
    uint32_t m = _movie2seq[mid];
    uint32_t n = _user2seq[uid];

    _user2str[n] = uids;
    _movie2str[m] = mids;

    if (rating > 0) {
      _nratings++;
      RatingMap *rm = _users2rating[n];
      (*rm)[m] = rating;
      _users[n]->push_back(m);
      _movies[m]->push_back(n);
      _ratings.push_back(Rating(n,m));
    }
    if (_nratings % 1000 == 0) {
      printf("\r+ read %d users, %d movies, %d ratings", 
	     _curr_user_seq, _curr_movie_seq, _nratings);
      fflush(stdout);
    }
  }
  fclose(f);
  return 0;
}

int
Ratings::read_nyt(string dir)
{
  printf("reading nyt dataset...\n");
  fflush(stdout);
  uint32_t mcurr = 1, scurr = 1;
  char buf[1024];
  sprintf(buf, "%s/nyt-clicks.tsv", dir.c_str());

  FILE *f = fopen(buf, "r");
  if (!f) {
    fprintf(stderr, "error: cannot open file %s:%s", buf, strerror(errno));
    fclose(f);
    exit(-1);
  }
  uint32_t mid = 0, uid = 0, rating = 0;
  char mids[512], uids[512];
  char b[128];
  while (!feof(f)) {
    if (fscanf(f, "%s\t%s\t%u\n", uids, mids, &rating) < 0) {
      printf("error: unexpected lines in file\n");
      fclose(f);
      exit(-1);
    }

    StrMap::iterator uiditr = _str2id.find(uids);
    if (uiditr == _str2id.end()) {
      _str2id[uids] = scurr;
      scurr++;
    }
    uid = _str2id[uids];
    
    StrMap::iterator miditr = _str2id.find(mids);
    if (miditr == _str2id.end()) {
      _str2id[mids] = mcurr;
      mcurr++;
    }
    mid = _str2id[mids];

    IDMap::iterator it = _user2seq.find(uid);
    if (it == _user2seq.end() && !add_user(uid)) {
      printf("error: exceeded user limit %d, %d, %d\n",
	     uid, mid, rating);
      fflush(stdout);
      continue;
    }
    
    IDMap::iterator mt = _movie2seq.find(mid);
    if (mt == _movie2seq.end() && !add_movie(mid)) {
      printf("error: exceeded movie limit %d, %d, %d\n",
	     uid, mid, rating);
      fflush(stdout);
      continue;
    }
    
    uint32_t m = _movie2seq[mid];
    uint32_t n = _user2seq[uid];

    _user2str[n] = uids;
    _movie2str[m] = mids;

    if (rating > 0) {
      _nratings++;
      RatingMap *rm = _users2rating[n];
      (*rm)[m] = rating;
      _users[n]->push_back(m);
      _movies[m]->push_back(n);
      _ratings.push_back(Rating(n,m));
    }
    if (_nratings % 1000 == 0) {
      printf("\r+ read %d users, %d movies, %d ratings", 
	     _curr_user_seq, _curr_movie_seq, _nratings);
      fflush(stdout);
    }
  }
  fclose(f);

  sprintf(buf, "%s/str2id.tsv", dir.c_str());

  uint32_t q = 0;
  f = fopen(buf, "w");
  for (StrMap::const_iterator i = _str2id.begin(); i != _str2id.end(); ++i) {
    if (strlen(i->first.c_str()) >= strlen("10219231518"))  {
      fprintf(f, "%s\t%d\n", i->first.c_str(), i->second);
      q++;
    }
  }
  fclose(f);
  Env::plog("wrote %d str2id entries", q);
  return 0;
}

int
Ratings::read_mendeley(string dir)
{
  char buf[1024];
  sprintf(buf, "%s/users.dat", dir.c_str());
  
  info("reading from %s\n", buf);

  FILE *f = fopen(buf, "r");
  if (!f) {
    fprintf(stderr, "error: cannot open file %s:%s", buf, strerror(errno));
    fclose(f);
    exit(-1);
  }
  
  uint32_t uid = 1, rating = 0;
  char b[128];
  while (!feof(f)) {
    vector<uint32_t> mids;
    uint32_t len = 0;
    if (fscanf(f, "%u\t", &len) < 0) {
      printf("error: unexpected lines in file\n");
      fclose(f);
      exit(-1);
    }

    uint32_t mid = 0;
    for (uint32_t i = 0; i < len; ++i) {
      if (i == len - 1) {
	if (fscanf(f, "%u\t", &mid) < 0) {
	  printf("error: unexpected lines in file\n");
	  fclose(f);
	  exit(-1);
	}
	mids.push_back(mid);
      } else {
	if (fscanf(f, "%u", &mid) < 0) {
	  printf("error: unexpected lines in file\n");
	  fclose(f);
	  exit(-1);
	}
	mids.push_back(mid);
      }
    }
    
    IDMap::iterator it = _user2seq.find(uid);
    if (it == _user2seq.end() && !add_user(uid)) {
      printf("error: exceeded user limit %d, %d, %d\n",
	     uid, mid, rating);
      fflush(stdout);
      continue;
    }
    
    for (uint32_t idx = 0; idx < mids.size(); ++idx) {
      uint32_t mid = mids[idx];
      IDMap::iterator mt = _movie2seq.find(mid);
      if (mt == _movie2seq.end() && !add_movie(mid)) {
	printf("error: exceeded movie limit %d, %d, %d\n",
	       uid, mid, rating);
	fflush(stdout);
	continue;
      }
      uint32_t m = _movie2seq[mid];
      uint32_t n = _user2seq[uid];

      yval_t rating = 1.0;
      _nratings++;
      RatingMap *rm = _users2rating[n];
      (*rm)[m] = rating;
      _users[n]->push_back(m);
      _movies[m]->push_back(n);
      _ratings.push_back(Rating(n,m));
    }
    uid++;
    if (_nratings % 1000 == 0) {
      printf("\r+ read %d users, %d movies, %d ratings", 
	     _curr_user_seq, _curr_movie_seq, _nratings);
      fflush(stdout);
    }
  }
  fclose(f);
  return 0;
}

int
Ratings::read_netflix_movie(string dir, uint32_t movie)
{
  char buf[1024];
  sprintf(buf, "%s/mv_%.7d.txt", dir.c_str(), movie);

  info("reading from %s\n", buf);

  FILE *f = fopen(buf, "r");
  if (!f) {
    fprintf(stderr, "error: cannot open file %s:%s", buf, strerror(errno));
    fclose(f);
    exit(-1);
  }

  uint32_t mid = 0;
  if (!fscanf(f, "%d:\n", &mid)) {
    fclose(f);
    return -1;
  }
  assert (mid == movie);
  
  IDMap::iterator mt = _movie2seq.find(mid);
  if (mt == _movie2seq.end() && !add_movie(mid)) {
    fclose(f);
    return 0;
  }
  
  uint32_t m = _movie2seq[mid];
  uint32_t uid = 0, rating = 0;
  char b[128];
  while (!feof(f)) {
    if (fscanf(f, "%u,%u,%s\n", &uid, &rating, b) < 0) {
	printf("error: unexpected lines in file\n");
	fclose(f);
	exit(-1);
    }

    IDMap::iterator it = _user2seq.find(uid);
    if (it == _user2seq.end() && !add_user(uid))
      continue;

    uint32_t n = _user2seq[uid];

    if (rating > 0) {
      _nratings++;
      RatingMap *rm = _users2rating[n];
      (*rm)[m] = rating;
      _users[n]->push_back(m);
      _movies[m]->push_back(n);
      _ratings.push_back(Rating(n,m));
    }
  }
  fclose(f);
  printf("\r+ read %d users, %d movies, %d ratings", 
	 _curr_user_seq, _curr_movie_seq, _nratings);
  fflush(stdout);
  return 0;
}

int
Ratings::read_movielens(string dir)
{
  char buf[1024];
  sprintf(buf, "%s/ml-1m_train.tsv", dir.c_str());

  info("reading from %s\n", buf);

  FILE *f = fopen(buf, "r");
  if (!f) {
    fprintf(stderr, "error: cannot open file %s:%s", buf, strerror(errno));
    fclose(f);
    exit(-1);
  }
  
  uint32_t mid = 0, uid = 0, rating = 0;
  char b[128];
  while (!feof(f)) {
    if (fscanf(f, "%u\t%u\t%u\n", &uid, &mid, &rating) < 0) {
      printf("error: unexpected lines in file\n");
      fclose(f);
      exit(-1);
    }

    IDMap::iterator it = _user2seq.find(uid);
    if (it == _user2seq.end() && !add_user(uid)) {
      printf("error: exceeded user limit %d, %d, %d\n",
	     uid, mid, rating);
      fflush(stdout);
      continue;
    }
    
    IDMap::iterator mt = _movie2seq.find(mid);
    if (mt == _movie2seq.end() && !add_movie(mid)) {
      printf("error: exceeded movie limit %d, %d, %d\n",
	     uid, mid, rating);
      fflush(stdout);
      continue;
    }
    
    uint32_t m = _movie2seq[mid];
    uint32_t n = _user2seq[uid];

    if (rating > 0) {
      _nratings++;
      RatingMap *rm = _users2rating[n];
      (*rm)[m] = rating;
      _users[n]->push_back(m);
      _movies[m]->push_back(n);
      _ratings.push_back(Rating(n,m));
    }
  }
  fclose(f);
  return 0;
}


void
Ratings::load_movies_metadata(string s)
{
  if (_env.dataset == Env::MOVIELENS)
    read_movielens_metadata(s);
  else if (_env.dataset == Env::NETFLIX)
    read_netflix_metadata(s);
  else if (_env.dataset == Env::MENDELEY)
    read_mendeley_metadata(s);
}

int
Ratings::read_movielens_metadata(string dir)
{
  uint32_t n = 0;
  char buf[1024];
  sprintf(buf, "movies.tsv");
  FILE *f = fopen(buf, "r");
  assert(f);
  uint32_t id;
  char name[4096];
  char type[4096];
  char *line = (char *)malloc(4096);
  while (!feof(f)) {
    if (fgets(line, 4096, f) == NULL)
      break;
    uint32_t k = 0;
    char *p = line;
    const char r[3] = "#";
    do {
      char *q = NULL;
      char *d = strtok_r(p, r, &q);
      if (q == p)
	break;
      if (k == 0) {
	id = atoi(d);
	id = _movie2seq[id];
      } else if (k == 1) {
	strcpy(name, d);
	_movie_names[id] = name;
	debug("%d -> %s", id, name);
      } else if (k == 2) {
	strcpy(type, d);
	_movie_types[id] = type;
	debug("%d -> %s", id, type);
      }
      p = q;
      k++;
    } while (p != NULL);
    n++;
    debug("read %d lines\n", n);
    memset(line, 0, 4096);
  }
  free(line);
  return 0;
}

int
Ratings::read_netflix_metadata(string dir)
{
  uint32_t n = 0;
  char buf[1024];
  sprintf(buf, "movie_titles.txt");
  FILE *f = fopen(buf, "r");
  assert(f);
  uint32_t id, year;
  char name[4096];
  char *line = (char *)malloc(4096);
  while (!feof(f)) {
    if (fgets(line, 4096, f) == NULL)
      break;
    uint32_t k = 0;
    char *p = line;
    const char r[3] = ",";
    do {
      char *q = NULL;
      char *d = strtok_r(p, r, &q);
      if (q == p)
	break;
      if (k == 0) {
	id = atoi(d);
	lerr("%d: ", id);
	id = _movie2seq[id];
	lerr("%d -> ", id);
      } else if (k == 1) {
	year  = atoi(d); // skip
	lerr("%d", year);
      } else if (k == 2) {
	strcpy(name, d);
	_movie_names[id] = name;
	_movie_types[id] = "";
	lerr("%d -> %s", id, name);
      }
      p = q;
      k++;
    } while (p != NULL);
    n++;
    debug("read %d lines\n", n);
    memset(line, 0, 4096);
  }
  free(line);
  return 0;
}

int
Ratings::read_mendeley_metadata(string dir)
{
  uint32_t n = 0;
  char buf[1024];
  sprintf(buf, "%s/titles.dat", dir.c_str());
  FILE *f = fopen(buf, "r");
  assert(f);
  char name[4096];
  char *line = (char *)malloc(4096);
  uint32_t id = 0;
  while (!feof(f)) {
    if (fgets(line, 4096, f) == NULL)
      break;
    strcpy(name, line);
    uint32_t seq = _movie2seq[id];
    _movie_names[seq] = name;
    id++;
  }
  lerr("read %d lines\n", n);
  free(line);
  return 0;
}


string
Ratings::movies_by_user_s() const
{
  ostringstream sa;
  sa << "\n[\n";
  for (uint32_t i = 0; i < _users.size(); ++i) {
    IDMap::const_iterator it = _seq2user.find(i);
    sa << it->second << ":";
    vector<uint32_t> *v = _users[i];
    if (v)  {
      for (uint32_t j = 0; j < v->size(); ++j) {
	uint32_t m = v->at(j);
	IDMap::const_iterator mt = _seq2movie.find(m);
	sa << mt->second;
	if (j < v->size() - 1)
	  sa << ", ";
      }
      sa << "\n";
    }
  }
  sa << "]";
  return sa.str();
}
