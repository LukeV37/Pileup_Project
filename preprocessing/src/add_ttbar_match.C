#include <TTree.h>
#include <TFile.h>

#include <vector>
using std::vector;

#include <iostream>
using std::cout;
using std::endl;

#include <cmath>

struct Jet {
  Jet(double aE, double aPx, double aPy, double aPz): E(aE), px(aPx), py(aPy), pz(aPz) {}

  double mass() const { return sqrt(E*E - px*px - py*py - pz*pz); }

  double E, px, py, pz;
};

struct Jet2: public Jet {
  Jet2(Jet& j1, Jet& j2, int aI1, int aI2): Jet(j1.E+j2.E, j1.px+j2.px, j1.py+j2.py, j1.pz+j2.pz), i1(aI1), i2(aI2) { wmatch = pow((mass()-mean)/sigma,2); }

  int i1, i2;
  double wmatch;
  static double mean, sigma;
};

struct Jet3: public Jet {
  Jet3(Jet2& jw, Jet& jb, int aIb): Jet(jw.E+jb.E, jw.px+jb.px, jw.py+jb.py, jw.pz+jb.pz), i1(jw.i1), i2(jw.i2), ib(aIb) { tmatch = jw.wmatch + pow((mass()-mean)/sigma,2); }

  bool overlaps(const Jet3& other) { return i1==other.i1 || i2==other.i2 || ib==other.ib; }

  int i1, i2, ib;
  double tmatch;
  static double mean, sigma;
};

double Jet2::mean = 0;
double Jet2::sigma = 1;
double Jet3::mean = 0;
double Jet3::sigma = 1;

void add_ttbar_match()
{
  // set the chi2 match parameters here
  //double mass_match[] = {78,5.4,163,12.8}; // mu=0
  //double mass_match[] = {95,11.4,191,22}; // mu=20
  double mass_match[] = {129,20,249,36}; // mu=60
  //double mass_match[] = {248,46,460,76}; // mu=200

  Jet2::mean  = mass_match[0];
  Jet2::sigma = mass_match[1];
  Jet3::mean  = mass_match[2];
  Jet3::sigma = mass_match[3];

  // the existing tree will be updated!
  TFile* ff = new TFile("dataset.root", "update");

  // fastjet ntuple
  TTree* treefj = (TTree*)ff->Get("fastjet");

  // new branches to add
  vector<int> jet_ttbar_match;
  TBranch* b_jet_ttbar_match = treefj->Branch("jet_ttbar_match", &jet_ttbar_match);

  // existing branches to use

  vector<float>* jet_pt;
  vector<float>* jet_eta;
  vector<float>* jet_phi;
  vector<float>* jet_m;
  treefj->SetBranchAddress("jet_pt", &jet_pt);
  treefj->SetBranchAddress("jet_eta", &jet_eta);
  treefj->SetBranchAddress("jet_phi", &jet_phi);
  treefj->SetBranchAddress("jet_m", &jet_m);

  vector<int>* trk_bcflag;
  vector<int>* jet_ntracks;
  vector<int>* jet_track_index;
  treefj->SetBranchAddress("trk_bcflag", &trk_bcflag);
  treefj->SetBranchAddress("jet_ntracks", &jet_ntracks);
  treefj->SetBranchAddress("jet_track_index", &jet_track_index);

  // loop over fastjet
  int nevfj = treefj->GetEntries();
  cout << "fastjet entries: " << nevfj << endl;
  for (int ievfj = 0; ievfj<nevfj; ++ievfj) {
    if (ievfj%1000==0) { cout << ievfj << '\r'; cout.flush(); }

    jet_pt = 0;
    jet_eta = 0;
    jet_phi = 0;
    jet_m = 0;
    trk_bcflag = 0;
    jet_ntracks = 0;
    jet_track_index = 0;

    treefj->GetEntry(ievfj);

    int njet = jet_ntracks->size();
    jet_ttbar_match = vector<int>(6,-1);
    if (njet>=6) {
      // collect good jets
      vector<Jet> vj, vb;
      for (int ijet = 0; ijet<njet; ++ijet) {
	if ((*jet_pt)[ijet]<20. || fabs((*jet_eta)[ijet])>2.5) continue;
	TLorentzVector v; v.SetPtEtaPhiM((*jet_pt)[ijet],(*jet_eta)[ijet],(*jet_phi)[ijet],(*jet_m)[ijet]); Jet jet(v.E(),v.Px(),v.Py(),v.Pz());
	// figure out the jet b/c flag (truth b-tagging)
	int ntr = (*jet_ntracks)[ijet];
	if (ntr<=0) continue;
	int jtr = (*jet_track_index)[ijet];
	int jet_bcflag = 0;
	for (int itr = jtr; itr<jtr+ntr; ++itr) {
	  int bcflag = (*trk_bcflag)[itr];
	  if (jet_bcflag<5 && bcflag==5) { jet_bcflag = 5; break; }
	  if (jet_bcflag<4 && bcflag==4) { jet_bcflag = 4; break; }
	}
	if (jet_bcflag==5) {
	  vb.push_back(jet);
	} else {
	  vj.push_back(jet);
	}
      }

      int nj = vj.size();
      int nb = vb.size();
      if (nb>=2 && nj>=4) {
	// loop over pairs of non-b-jets and additional b-jets and collect top candidates
	vector<Jet3> tops;
	for (int i1 = 0; i1<nj-1; ++i1) {
	  for (int i2 = i1+1; i2<nj; ++i2) {
	    Jet2 v12(vj[i1], vj[i2], i1, i2);
	    for (int ib = 0; ib<nb; ++ib) {
	      tops.push_back(Jet3(v12, vb[ib], ib));
	    }
	  }
	}

	// probe pairs of top candidates, starting from the best ones, making usre there is no overlaps
	std::sort(tops.begin(), tops.end(), [](const Jet3& j1, const Jet3& j2){ return j1.tmatch<j2.tmatch; });
	int nt = tops.size();
	bool found = false; int k1, k2;
	for (int i1 = 1; i1<nt; ++i1) {
	  for (int i2 = 0; i2<i1; ++i2) {
	    if (!tops[i1].overlaps(tops[i2])) { k1 = i1; k2 = i2; found = true; }
	    if (found) break;
	  }
	  if (found) break;
	}

	if (found) {
	  // record the results
	  jet_ttbar_match[0] = tops[k1].i1;
	  jet_ttbar_match[1] = tops[k1].i2;
	  jet_ttbar_match[2] = tops[k1].ib;
	  jet_ttbar_match[3] = tops[k2].i1;
	  jet_ttbar_match[4] = tops[k2].i2;
	  jet_ttbar_match[5] = tops[k2].ib;
	}
      }
    }

    b_jet_ttbar_match->Fill();
  }

  // save the output
  treefj->Write("",TObject::kOverwrite);
  delete ff;
}
