#include <TTree.h>
#include <TFile.h>

#include <vector>
using std::vector;

#include <iostream>
using std::cout;
using std::endl;

#include <cmath>

void add_true_pufr()
{
  // the existing tree will be updated!
  TFile* ff = new TFile("dataset.root", "update");

  // fastjet ntuple
  TTree* treefj = (TTree*)ff->Get("fastjet");

  // new branches to add
  vector<float> jet_pufr_truth;
  TBranch* b_jet_pufr_truth = treefj->Branch("jet_pufr_truth", &jet_pufr_truth);

  // existing branches to use

  vector<float>* trk_pT;
  vector<float>* trk_q;
  vector<int>* trk_label;
  vector<int>* jet_ntracks;
  vector<int>* jet_track_index;

  treefj->SetBranchAddress("trk_pT", &trk_pT);
  treefj->SetBranchAddress("trk_q", &trk_q);
  treefj->SetBranchAddress("trk_label", &trk_label);
  treefj->SetBranchAddress("jet_ntracks", &jet_ntracks);
  treefj->SetBranchAddress("jet_track_index", &jet_track_index);

  // loop over fastjet
  int nevfj = treefj->GetEntries();
  cout << "fastjet entries: " << nevfj << endl;
  for (int ievfj = 0; ievfj<nevfj; ++ievfj) {
    if (ievfj%1000==0) { cout << ievfj << '\r'; cout.flush(); }

    trk_pT = 0;
    trk_q = 0;
    trk_label = 0;
    jet_ntracks = 0;
    jet_track_index = 0;

    treefj->GetEntry(ievfj);

    // loop over jets
    int njet = jet_ntracks->size();
    jet_pufr_truth = vector<float>(njet,-1);
    for (int ijet = 0; ijet<njet; ++ijet) {
      // loop over good tracks
      int ntr = (*jet_ntracks)[ijet];
      int jtr = (*jet_track_index)[ijet];
      double sumpt_all = 0, sumpt_pu = 0;
      for (int itr = jtr; itr<jtr+ntr; ++itr) {
	// skip neutrals
	if ((*trk_q)[itr]==0) continue;
	// skip low pT particles
	double pttr = (*trk_pT)[itr];
	if (pttr<0.4) continue;
	// collect weights
	double weight = pow(pttr,2);
	sumpt_all += weight;
	int label = (*trk_label)[itr];
	if (label>=0) sumpt_pu += weight;
      }
      if (sumpt_all>0) jet_pufr_truth[ijet] = sumpt_pu/sumpt_all;
    }
    b_jet_pufr_truth->Fill();
  }

  // save the output
  treefj->Write("",TObject::kOverwrite);
  delete ff;
}
