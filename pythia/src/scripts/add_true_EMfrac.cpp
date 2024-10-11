#include <TTree.h>
#include <TFile.h>

#include <vector>
using std::vector;

#include <iostream>
using std::cout;
using std::endl;

#include <cmath>

void add_true_EMfrac(TString name)
{
  // the existing tree will be updated!
  TFile* ff = new TFile("../../output/"+name, "update");

  // fastjet ntuple
  TTree* treefj = (TTree*)ff->Get("fastjet");

  // new branches to add
  vector<float> jet_true_Efrac;
  vector<float> jet_true_Mfrac;
  TBranch* b_jet_true_Efrac = treefj->Branch("jet_true_Efrac", &jet_true_Efrac);
  TBranch* b_jet_true_Mfrac = treefj->Branch("jet_true_Mfrac", &jet_true_Mfrac);

  // existing branches to use

  vector<float>* trk_pT;
  vector<float>* trk_eta;
  vector<float>* trk_phi;
  vector<float>* trk_e;
  vector<int>* trk_label;
  vector<int>* jet_ntracks;
  vector<int>* jet_track_index;

  treefj->SetBranchAddress("trk_pT", &trk_pT);
  treefj->SetBranchAddress("trk_eta", &trk_eta);
  treefj->SetBranchAddress("trk_phi", &trk_phi);
  treefj->SetBranchAddress("trk_e", &trk_e);
  treefj->SetBranchAddress("trk_label", &trk_label);
  treefj->SetBranchAddress("jet_ntracks", &jet_ntracks);
  treefj->SetBranchAddress("jet_track_index", &jet_track_index);

  // loop over fastjet
  int nevfj = treefj->GetEntries();
  cout << "fastjet entries: " << nevfj << endl;
  for (int ievfj = 0; ievfj<nevfj; ++ievfj) {
    if (ievfj%1000==0) { cout << ievfj << '\r'; cout.flush(); }

    trk_pT = 0;
    trk_eta = 0;
    trk_phi = 0;
    trk_e = 0;
    trk_label = 0;
    jet_ntracks = 0;
    jet_track_index = 0;

    treefj->GetEntry(ievfj);

    // loop over jets
    int njet = jet_ntracks->size();
    jet_true_Efrac = vector<float>(njet);
    jet_true_Mfrac = vector<float>(njet);
    for (int ijet = 0; ijet<njet; ++ijet) {
      int ntr = (*jet_ntracks)[ijet];
      if (ntr<=0) continue;
      int jtr = (*jet_track_index)[ijet];
      TLorentzVector vtot, vhs;
      for (int itr = jtr; itr<jtr+ntr; ++itr) {
	TLorentzVector v; v.SetPtEtaPhiE((*trk_pT)[itr],(*trk_eta)[itr],(*trk_phi)[itr],(*trk_e)[itr]);
	vtot += v;
	if ((*trk_label)[itr]<0) {
	  vhs += v;
	}
      }
      double true_Efrac = vtot.E(); if (true_Efrac>0) true_Efrac = vhs.E()/true_Efrac;
      double true_Mfrac = vtot.M(); if (true_Mfrac>0) true_Mfrac = vhs.M()/true_Mfrac;
      jet_true_Efrac[ijet] = true_Efrac;
      jet_true_Mfrac[ijet] = true_Mfrac;
    }
    b_jet_true_Efrac->Fill();
    b_jet_true_Mfrac->Fill();
  }

  // save the output
  treefj->Write("",TObject::kOverwrite);
  delete ff;
}
