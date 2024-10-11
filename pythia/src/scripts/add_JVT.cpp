#include <TTree.h>
#include <TFile.h>

#include <vector>
using std::vector;

#include <iostream>
using std::cout;
using std::endl;

#include <cmath>

void add_JVT(TString name)
{
  // the existing tree will be updated!
  TFile* ff = new TFile("../../output/"+name, "update");

  // fastjet ntuple
  TTree* treefj = (TTree*)ff->Get("fastjet");

  // new branches to add
  vector<float> jet_corrJVF;
  TBranch* b_jet_corrJVF = treefj->Branch("jet_corrJVF", &jet_corrJVF);
  vector<float> jet_RpT;
  TBranch* b_jet_RpT = treefj->Branch("jet_RpT", &jet_RpT);

  // existing branches to use

  vector<float>* jet_pT;
  vector<float>* trk_pT;
  vector<float>* trk_q;
  vector<int>* trk_label;
  vector<int>* jet_ntracks;
  vector<int>* jet_track_index;

  treefj->SetBranchAddress("jet_pt", &jet_pT);
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

    jet_pT = 0;
    trk_pT = 0;
    trk_q = 0;
    trk_label = 0;
    jet_ntracks = 0;
    jet_track_index = 0;
    
    float k = 0.01;
    int num_PU = 0;

    treefj->GetEntry(ievfj);

    // First count number of PU tracks per event
    int ntrk = trk_pT->size();
    for (int itrk = 0; itrk<ntrk; ++itrk){
        // skip neutrals
        if ((*trk_q)[itrk]==0) continue;
        // skip low pT particles
        double pttr = (*trk_pT)[itrk];
        if (pttr<0.4) continue;
        // count num_PU
        int label = (*trk_label)[itrk];
        if (label>=0) num_PU++;
    }

    // loop over jets
    int njet = jet_ntracks->size();
    jet_corrJVF = vector<float>(njet,-1);
    jet_RpT = vector<float>(njet,-1);
    for (int ijet = 0; ijet<njet; ++ijet) {
        // loop over good tracks
        int ntr = (*jet_ntracks)[ijet];
        int jtr = (*jet_track_index)[ijet];
        double sumpt_hs = 0, sumpt_pu = 0;
        for (int itr = jtr; itr<jtr+ntr; ++itr) {
	        // skip neutrals
            if ((*trk_q)[itr]==0) continue;
            // skip low pT particles
            double pttr = (*trk_pT)[itr];
            if (pttr<0.4) continue;
            // collect weights
            int label = (*trk_label)[itr];
            double weight = pow(pttr,1);
            if (label==-1) sumpt_hs += weight;
            if (label>=0) sumpt_pu += weight;
        }
        if ((sumpt_hs+sumpt_pu)>0) jet_corrJVF[ijet] = sumpt_hs/(sumpt_hs+sumpt_pu/(k*num_PU));
        if ((*jet_pT)[ijet]>0) jet_RpT[ijet] = sumpt_hs/(*jet_pT)[ijet];
    }
    b_jet_corrJVF->Fill();
    b_jet_RpT->Fill();
  }

  // save the output
  treefj->Write("",TObject::kOverwrite);
  delete ff;
}
