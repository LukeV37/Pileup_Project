#include <TTree.h>
#include <TFile.h>

#include <vector>
using std::vector;

#include <iostream>
using std::cout;
using std::endl;

#include <cmath>

#include "../include/sbtf.C"

void add_Likelihood(TString name)
{
  // the existing tree will be updated!
  TFile* ff = new TFile("../../output/"+name, "update");

  // fastjet ntuple
  TTree* treefj = (TTree*)ff->Get("fastjet");

  // new branches to add
  vector<vector<float>> jet_Likelihood;
  TBranch* b_jet_Likelihood = treefj->Branch("jet_Likelihood", &jet_Likelihood);

  // existing branches to use

  vector<float>* jet_pt;
  vector<float>* jet_eta;
  vector<float>* jet_phi;
  vector<float>* jet_m;

  treefj->SetBranchAddress("jet_pt", &jet_pt);
  treefj->SetBranchAddress("jet_eta", &jet_eta);
  treefj->SetBranchAddress("jet_phi", &jet_phi);
  treefj->SetBranchAddress("jet_m", &jet_m);

  // loop over fastjet
  int nevfj = treefj->GetEntries();
  //cout << "fastjet entries: " << nevfj << endl;
  for (int ievfj = 0; ievfj<nevfj; ++ievfj) {
    if (ievfj%1000==0) { cout << ievfj << '\r'; cout.flush(); }

    jet_pt = 0;
    jet_eta = 0;
    jet_phi = 0;
    jet_m = 0;

    treefj->GetEntry(ievfj);

    float jet1_pt, jet1_eta, jet1_phi, jet1_m;
    float jet2_pt, jet2_eta, jet2_phi, jet2_m;

    // loop over jets
    int njet = jet_pt->size();
    jet_Likelihood = vector(njet, vector<float>(njet,-1));
    for (int ijet = 0; ijet<njet; ++ijet) {
        jet1_pt = (*jet_pt)[ijet];
        jet1_eta = (*jet_eta)[ijet];
        jet1_phi = (*jet_phi)[ijet];
        jet1_m = (*jet_m)[ijet];
        float Likelihood = 0;
        // loop over neighbors
        for (int jjet = ijet; jjet<njet; ++jjet) {
            if (ijet == jjet) {
              jet_Likelihood[ijet][jjet] = Likelihood;
              continue;
            }
            jet2_pt = (*jet_pt)[jjet];
            jet2_eta = (*jet_eta)[jjet];
            jet2_phi = (*jet_phi)[jjet];
            jet2_m = (*jet_m)[jjet];

            // Calculate Likelihood Between pair to originate from H
            float Likelihood = sbtf(jet1_pt, jet1_eta, jet1_phi, jet1_m,
                                    jet2_pt, jet2_eta, jet2_phi, jet2_m);
            jet_Likelihood[ijet][jjet] = Likelihood;
            jet_Likelihood[jjet][ijet] = Likelihood;
        }
    }
    b_jet_Likelihood->Fill();
  }

  // save the output
  treefj->Write("",TObject::kOverwrite);
  delete ff;
}

int main(int argc, char* argv[]){
    char *name = argv[1];
    add_Likelihood(TString(name));
    return 0;
}

