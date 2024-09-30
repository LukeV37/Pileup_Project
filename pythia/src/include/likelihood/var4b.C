#include "AtlasStyle.C"

void var4b(TString muval)
{
  SetAtlasStyle();

  TFile* ff = new TFile(TString("res/h_no_pufr/a1_dataset_4b_FILTER60_")+muval+"_NumEvents10k_MinJetpT25.root");

  const char* chname = "jjmass";

  TH1* h = (TH1*)ff->Get(chname);
  h->GetXaxis()->SetTitle("jj mass [GeV]");

  double xmax = 1e3;
  TF1* fitfun = new TF1("fitfun","[0]*TMath::Landau(pow(x,[1]),[2],[3])",0,xmax);
  double pars[] = {100,1,100,50};
  fitfun->SetParameters(pars);

  TCanvas* c1 = new TCanvas("c1","c1");
  h->Fit(fitfun);
  c1->Modified();
  c1->Print(TString(chname)+"_"+muval+".png");

  TFile* fout = new TFile(TString("bfit_")+muval+".root","RECREATE");
  // normalize the fitting function
  TF1* nfitfun = (TF1*)fitfun->Clone("bfun");
  double sum = nfitfun->Integral(0,xmax);
  nfitfun->SetParameter(0,nfitfun->GetParameter(0)/sum);
  nfitfun->Write();
  fout->Write();
  delete fout;
}

void var4b()
{
  var4b("mu60");
}
