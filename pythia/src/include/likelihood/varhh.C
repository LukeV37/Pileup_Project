#include "AtlasStyle.C"

void varhh(TString muval)
{
  SetAtlasStyle();

  TFile* ff = new TFile(TString("res/h_no_pufr/a1_dataset_")+muval+"_H1A3_10k.root");

  const char* chname = "hmass";

  TH1* h = (TH1*)ff->Get(chname);
  h->Add((TH1*)ff->Get("amass"));
  h->GetXaxis()->SetTitle("h mass [GeV]");

  double xmax = 1e3;
  TF1* fitfun = new TF1("fitfun","[0]*exp(-0.5*pow((x-[1])/[2],2))",0,xmax);
  double pars[] = {100,165,50};
  fitfun->SetParameters(pars);

  TCanvas* c1 = new TCanvas("c1","c1");
  h->Fit(fitfun,"","",125.,205.);
  c1->Modified();
  c1->Print(TString(chname)+"_"+muval+".png");

  TFile* fout = new TFile(TString("sfit_")+muval+".root","RECREATE");
  // normalize the fitting function
  TF1* nfitfun = (TF1*)fitfun->Clone("sfun");
  double sum = nfitfun->Integral(0,xmax);
  nfitfun->SetParameter(0,nfitfun->GetParameter(0)/sum);
  nfitfun->Write();
  fout->Write();
  delete fout;
}

void varhh()
{
  varhh("mu60");
}
