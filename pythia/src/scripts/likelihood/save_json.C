void save_json()
{
  TF1* sfun = (TF1*)((new TFile("sfit_mu60.root"))->Get("sfun"));
  TF1* bfun = (TF1*)((new TFile("bfit_mu60.root"))->Get("bfun"));

  double xmin = 0, xmax = 1e3;
  const int nbin = 1000;
  double tf[nbin];
  for (int i = 0; i<nbin; ++i) {
    double x = xmin + (i+0.5)/nbin*(xmax-xmin);
    double s = sfun->Eval(x);
    double b = bfun->Eval(x);
    double r = s/(s+b);
    tf[i] = r<1e-7||s<1e-7 ? 0:r;
  }

  ofstream fjout("sbfun.json");
  fjout << "{" << endl;
  fjout << "  \"nbin\": " << nbin << "," << endl;
  fjout << "  \"xmin\": " << xmin << "," << endl;
  fjout << "  \"xmax\": " << xmax << "," << endl;
  fjout << "  \"tf\": [" << endl;
  for (int i = 0; i<nbin-1; ++i) fjout << tf[i] << "," << endl;
  fjout << tf[nbin-1] << "]" << endl;
  fjout << "}" << endl;
  fjout.close();
}
