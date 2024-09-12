void find_ip(double pT, double eta, double phi, double xProd, double yProd, double zProd, double& d0, double& z0)
{
  // calculate IP

  double r = sqrt(pow(xProd,2) + pow(yProd,2));
  double dphi = phi - atan2(yProd,xProd);
  if (dphi>M_PI) dphi -= 2*M_PI;
  else if (dphi<=-M_PI) dphi += 2*M_PI;
  d0 = r*sin(dphi);
  z0 = zProd - r*sinh(eta);

  // smear according to resolution
  // tentative resolution parameterization: ATL-COM-PHYS-2021-377, Figs.12-13

  double sigma_d0 = sqrt(pow(80/pT,2) + pow(4,2))*1e-3; // um->mm
  double sigma_z0 = sqrt(pow(80/pT,2) + pow(10,2))*1e-3; // um->mm

  static TRandom3 rnd;
  d0 += rnd.Gaus(0,sigma_d0);
  z0 += rnd.Gaus(0,sigma_z0);
}

int trace_origin(const Pythia8::Event& event, int ix, int& bcflag) {
  // see if found W or top
  int id = event[ix].id();
  int ida = abs(id);
  if (ida==24 || ida==6) return id;
  // check b/c origin
  if (bcflag<5 && (ida/100==5 || ida/1000==5)) bcflag = 5;
  else if (bcflag<4 && (ida/100==4 || ida/1000==4)) bcflag = 4;
  // keep digging
  int mother1 = event[ix].mother1();
  int mother2 = event[ix].mother2();
  if (mother1==0) return 0;
  if (mother2==0 || mother2==mother1 || mother2<mother1) return trace_origin(event, mother1, bcflag);
  for (int j = mother1; j<=mother2; ++j) {
    // only trace quarks
    int ida = abs(event[j].id());
    if (ida>=1 && ida<=5) {
      int id = trace_origin(event, j, bcflag);
      if (abs(id)==24 || abs(id)==6) return id;
    }
  }
  // nothing good
  return 0;
}
