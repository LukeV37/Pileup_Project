#include <fstream>
#include <iostream>
#include <string>

using std::cout;
using std::endl;
using std::ifstream;
using std::string;

double sbtf(
  double pt1, double eta1, double phi1, double m1,
  double pt2, double eta2, double phi2, double m2
) {
  // calculate invariant mass
  double px1 = pt1*cos(phi1);
  double py1 = pt1*sin(phi1);
  double pz1 = pt1*sinh(eta1);
  double e1 = sqrt(pt1*pt1 + pz1*pz1 + m1*m1);
  double px2 = pt2*cos(phi2);
  double py2 = pt2*sin(phi2);
  double pz2 = pt2*sinh(eta2);
  double e2 = sqrt(pt2*pt2 + pz2*pz2 + m2*m2);
  double x = sqrt(fabs((e1+e2)*(e1+e2)-(px1+px2)*(px1+px2)-(py1+py2)*(py1+py2)-(pz1+pz2)*(pz1+pz2)));

  // load tabulated likelihood
  static double* tf = 0;
  static int nbin = 0;
  static double xmin = 0, xmax = 0;
  if (tf==0) {
    cout << "Preprocessing Likelihood Edge Features" << endl;
    //cout << "loading tabbed function" << endl;
    ifstream fj("../include/sbfun.json");
    string s;
    getline(fj,s); // skip {
    //cout << s << endl;
    fj >> s >> nbin >> s;
    tf = new double[nbin];
    fj >> s >> xmin >> s;
    fj >> s >> xmax >> s;
    //cout << "nbin=" << nbin << ", xmin=" << xmin << ", xmax=" << xmax << endl;
    getline(fj,s); // get past end of line
    getline(fj,s); // skip tf
    //cout << "the next line should be '  \"tf\": ['" << endl << s << endl;
    for (int i = 0; i<nbin; ++i) {
      fj >> tf[i] >> s;
    }
    getline(fj,s);
    getline(fj,s);
    //cout << s << endl;
  }

  // evaluate likelihood
  int ibin = 0;
  if (x>xmin) {
    ibin = (x-xmin)/(xmax-xmin)*nbin;
    if (ibin>=nbin) ibin = nbin-1;
  }
  return tf[ibin];
}
