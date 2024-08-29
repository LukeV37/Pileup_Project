#include <TTree.h>
#include <TFile.h>

#include <vector>
using std::vector;

int main()
{
  // tree example
  TFile* fout = new TFile("a1.root","RECREATE");
  TTree* mytree = new TTree("mytree","");
  // vector type
  vector<int> vvar;
  mytree->Branch("vvar",&vvar);
  // vector<vector> type
  vector<vector<int> > vvvar;
  mytree->Branch("vvvar",&vvvar);

  for (int i = 0; i<10; ++i) {
    vvar = vector<int>({1,2,3});
    vvvar.clear();
    vvvar.push_back(vvar);
    mytree->Fill();
  }
  fout->Write();
  delete fout;

  return 0;
}
