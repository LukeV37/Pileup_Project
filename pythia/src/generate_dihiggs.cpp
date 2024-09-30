#include <iostream>
#include <vector>

#include "TFile.h"
#include "TTree.h"
#include "TRandom3.h"
#include "TString.h"
#include <TRandom.h>

#include "Pythia8/Pythia.h"

#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"

#include "include/helper_functions.h"

// Main pythia loop
int main(int argc, char *argv[])
{
    std::cout << "You have entered " << argc 
         << " arguments:" << std::endl; 
  
    // Using a while loop to iterate through arguments 
    char *settings[] = { " ", "Process: ", "Average Pileup (mu): ", "Num Events from MadGraph: ", "Min pT of Jet: " };
    int i = 0; 
    while (i < argc) { 
        std::cout << settings[i] << argv[i] 
             << std::endl; 
        i++; 
    } 

    if (argc < 4){
        std::cout << "Error! Must enter 3 arguments" << std::endl;
        std::cout << "1: Process {diHiggs|4b}" << std::endl;
        std::cout << "2: Average PU, mu, (int)" << std::endl;
        std::cout << "3: Num Events from MadGraph (int)" << std::endl;
        std::cout << "4: MinJetpT (float)" << std::endl;
        return 1;
    }

    char *process = argv[1];
    int mu = atoi(argv[2]);
    double pTmin_jet = atof(argv[3]);
    
    TString filename = TString("dataset_")+TString(argv[1])+TString("_mu")+TString(argv[2])+TString("_NumEvents")+TString(argv[3])+TString("_MinJetpT")+TString(argv[4])+TString(".root");

    // Initialiaze output ROOT file
    TFile *output = new TFile("../output/"+filename, "recreate");
    
    // Define local vars to be linked to TTree branches
    int id, status, ID, label;
    double pT, eta, phi, e, q, xProd, yProd, zProd, tProd, xDec, yDec, zDec, tDec;

    // Define tree with jets clustered using fast jet
    TTree *FastJet = new TTree("fastjet", "fastjet");
    std::vector<float> jet_pt, jet_eta, jet_phi, jet_m;
    FastJet->Branch("jet_pt", &jet_pt);
    FastJet->Branch("jet_eta", &jet_eta);
    FastJet->Branch("jet_phi", &jet_phi);
    FastJet->Branch("jet_m", &jet_m);

    std::vector<std::vector<float>> trk_jet_pT, trk_jet_eta, trk_jet_phi, trk_jet_e;
    std::vector<std::vector<float>> trk_jet_q, trk_jet_d0, trk_jet_z0;
    std::vector<std::vector<int>> trk_jet_pid, trk_jet_label, trk_jet_origin, trk_jet_bcflag;
    FastJet->Branch("trk_jet_pT", &trk_jet_pT);
    FastJet->Branch("trk_jet_eta", &trk_jet_eta);
    FastJet->Branch("trk_jet_phi", &trk_jet_phi);
    FastJet->Branch("trk_jet_e", &trk_jet_e);
    FastJet->Branch("trk_jet_q", &trk_jet_q);
    FastJet->Branch("trk_jet_d0", &trk_jet_d0);
    FastJet->Branch("trk_jet_z0", &trk_jet_z0);
    FastJet->Branch("trk_jet_pid", &trk_jet_pid);
    FastJet->Branch("trk_jet_label", &trk_jet_label);
    FastJet->Branch("trk_jet_origin", &trk_jet_origin);
    FastJet->Branch("trk_jet_bcflag", &trk_jet_bcflag);

    std::vector<float> trk_pT, trk_eta, trk_phi, trk_e;
    std::vector<float> trk_q, trk_d0, trk_z0;
    std::vector<int> trk_pid, trk_label, trk_origin, trk_bcflag;
    FastJet->Branch("trk_pT", &trk_pT);
    FastJet->Branch("trk_eta", &trk_eta);
    FastJet->Branch("trk_phi", &trk_phi);
    FastJet->Branch("trk_e", &trk_e);
    FastJet->Branch("trk_q", &trk_q);
    FastJet->Branch("trk_d0", &trk_d0);
    FastJet->Branch("trk_z0", &trk_z0);
    FastJet->Branch("trk_pid", &trk_pid);
    FastJet->Branch("trk_label", &trk_label);
    FastJet->Branch("trk_origin", &trk_origin);
    FastJet->Branch("trk_bcflag", &trk_bcflag);

    std::vector<int> jet_ntracks;
    std::vector<int> jet_track_index;
    FastJet->Branch("jet_ntracks", &jet_ntracks);
    FastJet->Branch("jet_track_index", &jet_track_index);

    // Configure HS Process
    Pythia8::Pythia pythia;

    // Initialize Les Houches Event File run. List initialization information.
    pythia.readString("Beams:frameType = 4");
    if (strcmp(process,"diHiggs")==0) pythia.readString("Beams:LHEF = ../../madgraph/output/DiHiggs/Events/run_01/unweighted_events.lhe.gz");
    if (strcmp(process,"4b")==0) pythia.readString("Beams:LHEF = ../../madgraph/output/4b/Events/run_01/unweighted_events.lhe.gz");

    // Force H->bb decay
    pythia.readString("25:onMode = off");
    pythia.readString("25:onIfAny = 5");

    // If Pythia fails to initialize, exit with error.
    if (!pythia.init()) return 1;

    // Configure PU Process
    Pythia8::Pythia pythiaPU;
    pythiaPU.readFile("./config/pileup.cmnd");
    if (mu > 0) pythiaPU.init();

    // Configure antikt_algorithm
    std::map<TString, fastjet::JetDefinition> jetDefs;
    jetDefs["Anti-#it{k_{t}} jets, #it{R} = 0.4"] = fastjet::JetDefinition(fastjet::antikt_algorithm, 0.4, fastjet::E_scheme, fastjet::Best);

    // Allow for possibility of a few faulty events.
    int nAbort = 10;
    int iAbort = 0;

    // Begin Event Loop; generate until none left in input file
    while (iAbort < nAbort) {

        // Generate events, and check whether generation failed.
        if (!pythia.next()) {
          // If failure because reached end of file then exit event loop.
          if (pythia.info.atEndOfFile()) break;
          ++iAbort;
          continue;
        }

        ID = 0;
        std::vector<float> event_trk_pT;
        std::vector<float> event_trk_eta;
        std::vector<float> event_trk_phi;
        std::vector<float> event_trk_e;
        std::vector<float> event_trk_q;
        std::vector<float> event_trk_d0;
        std::vector<float> event_trk_z0;
        std::vector<int> event_trk_pid;
        std::vector<int> event_trk_label;

        int entries = pythia.event.size();
        std::vector<Pythia8::Particle> ptcls_hs, ptcls_pu;
        std::vector<fastjet::PseudoJet> stbl_ptcls;

        // Add in hard scatter particles!
        for(int j=0;j<pythia.event.size();j++){
            auto &p = pythia.event[j];
            id = p.id();
            status = p.status();
            
            pT = p.pT();
            eta = p.eta();
            phi = p.phi();
            e = p.e();
            q = p.charge();
            xProd = p.xProd();
            yProd = p.yProd();
            zProd = p.zProd();
            tProd = p.tProd();
            xDec = p.xDec();
            yDec = p.yDec();
            zDec = p.zDec();
            tDec = p.tDec();

            label = -1; // HS Process

            double d0,z0; find_ip(pT,eta,phi,xProd,yProd,zProd,d0,z0);

            ID++;
            event_trk_pT.push_back(pT);
            event_trk_eta.push_back(eta);
            event_trk_phi.push_back(phi);
            event_trk_e.push_back(e);
            event_trk_q.push_back(q);
            event_trk_d0.push_back(d0);
            event_trk_z0.push_back(z0);
            event_trk_pid.push_back(id);
            event_trk_label.push_back(label);

            if (not p.isFinal()) continue;
            // A.X.: skip neutrinos
            if (abs(id)==12 || abs(id)==14 || abs(id)==16) continue;
                fastjet::PseudoJet fj(p.px(), p.py(), p.pz(), p.e());
                fj.set_user_index(ID);
                stbl_ptcls.push_back(fj);
                ptcls_hs.push_back(p);
        }

        // Add in pileup particles!
        int n_inel = 0;
        if (mu>0) {
            n_inel = gRandom->Poisson(mu);
            printf("Overlaying particles from %i pileup interactions!\n", n_inel);
        }
        for (int i_pu= 0; i_pu<n_inel; ++i_pu) {
            if (!pythiaPU.next()) continue;
            for (int j = 0; j < pythiaPU.event.size(); ++j) {
                auto &p = pythiaPU.event[j];
                id = p.id();
                status = p.status();

                pT = p.pT();
                eta = p.eta();
                phi = p.phi();
                e = p.e();
                q = p.charge();
                xProd = p.xProd();
                yProd = p.yProd();
                zProd = p.zProd();
                tProd = p.tProd();
                xDec = p.xDec();
                yDec = p.yDec();
                zDec = p.zDec();
                tDec = p.tDec();

                label = i_pu; // PU Process

                double d0,z0; find_ip(pT,eta,phi,xProd,yProd,zProd,d0,z0);

                ID++;
                event_trk_pT.push_back(pT);
                event_trk_eta.push_back(eta);
                event_trk_phi.push_back(phi);
                event_trk_e.push_back(e);
                event_trk_q.push_back(q);
                event_trk_d0.push_back(d0);
                event_trk_z0.push_back(z0);
                event_trk_pid.push_back(id);
                event_trk_label.push_back(label);

                if (not p.isFinal()) continue;
                // A.X.: skip neutrinos
                if (abs(id)==12 || abs(id)==14 || abs(id)==16) continue;
                        fastjet::PseudoJet fj(p.px(), p.py(), p.pz(), p.e());
                        fj.set_user_index(ID);
                        stbl_ptcls.push_back(fj);
                        ptcls_pu.push_back(p);
            }
        }

        // prepare for filling
        jet_pt.clear();
        jet_eta.clear();
        jet_phi.clear();
        jet_m.clear();

        trk_jet_pT.clear();
        trk_jet_eta.clear();
        trk_jet_phi.clear();
        trk_jet_e.clear();
        trk_jet_q.clear();
        trk_jet_d0.clear();
        trk_jet_z0.clear();
        trk_jet_pid.clear();
        trk_jet_label.clear();
        trk_jet_origin.clear();
        trk_jet_bcflag.clear();

        trk_pT.clear();
        trk_eta.clear();
        trk_phi.clear();
        trk_e.clear();
        trk_q.clear();
        trk_d0.clear();
        trk_z0.clear();
        trk_pid.clear();
        trk_label.clear();
        trk_origin.clear();
        trk_bcflag.clear();

        jet_ntracks.clear();
        jet_track_index.clear();
        int track_index = 0;

        // Cluster stable particles using anti-kt
        for (auto jetDef:jetDefs) {
            fastjet::ClusterSequence clustSeq(stbl_ptcls, jetDef.second);
            auto jets = fastjet::sorted_by_pt( clustSeq.inclusive_jets(pTmin_jet) );
            // For each jet:
            for (auto jet:jets) {
                jet_pt.push_back(jet.pt());
                jet_eta.push_back(jet.eta());
                jet_phi.push_back(jet.phi());
                jet_m.push_back(jet.m());

                std::vector<float> trk_pT_tmp, trk_eta_tmp, trk_phi_tmp, trk_e_tmp;
                std::vector<float> trk_q_tmp, trk_d0_tmp, trk_z0_tmp;
                std::vector<int> trk_pid_tmp, trk_label_tmp, trk_origin_tmp, trk_bcflag_tmp;

                // For each particle:
                jet_track_index.push_back(track_index);
                int ntracks = 0;
                for (auto trk:jet.constituents()) {
                    int ix = trk.user_index()-1;
                    trk_pT.push_back(event_trk_pT[ix]);
                    trk_eta.push_back(event_trk_eta[ix]);
                    trk_phi.push_back(event_trk_phi[ix]);
                    trk_e.push_back(event_trk_e[ix]);
                    trk_q.push_back(event_trk_q[ix]);
                    trk_d0.push_back(event_trk_d0[ix]);
                    trk_z0.push_back(event_trk_z0[ix]);
                    trk_pid.push_back(event_trk_pid[ix]);
                    trk_label.push_back(event_trk_label[ix]);
                    int bcflag = 0;
                    int origin = event_trk_label[ix]<0 ? trace_origin_higgs(pythia.event,ix,bcflag):-999;
                    trk_origin.push_back(origin);
                    trk_bcflag.push_back(bcflag);
                    ++ntracks;
                    
                    // L.V. store trks as vector<vector<>>
                    trk_pT_tmp.push_back(event_trk_pT[ix]);
                    trk_eta_tmp.push_back(event_trk_eta[ix]);
                    trk_phi_tmp.push_back(event_trk_phi[ix]);
                    trk_e_tmp.push_back(event_trk_e[ix]);
                    trk_q_tmp.push_back(event_trk_q[ix]);
                    trk_d0_tmp.push_back(event_trk_d0[ix]);
                    trk_z0_tmp.push_back(event_trk_z0[ix]);
                    trk_pid_tmp.push_back(event_trk_pid[ix]);
                    trk_label_tmp.push_back(event_trk_label[ix]);
                    trk_origin_tmp.push_back(origin);
                    trk_bcflag_tmp.push_back(bcflag);
                }
                jet_ntracks.push_back(ntracks);
                track_index += ntracks;
                
                trk_jet_pT.push_back(trk_pT_tmp);
                trk_jet_eta.push_back(trk_eta_tmp);
                trk_jet_phi.push_back(trk_phi_tmp);
                trk_jet_e.push_back(trk_e_tmp);
                trk_jet_q.push_back(trk_q_tmp);
                trk_jet_d0.push_back(trk_d0_tmp);
                trk_jet_z0.push_back(trk_z0_tmp);
                trk_jet_pid.push_back(trk_pid_tmp);
                trk_jet_label.push_back(trk_label_tmp);
                trk_jet_origin.push_back(trk_origin_tmp);
                trk_jet_bcflag.push_back(trk_bcflag_tmp);
            }
        }
        FastJet->Fill();
    }

    output->Write();
    output->Close();

    return 0;
}
