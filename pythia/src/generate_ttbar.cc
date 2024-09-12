#include <iostream>

#include "TFile.h"
#include "TTree.h"
#include "TRandom3.h"
#include "TString.h"

#include "Pythia8/Pythia.h"

#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"

#include <TRandom.h>

#include "helper_functions.h"

// Main pythia loop
int main(int argc, char *argv[])
{
    std::cout << "You have entered " << argc 
         << " arguments:" << std::endl; 
  
    // Using a while loop to iterate through arguments 
    char *settings[] = { " ", "Number of Events: ", "Average Pileup (mu): ", "Process: ", "Min pT of Jet: " };
    int i = 0; 
    while (i < argc) { 
        std::cout << settings[i] << argv[i] 
             << std::endl; 
        i++; 
    } 

    if (argc < 4){
        std::cout << "Error! Must enter 4 arguments" << std::endl;
        std::cout << "1: Num Events (int)" << std::endl;
        std::cout << "2: Average PU, mu, (int)" << std::endl;
        std::cout << "3: Process {ttbar|zprime}" << std::endl;
        std::cout << "4: MinJetpT (float)" << std::endl;
        return 1;
    }

    int nevents = atoi(argv[1]);
    int mu = atoi(argv[2]);
    char *process = argv[3];
    double pTmin_jet = atof(argv[4]);
    
    TString filename = TString("dataset_")+TString(process)+TString("_mu")+TString(argv[2])+TString("_NumEvents")+TString(argv[1])+TString("_MinJetpT")+TString(argv[4])+TString(".root");

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
    if (strcmp(process,"ttbar")==0) pythia.readFile("ttbar.cmnd");
    if (strcmp(process,"zprime")==0) pythia.readFile("zprime.cmnd");
    pythia.init();

    // Configure PU Process
    Pythia8::Pythia pythiaPU;
    pythiaPU.readFile("pileup.cmnd");
    if (mu > 0) pythiaPU.init();

    // Configure antikt_algorithm
    std::map<TString, fastjet::JetDefinition> jetDefs;
    jetDefs["Anti-#it{k_{t}} jets, #it{R} = 0.4"] = fastjet::JetDefinition(fastjet::antikt_algorithm, 0.4, fastjet::E_scheme, fastjet::Best);

    // Start main event loop
    auto &event = pythia.event;
    for(int i=0;i<nevents;i++){

        if(!pythia.next()) continue;

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
        for(int j=0;j<event.size();j++){
            auto &p = event[j];
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
                    int origin = event_trk_label[ix]<0 ? trace_origin(event,ix,bcflag):-999;
                    trk_origin.push_back(origin);
                    trk_bcflag.push_back(bcflag);
                    ++ntracks;
                }
                jet_ntracks.push_back(ntracks);
                track_index += ntracks;
            }
        }
        FastJet->Fill();
    }

    output->Write();
    output->Close();

    return 0;
}