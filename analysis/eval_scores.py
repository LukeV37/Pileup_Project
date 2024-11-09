import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import pickle
import sys

import ROOT

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

in_file = str(sys.argv[1])
Efrac_model = str(sys.argv[2])
Mfrac_model = str(sys.argv[3])
in_baseline = str(sys.argv[4])
in_pred = str(sys.argv[5])
in_truth = str(sys.argv[6])
out_file = str(sys.argv[7])

print("Loading sample into memory...")
with uproot.open(in_file+":fastjet") as f:
    jet_pt = f["jet_pt"].array()
    jet_eta = f["jet_eta"].array()
    jet_phi = f["jet_phi"].array()
    jet_m = f["jet_m"].array()
    jet_true_Efrac = f["jet_true_Efrac"].array()
    jet_true_Mfrac = f["jet_true_Mfrac"].array()
    
    trk_pt = f["trk_jet_pT"].array()
    trk_eta = f["trk_jet_eta"].array()
    trk_phi = f["trk_jet_phi"].array()
    trk_q = f["trk_jet_q"].array()
    trk_d0 = f["trk_jet_d0"].array()
    trk_z0 = f["trk_jet_z0"].array()
    trk_label = f["trk_jet_label"].array()
    
    event_no = ak.zeros_like(jet_pt).to_list()
    jet_no = ak.zeros_like(jet_pt).to_list()

# Calculate Event Num and Jet Num
for event in range(len(jet_pt)):
    for jet in range(len(jet_pt[event])):
        event_no[event][jet] = event
        jet_no[event][jet] = jet
event_no = ak.Array(event_no)
jet_no = ak.Array(jet_no)

print("Joining jet features...")
jet_feat_list = [jet_pt,jet_eta,jet_phi,jet_m,jet_true_Efrac,jet_true_Mfrac,event_no,jet_no]
jet_feat_list = [x[:,:,np.newaxis] for x in jet_feat_list]
jet_feats = ak.concatenate(jet_feat_list, axis=2)
print("\tNum Events: ", len(jet_feats))
print("\tNum Jets in first event: ", len(jet_feats[0]))
print("\tNum Jet Features: ", len(jet_feats[0][0]))

print("Joining track features...")
trk_feat_list = [trk_pt,trk_eta,trk_phi,trk_q,trk_d0,trk_z0,trk_label]
trk_feat_list = [x[:,:,:,np.newaxis] for x in trk_feat_list]
trk_feats = ak.concatenate(trk_feat_list, axis=3)
print("\tNum Events: ", len(trk_feats))
print("\tNum Jets in first event: ", len(trk_feats[0]))
print("\tNum Tracks in first event first jet: ", len(trk_feats[0][0]))
print("\tNum Tracks features: ", len(trk_feats[0][0][0]))

print("Applying Cuts...")
# Apply Jet cuts
jet_mask = abs(jet_feats[:,:,1])<4
selected_jets = jet_feats[jet_mask]
selected_tracks = trk_feats[jet_mask]

# Apply Track cuts
trk_q_cut = selected_tracks[:,:,:,3]!=0            # Skip neutral particles
trk_eta_cut = abs(selected_tracks[:,:,:,1])<4.5    # Skip forward region
trk_pt_cut = selected_tracks[:,:,:,0]>0.4          # 400MeV Cut
mask = trk_q_cut & trk_eta_cut & trk_pt_cut
selected_tracks = selected_tracks[mask]

# Skip trackless jets!
trackless_jets_mask = (ak.num(selected_tracks, axis=2)!=0)
selected_jets = selected_jets[trackless_jets_mask]
selected_tracks = selected_tracks[trackless_jets_mask]

print("Normalizing Jet Features...")

var_list = ['pT','Eta','Phi','Mass']

# Normalize Jet Features
norm_list = []
for i in range(len(var_list)):
    feat = selected_jets[:,:,i]
    mean = ak.mean(feat)
    std = ak.std(feat)
    norm = (feat-mean)/std
    norm_list.append(norm)
    
# Append Labels
norm_list.append(selected_jets[:,:,-4]) # Append Efrac
norm_list.append(selected_jets[:,:,-3]) # Append Mfrac
norm_list.append(selected_jets[:,:,-2]) # Append Event no
norm_list.append(selected_jets[:,:,-1]) # Append Jet no

# Combine features
Norm_list = [x[:,:,np.newaxis] for x in norm_list]
selected_jets = ak.concatenate(Norm_list, axis=2)

print("Normalizing Track Features...")
var_list = ['pT','Eta','Phi','Charge', 'd0', 'z0']

norm_list = []
for i in range(len(var_list)):
    feat = selected_tracks[:,:,:,i]
    mean = ak.mean(feat)
    std = ak.std(feat)
    norm = (feat-mean)/std
    norm_list.append(norm)
# Add label
norm_list.append(selected_tracks[:,:,:,-1])
    
# Combine features
Norm_list = [x[:,:,:,np.newaxis] for x in norm_list]
selected_tracks = ak.concatenate(Norm_list, axis=3)


print("Padding Tracks to common length...")
all_tracks = ak.flatten(selected_tracks, axis=2)
num_events = len(selected_jets)
Event_Data = []
Event_Labels = []
for event in range(num_events):
    if event%1==0:
        print("\tProcessing: ", event, " / ", num_events, end="\r")
    jets = torch.Tensor(selected_jets[event,:,:])
    num_trks = ak.num(selected_tracks[event], axis=1)
    max_num_trks = ak.max(num_trks)
    trk_list = []
    num_jets = len(selected_jets[event])
    for jet in range(num_jets):
        tracks = torch.Tensor(selected_tracks[event][jet,:])        
        pad = (0,0,0,max_num_trks-len(tracks))        
        tracks = F.pad(tracks,pad)
        trk_list.append(torch.unsqueeze(tracks,dim=0))
    tracks = torch.cat(trk_list,dim=0)
    # Append all data but don't include label 0:-1!
    flat_tracks = torch.Tensor(all_tracks[event][:,0:-1])
    Event_Data.append((jets[:,0:6],tracks[:,:,0:-1],flat_tracks))
    Event_Labels.append((jets[:,-2],jets[:,-1]))
print("\tProcessing: ", num_events, " / ", num_events)

class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Encoder, self).__init__()
        self.pre_norm_Q = nn.LayerNorm(embed_dim)
        self.pre_norm_K = nn.LayerNorm(embed_dim)
        self.pre_norm_V = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim,num_heads=num_heads,batch_first=True, dropout=0.25)
        self.post_norm = nn.LayerNorm(embed_dim)
        self.out = nn.Linear(embed_dim,embed_dim)
    def forward(self, Query, Key, Value):
        Query = self.pre_norm_Q(Query)
        Key = self.pre_norm_K(Key)
        Value = self.pre_norm_V(Value)
        context, weights = self.attention(Query, Key, Value)
        context = self.post_norm(context)
        latent = Query + context
        tmp = F.gelu(self.out(latent))
        latent = latent + tmp
        return latent, weights

class Model(nn.Module):  
    def __init__(self):
        super(Model, self).__init__()   
        
        self.embed_dim = 256
        self.num_heads = 8
        self.num_jet_feats = 4
        self.num_trk_feats = 6
        
        self.jet_initializer = nn.Linear(self.num_jet_feats, self.embed_dim)
        self.jet_trk_initializer = nn.Linear(self.num_trk_feats, self.embed_dim)
        self.trk_initializer = nn.Linear(self.num_trk_feats, self.embed_dim)
            
        # Track Encoder Stack
        self.trk_encoder1 = Encoder(self.embed_dim, self.num_heads)
        self.trk_encoder2 = Encoder(self.embed_dim, self.num_heads)
        self.trk_encoder3 = Encoder(self.embed_dim, self.num_heads)
        self.jet_postprocess = nn.Linear(self.embed_dim*2, self.embed_dim)
        
        # All Track Encoder Stack
        self.all_trk_encoder1 = Encoder(self.embed_dim, self.num_heads)
        self.all_trk_encoder2 = Encoder(self.embed_dim, self.num_heads)
        self.all_trk_encoder3 = Encoder(self.embed_dim, self.num_heads)
        
        # Cross Encoder Stack
        self.cross_encoder1 = Encoder(self.embed_dim, self.num_heads)
        self.cross_encoder2 = Encoder(self.embed_dim, self.num_heads)
        self.cross_encoder3 = Encoder(self.embed_dim, self.num_heads)
        
        # Jet Encoder Stack
        self.jet_encoder1 = Encoder(self.embed_dim, self.num_heads)
        self.jet_encoder2 = Encoder(self.embed_dim, self.num_heads)
        self.jet_encoder3 = Encoder(self.embed_dim, self.num_heads)

        # Regression Task
        self.regression = nn.Linear(self.embed_dim, 1)
        
    def forward(self, jets, jet_trks, trks):
        # Feature preprocessing layers
        jet_init = F.relu(self.jet_initializer(jets))
        jet_trk_init = F.relu(self.jet_trk_initializer(jet_trks))
        trk_init = F.relu(self.trk_initializer(trks))
        
        # Calculate aggregated tracks using attention
        jet_trk_embedding, trk_weights = self.trk_encoder1(jet_trk_init, jet_trk_init, jet_trk_init)
        jet_trk_embedding, trk_weights = self.trk_encoder2(jet_trk_embedding, jet_trk_embedding, jet_trk_embedding)
        jet_trk_embedding, trk_weights = self.trk_encoder3(jet_trk_embedding, jet_trk_embedding, jet_trk_embedding)
        
        # Generate meaningful jet_embedding using info from trk_aggregated  
        jet_trk_aggregated = jet_trk_embedding.sum(dim=1)
        jet_embedding = torch.cat((jet_init, jet_trk_aggregated),1)
        jet_embedding = F.relu(self.jet_postprocess(jet_embedding))
        
        # All Track Attention
        all_trk_embedding, all_trk_weights = self.all_trk_encoder1(trk_init, trk_init, trk_init)
        all_trk_embedding, all_trk_weights = self.all_trk_encoder2(all_trk_embedding, all_trk_embedding, all_trk_embedding)
        all_trk_embedding, all_trk_weights = self.all_trk_encoder3(all_trk_embedding, all_trk_embedding, all_trk_embedding)

        # Cross Attention
        jet_embedding, cross_weights = self.cross_encoder1(jet_embedding, all_trk_embedding, all_trk_embedding)
        jet_embedding, cross_weights = self.cross_encoder2(jet_embedding, all_trk_embedding, all_trk_embedding)
        jet_embedding, cross_weights = self.cross_encoder3(jet_embedding, all_trk_embedding, all_trk_embedding)
        
        # Update embeddings of jets in the contex of entire event
        jet_embedding, jet_weights = self.jet_encoder1(jet_embedding, jet_embedding, jet_embedding)
        jet_embedding, jet_weights = self.jet_encoder2(jet_embedding, jet_embedding, jet_embedding)
        jet_embedding, jet_weights = self.jet_encoder3(jet_embedding, jet_embedding, jet_embedding)
        
        # Get output
        output = F.sigmoid(self.regression(jet_embedding))
        
        return output, jet_weights, trk_weights, cross_weights


print("GPU Available: ", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

EfracNN = torch.load(Efrac_model).to(device)
MfracNN = torch.load(Mfrac_model).to(device)

Efrac = ak.ones_like(jet_pt) * -1
Efrac = Efrac.to_list()

Mfrac = ak.ones_like(jet_pt) * -1
Mfrac = Mfrac.to_list()

print("Evaluating Efrac and Mfrac")
num_events=len(Event_Data)
for event in range(num_events):
    if event%5==0:
        print("\tProcessing: ", event, " / ", num_events, end="\r")
    jets = Event_Data[event][0][:,0:4]
    jet_trks = Event_Data[event][1]
    trks = Event_Data[event][2]
    
    Efr_pred = EfracNN(jets.to(device),jet_trks.to(device),trks.to(device))[0].detach().cpu().numpy()
    Mfr_pred = MfracNN(jets.to(device),jet_trks.to(device),trks.to(device))[0].detach().cpu().numpy()
    
    for jet in range(len(Event_Data[event][0])):
        event_No = int(Event_Labels[event][0][jet].detach().numpy())
        jet_No = int(Event_Labels[event][1][jet].detach().numpy())

        Efrac[event_No][jet_No] = float(Efr_pred[jet][0])
        Mfrac[event_No][jet_No] = float(Mfr_pred[jet][0])
print("\tProcessing: ", num_events, " / ", num_events)
Efrac=ak.Array(Efrac)
Mfrac=ak.Array(Mfrac)

class Encoder2(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Encoder2, self).__init__()
        self.pre_norm_Q = nn.LayerNorm(embed_dim)
        self.pre_norm_K = nn.LayerNorm(embed_dim)
        self.pre_norm_V = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim,num_heads=num_heads,batch_first=True, dropout=0.25)
        self.post_norm = nn.LayerNorm(embed_dim)
        self.out = nn.Linear(embed_dim,embed_dim)
    def forward(self, Query, Key, Value):
        #Query = self.pre_norm_Q(Query)
        #Key = self.pre_norm_K(Key)
        #Value = self.pre_norm_V(Value)
        context, weights = self.attention(Query, Key, Value)
        #context = self.post_norm(context)
        latent = Query + context
        tmp = F.gelu(self.out(latent))
        latent = latent + tmp
        return latent, weights

class Model2(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(Model2, self).__init__()        
        self.jet_preprocess = nn.Linear(in_feats,hidden_feats)
        self.encoder1 = Encoder2(hidden_feats,num_heads=8)
        self.encoder2 = Encoder2(hidden_feats,num_heads=8)
        self.encoder3 = Encoder2(hidden_feats,num_heads=8)
        self.encoder4 = Encoder2(hidden_feats,num_heads=8)
        self.encoder5 = Encoder2(hidden_feats,num_heads=8)

        self.jet_postprocess = nn.Linear(hidden_feats,hidden_feats)
        self.jet_classifier = nn.Linear(hidden_feats,out_feats)
    def forward(self, jet_feats):
        # Preprocess Jet Feats
        jet_embedding = F.gelu(self.jet_preprocess(jet_feats))
        
        # Attention Layer + Skip Connection + Post-Process
        jet_embedding, jet_weights = self.encoder1(jet_embedding,jet_embedding,jet_embedding)
        jet_embedding, jet_weights = self.encoder2(jet_embedding,jet_embedding,jet_embedding)
        jet_embedding, jet_weights = self.encoder3(jet_embedding,jet_embedding,jet_embedding)
        jet_embedding, jet_weights = self.encoder4(jet_embedding,jet_embedding,jet_embedding)
        jet_embedding, jet_weights = self.encoder5(jet_embedding,jet_embedding,jet_embedding)
      
        # Aggregate and Classify
        jet_aggregated = jet_embedding.sum(dim=0)
        jet_aggregated = F.gelu(self.jet_postprocess(jet_aggregated))
        output = F.sigmoid(self.jet_classifier(jet_aggregated))
        return output

BaselineNN = torch.load(in_baseline).to(device)
PredNN = torch.load(in_pred).to(device)
TrueNN = torch.load(in_truth).to(device)

BaselinePred = [-1 for x in range(len(jet_pt))]
PredPred = [-1 for x in range(len(jet_pt))]
TruthPred = [-1 for x in range(len(jet_pt))]

print("Evaluating Binary Classification Score")
num_events=len(Event_Data)
for event in range(num_events):
    if event%5==0:
        print("\tProcessing: ", event, " / ", num_events, end="\r")
    baseline_jets = Event_Data[event][0][:,0:4]

    # Skip discarded jets
    mask=ak.Array(Efrac[event])!=-1

    # Reshape Data
    Efr = torch.unsqueeze(torch.Tensor(Efrac[event][mask]),1)
    Mfr = torch.unsqueeze(torch.Tensor(Mfrac[event][mask]),1)

    # Concat Dataset
    pred_jets = torch.cat((Event_Data[event][0][:,0:4],Efr,Mfr),dim=1)
    truth_jets = Event_Data[event][0][:,0:6]

    BaselinePred[event] = float(BaselineNN(baseline_jets.to(device)).detach().cpu().numpy()[0])
    PredPred[event] = float(PredNN(pred_jets.to(device)).detach().cpu().numpy()[0])
    TruthPred[event] = float(TrueNN(truth_jets.to(device)).detach().cpu().numpy()[0])

print("\tProcessing: ", num_events, " / ", num_events)
BaselinePred = ak.Array(BaselinePred)
PredPred = ak.Array(PredPred)
TruthPred = ak.Array(TruthPred)

print("Saving Scores to ROOT File")
df = ak.to_rdataframe({"Efrac_Pred": Efrac,"Mfrac_Pred": Mfrac, "Baseline_Classifier": BaselinePred,"Pred_Classifier": PredPred,"Truth_Classifier": TruthPred})
df.Snapshot("jet_scores", out_file, ("Efrac_Pred","Mfrac_Pred","Baseline_Classifier","Pred_Classifier","Truth_Classifier"))
print("Done!")
