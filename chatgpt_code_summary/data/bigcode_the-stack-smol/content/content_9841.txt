import os, sys
import ROOT
from ROOT import TH1F,TH2F,TFile,TTree,TCanvas, TProfile, TNtuple, gErrorIgnoreLevel, kInfo, kWarning
from tqdm import tqdm
from particle import Particle, PDGID

tqdm_disable = False
ROOT.gErrorIgnoreLevel = kWarning;

File = TFile("/home/kshi/Zprime/Zp_data_Ntuple/WmTo3l_ZpM45.root","READ")
tree = File.Get("Ana/passedEvents")

nEntries = tree.GetEntries()

W, p, none, other = 0, 0, 0, 0
others = []

for i in tqdm(range(0, nEntries)):
  tree.GetEntry(i)

  #for j in range(0,tree.lep_matchedR03_MomMomId.size()):
  #  if abs(tree.lep_matchedR03_MomMomId[j])>=11 and abs(tree.lep_matchedR03_MomMomId[j])<=18:
  #    print "Event:" + str(tree.Event) + ", Lepton " + str(j) + " MomMomid is: " + lepton#str(tree.lep_matchedR03_MomMomId[j])

  #for j in range(0,tree.lep_matchedR03_PdgId.size()):
  #  if (abs(tree.lep_matchedR03_PdgId[j])<11 or abs(tree.lep_matchedR03_PdgId[j]>18)) and tree.lep_matchedR03_PdgId[j]!=0:
  #    print "Event:" + str(tree.Event) + " has lepton id of " + Particle.from_pdgid(tree.lep_matchedR03_PdgId[j]).name

  #for j in range(0,tree.GENlep_id.size()):
  #  if PDGID(tree.GENlep_id[j]).is_valid==False:
  #    print "Invalid lep id " + str(tree.GENlep_id[j])
  #  if PDGID(tree.GENlep_MomId[j]).is_valid==False:
  #    print "Invalid lep mom id " + str(tree.GENlep_MomId[j])
  #  if PDGID(tree.GENlep_MomMomId[j]).is_valid==False:
  #    print "Invalid lep mom mom id " + str(tree.GENlep_MomMomId[j])
  #  else:
  #    print "Event:" + str(tree.Event) + ", Lepton " + str(j) + " is a " + Particle.from_pdgid(tree.GENlep_id[j]).name + " that came from a " + Particle.from_pdgid(tree.GENlep_MomId[j]).name + " which came from a " + Particle.from_pdgid(tree.GENlep_MomMomId[j]).name

  for j in range(0,tree.lep_matchedR03_PdgId.size()):
    #if PDGID(tree.lep_matchedR03_PdgId[j]).is_valid==False:
    #  print "Invalid lep id " + str(tree.lep_matchedR03_PdgId[j])
    #if PDGID(tree.lep_matchedR03_MomId[j]).is_valid==False:
    #  print "Invalid lep mom id " + str(tree.lep_matchedR03_MomId[j])
    #if PDGID(tree.lep_matchedR03_MomMomId[j]).is_valid==False:
    #  print "Invalid lep mom mom id " + str(tree.lep_matchedR03_MomMomId[j])
    ##if tree.lep_matchedR03_PdgId[j]!=999888 and tree.lep_matchedR03_MomId!=999888 and tree.lep_matchedR03_MomMomId[j]!=999888:
    ##  print "Event:" + str(tree.Event) + ", Lepton " + str(j) + " is a " + Particle.from_pdgid(tree.lep_matchedR03_PdgId[j]).name + " that came from a " + Particle.from_pdgid(tree.lep_matchedR03_MomId[j]).name + " which came from a " + Particle.from_pdgid(tree.lep_matchedR03_MomMomId[j]).name
    #elif tree.lep_matchedR03_MomId[j]==999888:
    #  print "Event:" + str(tree.Event) + ", Lepton " + str(j) + " is a " + Particle.from_pdgid(tree.lep_matchedR03_PdgId[j]).name + " that came from a " + str(tree.lep_matchedR03_MomId[j]) + " which came from a " + Particle.from_pdgid(tree.lep_matchedR03_MomMomId[j]).name
    
    if tree.lep_matchedR03_MomId[j]==999888:
      if abs(tree.lep_matchedR03_MomMomId[j])==24:
        W+=1
      elif abs(tree.lep_matchedR03_MomMomId[j])==2212:
        p+=1
      elif abs(tree.lep_matchedR03_MomMomId[j])==0:
        none+=1
      else:
        other+=1
        others.append(tree.lep_matchedR03_MomMomId[j])

print "Sources of Z':"
print "W = " + str(W) + ", p = " + str(p) + ", none = " + str(none) + ", other = " + str(other)
for i in range(0, len(others)):
   print "Other MomMomId: " + str(others[i])
