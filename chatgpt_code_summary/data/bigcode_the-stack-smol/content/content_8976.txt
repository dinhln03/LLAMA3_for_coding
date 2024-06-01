import os
from WMCore.Configuration import Configuration
from CRABClient.UserUtilities import config, getUsernameFromCRIC

config = Configuration()

config.section_("General")
config.General.requestName = '2017_tt_SL-HDAMPdown'
config.General.transferOutputs = True
config.General.transferLogs = True
config.section_("JobType")
config.JobType.allowUndistributedCMSSW = True
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'crab_PSet_2017_tt_SL-HDAMPdown.py'
config.JobType.maxMemoryMB = 3000
config.JobType.maxJobRuntimeMin = 1800
config.JobType.numCores = 1
config.JobType.scriptExe = 'crab_script_2017_tt_SL-HDAMPdown.sh'
config.JobType.inputFiles = ['crab_script_2017_tt_SL-HDAMPdown.py', 
                             os.path.join(os.environ['CMSSW_BASE'],'src/PhysicsTools/NanoAODTools/scripts/haddnano.py'),
                         ]
config.JobType.outputFiles = [] #['hist.root']
config.JobType.sendPythonFolder  = True
config.section_("Data")
config.Data.inputDataset = '/TTToSemiLeptonic_hdampDOWN_TuneCP5_PSweights_13TeV-powheg-pythia8/RunIIFall17NanoAODv7-PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/NANOAODSIM'
config.Data.inputDBS = 'global'
config.Data.splitting = 'FileBased'
if config.Data.splitting == 'FileBased':
    config.Data.unitsPerJob = 1
    # config.Data.totalUnits = $TOTAL_UNITS
# config.Data.userInputFiles = []

# config.Data.outLFNDirBase = '/store/user/{}/NoveCampaign/{}'.format(getUsernameFromCRIC(), "2017")
config.Data.outLFNDirBase = '/store/group/fourtop/NoveCampaign/{}'.format("2017")
config.Data.publication = True
config.Data.outputDatasetTag = 'NoveCampaign'
config.section_("Site")
config.Site.storageSite = 'T2_BE_IIHE'

