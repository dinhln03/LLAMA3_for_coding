#!/home/observer/miniconda2/bin/python

import numpy as N
import sys, os
import logging as L
import subprocess as S
from collections import namedtuple
from sigpyproc.Readers import FilReader as F
sys.path.append("/home/vgupta/Codes/Fake_FRBs/")
from Furby_reader import Furby_reader

class FileNotFound(Exception):
  pass

class Observation():
  def __init__(self, utc, cfg_file = "/home/vgupta/resources/observations.cfg"):
    self.utc = utc
    self.cfg_file = cfg_file
    self.read_conf()
    self.get_results_dir()
    self.get_archives_dir()
    self.is_failed = self.if_failed()
    self.read_info()
    self.processed_offline()
    self.annotation = self.read_annotation()

  def __str__(self):
    return self.utc

  def __repr__(self):
    return self.utc

  def read_annotation(self):
    afile = os.path.join(self.results_dir, "obs.txt")
    if not os.path.exists(afile):
      return None
    with open(afile, 'r') as f:
      return f.read()

  def read_conf(self):
    if not os.path.exists(self.cfg_file):
      raise Exception("Cannot find observation configuration file - {0}".format(self.cfg_file))
      #raise FileNotFound("Cannot find observation configuration file - {0}".format(self.cfg_file))
    conf_tmp = {}
    with open(self.cfg_file) as c:
      lines = c.readlines()

    for line in lines:
      if (line.startswith("#") or line == "" or line == "\n"):
        continue
      key = line.strip().split()[0].strip()
      val = line.strip().split()[1].strip()
      val = self.check_type(val)
      conf_tmp[key] = val

    tmp = namedtuple("CONF", conf_tmp.keys())
    self.conf = tmp(*conf_tmp.values()) 

  def get_results_dir(self):
    path1 = os.path.join(self.conf.results_dir, self.utc)
    path2 = os.path.join(self.conf.old_results_dir, self.utc)
    if os.path.isdir(path1):
      self.results_dir =  self.conf.results_dir
    elif os.path.isdir(path2):
      self.results_dir =  self.conf.old_results_dir
    else:
      raise IOError("Directory for UTC: {0} does not exist in any of the new or old results. Neither {1} nor {2} exists".format(self.utc, path1, path2))

  def get_archives_dir(self):
    path1 = os.path.join(self.conf.archives_dir, self.utc)
    path2 = os.path.join(self.conf.old_archives_dir, self.utc)
    if os.path.isdir(path1):
      self.archives_dir = self.conf.archives_dir
    elif os.path.isdir(path2):
      self.archives_dir = self.conf.old_archives_dir
    else:
      raise IOError("Directory for UTC: {0} does not exist in any of the new or old archives".format(self.utc))

  def processed_offline(self):
    self.offline_cand_file = os.path.join(self.archives_dir, self.utc, self.conf.offline_output_dir, self.conf.offline_output_file)
    self.processed_offline =  os.path.exists(self.offline_cand_file) and not self.is_failed
      
  def read_header(self):
    if self.is_failed:
      self.header = None
      return
    self.header_file = os.path.join(self.results_dir, self.utc, "FB", self.conf.header_file)
    if not os.path.exists(self.header_file):
      raise Exception("Header file({0}) does not exist".format(self.header_file))

    with open(self.header_file) as h:
      lines = h.readlines()
      
    hdr_tmp = {} 
    for line in lines:
      key = line.split()[0].strip()
      val = line.split()[1].strip()
      cval = self.check_type(val)
      if key.startswith("FURBY"):
        cval = str(val)
      hdr_tmp[key] = cval
        
    keys = hdr_tmp.keys()
    values = hdr_tmp.values()
    tmp = namedtuple("HEADER", keys)
    self.header = tmp(*values)
    self.tres = self.header.TSAMP * 1e-6
   
    return self.header

  def read_info(self):
    self.obs_info_file = os.path.join(self.results_dir, self.utc, "obs.info")
    if not os.path.exists(self.obs_info_file):
      raise Exception("obs.info file({0}) does not exist".format(self.obs_info_file))
      
    with open(self.obs_info_file) as h:
      lines = h.readlines()
      
    hdr_tmp = {} 
    for line in lines:
      if line.startswith("#") or line == "" or line == "\n":
        continue
      key = line.split()[0].strip()
      val = line.split()[1].strip()
      val = self.check_type(val)
      hdr_tmp[key] = val
      if key=="INT" and self.is_failed:
        val = 0
        
    keys = hdr_tmp.keys()
    values = hdr_tmp.values()
    tmp = namedtuple("INFO", keys)
    self.info = tmp(*values)
    #Getting Tobs-----------------
    filterbank_name = self.utc + ".fil"
    filterbank_file = os.path.join(self.archives_dir, self.utc, "FB/BEAM_001/", filterbank_name)
    if os.path.exists(filterbank_file):
      filt_header = F(filterbank_file).header
      self.tobs = filt_header.tobs
      if self.info.INT > self.tobs:
        self.tobs = self.info.INT
    else:
      self.tobs = self.info.INT
    #-----------------------------
    return self.info
 
  def check_type(self, val):
    try:
      ans=int(val)
      return ans
    except ValueError:
      try:    
        ans=float(val)
        return ans
      except ValueError:
        if val.lower()=="false":
          return False
        elif val.lower()=="true": 
          return True
        else:
          return val

  def if_processing(self):
    processing_file = os.path.join(self.results_dir, self.utc, "obs.processing")
    return os.path.exists(processing_file)

  def if_failed(self):
    obs_failed_file = os.path.join(self.results_dir, self.utc, "obs.failed")
    return os.path.exists(obs_failed_file)

  def read_furby_params(self):
    if self.is_failed:
      self.inj_furbys = -1
      return
    if (self.info.MB_ENABLED or self.info.CORR_ENABLED):
       self.inj_furbys = -1
    else:
      self.read_header()
      try:

         self.inj_furbys = self.header.INJECTED_FURBYS

      except AttributeError as e:

        #log.warn("Could not find INJECTED_FURBYS in the header file for UTC: {0}".format(self.utc))
        #log.warn("Assuming no furby injection happened in this observation ({0})".format(self.utc))
        self.inj_furbys = 0

      else:
        if self.inj_furbys > 0:
          self.furby_beams = self.header.FURBY_BEAMS.strip(",")
          self.furby_ids = self.header.FURBY_IDS.strip(",")
          self.furby_tstamps = self.header.FURBY_TSTAMPS.strip(",")

          #log.debug("Found: injected_furbys: {0}, furby_ids: {1}, furby_beams: {2}, furby_tstamps: {3}".format(self.inj_furbys, self.furby_ids, self.furby_beams, self.furby_tstamps))

  def split_and_filter_furby_params(self):
    if self.inj_furbys < 1:
        raise ValueError("No furbies to split")

    f_ids = N.array(self.furby_ids.split(","))
    f_beams = N.array(self.furby_beams.split(","))
    f_tstamps = N.array(self.furby_tstamps.split(","))
  
    f_ids = f_ids[N.where(f_ids!='')]
    f_beams = f_beams[N.where(f_beams!='')]
    f_tstamps = f_tstamps[N.where(f_tstamps!='')]
  
    test = N.array([len(f_ids), len(f_beams), len(f_tstamps)])
    if N.any(test-self.inj_furbys):
      raise ValueError("Incorrect number of furby params, observation should have failed")
  
    self.furbies = []
    self.dropped_furbies = []
    for i in range(self.inj_furbys):
      furby = Furby(f_ids[i], db = os.path.join(self.archives_dir, self.utc, "Furbys"))
      furby.i_beam = int(f_beams[i])
      furby.i_tstamp = float(f_tstamps[i])
      furby.calc_times()
  
      if (self.check_if_dropped(furby)):
        self.dropped_furbies.append(furby)
      else:
        self.furbies.append(furby)

  def check_if_dropped(self, furby):
    if not hasattr(furby, 'header'):
      furby.read_fheader()
    if not hasattr(furby, 'length'):
      furby.calc_times()

    if furby.i_tstamp < furby.length/2:
      return True
    if (furby.i_tstamp - furby.length/2) > self.tobs:
      return True
    
    all_furby_tstamps = N.array([float(i.i_tstamp) for i in self.furbies])
    diff = furby.i_tstamp - all_furby_tstamps
    if N.any((diff < (furby.length + 512*self.tres)) & (diff > 0)):
      return True

    return False


#----------------------------------------------------------------------------------------#


class Furby(Furby_reader):
  def __init__(self, ID, db = "/home/dada/furby_database"):
    self.ID = ID
    self.name = "furby_"+ID
    self.DB = db
    self.file = os.path.join(self.DB, self.name)
    self.i_beam = None
    self.i_tstamp = None
    self.i_snr = None

  def __repr__(self):
    return str(self.ID)

  def read_fheader(self):
    #self.header = self.read_header(self.file)
    self.read_header(self.file)

  def calc_times(self):
    log = L.getLogger("furby_manager")
    if not hasattr(self, 'header'):
      self.read_fheader()
    chw = (self.header.FTOP - self.header.FBOTTOM) / self.header.NCHAN
    f_chtop = self.header.FTOP - chw/2
    f_chmid = f_chtop - (self.header.NCHAN/2 * chw)
    f_chbottom = self.header.FBOTTOM + chw/2

    delay_to_top = 4.14881 * 1e6 * self.header.DM * ( f_chtop**(-2) - f_chmid**(-2) ) *1e-3   #in s
    delay_to_bottom = 4.14881 * 1e6 * self.header.DM * ( f_chbottom**(-2) - f_chmid**(-2) ) *1e-3   #in s

    self.s_time = self.i_tstamp + delay_to_top
    self.e_time = self.i_tstamp + delay_to_bottom
    self.c_time = self.i_tstamp

    self.length = self.header.NSAMPS * self.header.TSAMP * 1e-6

#---------------------------------------------------------------------------------------#

def list_UTCs_from(start_utc):
  #Note to someone editing this in future: Keep in mind that other scripts depend upon that fact that this function returns the list of UTCs in correctly sorted order. Do not change that, even if that costs speed. Or make sure that the scripts using this can be edited accordingly.
  start = Observation(start_utc)
  cmd = "ls -1d "+start.results_dir+"/202* | grep -A 999999 "+start_utc+" | awk -F/ '{print $5}'"
  utcs = S.Popen(cmd, shell=True, stdout=S.PIPE).communicate()[0].strip().split("\n")

  #VG: 02/05/2020 -- disabling the section below -- It doesn't work, and I don't have a quick fix either. 
  '''
  if start.results_dir == start.conf.old_results_dir:
    #Also append utcs from the new results directory
    cmd = "ls -1d "+conf.results_dir+"/20* | grep -A 999999 "+start_utc+" | awk -F/ '{print $5}'"
    utcs.extend(S.Popen(cmd, shell=True, stdout=S.PIPE).communicate()[0].strip().split("\n"))
  '''
  if len(utcs) == 0:
    raise Exception("Given start UTC ({}) not found in {}".format(start_utc, start.results_dir))

  return utcs

def list_UTCs_until(utc):
  check = Observation(utc)
  
  start_utc = get_first_UTC()
  UTCs_from_start = list_UTCs_from(start_utc)
  #Assume that list_UTCs_from() returns UTCs sorted in correct order, which it should.
  end_utc = utc
  index = N.where(UTCs_from_start == end_utc)[0]
  UTCs_until = UTCs_from_start[:index+1]
  return UTCs_until
  
def list_UTCs_after(utc):
  inclusive_utcs = list_UTCS_from(utc)
  return inclusive_utcs[1:]

def get_latest_UTC():
  cmd = "ls -1d -rt "+conf.results_dir+"/20* | tail -1 | awk -F/ '{print $5}'"
  utc = S.Popen(cmd, shell=True, stdout=S.PIPE).communcate()[0].strip()
  return utc

def get_first_UTC():
  '''
  Returns the first UTC recorded by Molonglo after the disk crash in October 2017
  '''
  return "2017-10-31-08:49:32"
