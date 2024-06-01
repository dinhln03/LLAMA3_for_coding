# -*- coding: utf-8 -*-
#BEGIN_HEADER
# The header block is where all import statements should live
from __future__ import print_function
import os
import re
import uuid
import requests
import json
import psutil
import subprocess
import numpy as np
import yaml
import time
from pprint import pformat

from installed_clients.WorkspaceClient import Workspace
from installed_clients.ReadsUtilsClient import ReadsUtils  # @IgnorePep8
from installed_clients.baseclient import ServerError
from installed_clients.AssemblyUtilClient import AssemblyUtil
from installed_clients.KBaseReportClient import KBaseReport
from installed_clients.kb_quastClient import kb_quast
from installed_clients.kb_ea_utilsClient import kb_ea_utils

from kb_SPAdes.utils.spades_assembler import SPAdesAssembler


class ShockException(Exception):
    pass

#END_HEADER


class kb_SPAdes:
    '''
    Module Name:
    kb_SPAdes

    Module Description:
    A KBase module: kb_SPAdes
A wrapper for the SPAdes assembler with hybrid features supported.
http://bioinf.spbau.ru/spades

Always runs in careful mode.
Runs 3 threads / CPU.
Maximum memory use is set to available memory - 1G.
Autodetection is used for the PHRED quality offset and k-mer sizes.
A coverage cutoff is not specified.
    '''

    ######## WARNING FOR GEVENT USERS ####### noqa
    # Since asynchronous IO can lead to methods - even the same method -
    # interrupting each other, you must be *very* careful when using global
    # state. A method could easily clobber the state set by another while
    # the latter method is running.
    ######################################### noqa
    VERSION = "1.2.0"
    GIT_URL = "https://github.com/qzzhang/kb_SPAdes"
    GIT_COMMIT_HASH = "5b7e88d6993728abc26c93cfef780ee7feb16c63"

    #BEGIN_CLASS_HEADER
    # Class variables and functions can be defined in this block
    DISABLE_SPADES_OUTPUT = False  # should be False in production

    PARAM_IN_WS = 'workspace_name'
    PARAM_IN_LIB = 'read_libraries'
    PARAM_IN_CS_NAME = 'output_contigset_name'
    PARAM_IN_DNA_SOURCE = 'dna_source'
    PARAM_IN_SINGLE_CELL = 'single_cell'
    PARAM_IN_METAGENOME = 'metagenomic'
    PARAM_IN_PLASMID = 'plasmid'
    PARAM_IN_MIN_CONTIG_LENGTH = 'min_contig_length'
    PARAM_IN_KMER_SIZES = 'kmer_sizes'
    PARAM_IN_SKIP_ERR_CORRECT = 'skip_error_correction'

    INVALID_WS_OBJ_NAME_RE = re.compile('[^\\w\\|._-]')
    INVALID_WS_NAME_RE = re.compile('[^\\w:._-]')

    THREADS_PER_CORE = 3
    MAX_THREADS = 64  # per email thread with Anton Korobeynikov
    MAX_THREADS_META = 128  # Increase threads for metagenomic assemblies
    MEMORY_OFFSET_GB = 1  # 1GB
    MIN_MEMORY_GB = 5
    MAX_MEMORY_GB_SPADES = 500
    MAX_MEMORY_GB_META_SPADES = 1000
    GB = 1000000000

    URL_WS = 'workspace-url'
    URL_SHOCK = 'shock-url'
    URL_KB_END = 'kbase-endpoint'

    TRUE = 'true'
    FALSE = 'false'

    def log(self, message, prefix_newline=False):
        print(('\n' if prefix_newline else '') +
              str(time.time()) + ': ' + str(message))

    def check_shock_response(self, response, errtxt):
        if not response.ok:
            try:
                err = json.loads(response.content)['error'][0]
            except:
                # this means shock is down or not responding.
                self.log("Couldn't parse response error content from Shock: " +
                         response.content)
                response.raise_for_status()
            raise ShockException(errtxt + str(err))

    # Helper script borrowed from the transform service, logger removed
    def upload_file_to_shock(self, file_path, token):
        """
        Use HTTP multi-part POST to save a file to a SHOCK instance.
        """

        if token is None:
            raise Exception("Authentication token required!")

        header = {'Authorization': "Oauth {0}".format(token)}

        if file_path is None:
            raise Exception("No file given for upload to SHOCK!")

        with open(os.path.abspath(file_path), 'rb') as data_file:
            files = {'upload': data_file}
            response = requests.post(
                self.shockURL + '/node', headers=header, files=files,
                stream=True, allow_redirects=True)
        self.check_shock_response(
            response, ('Error trying to upload contig FASTA file {} to Shock: '
                       ).format(file_path))
        return response.json()['data']

    # spades is configured with yaml
    #
    def generate_spades_yaml(self, reads_data):
        left = []  # fwd in fr orientation
        right = []  # rev
        single = []  # single end reads
        pacbio = []  # pacbio CLR reads (for pacbio CCS use -s option.)
        interlaced = []
        illumina_present = 0
        iontorrent_present = 0
        for read in reads_data:
            seq_tech = read['seq_tech']
            if seq_tech == "PacBio CLR":
                pacbio.append(read['fwd_file'])
            elif read['type'] == "paired":
                if 'rev_file' in read and read['rev_file']:
                    left.append(read['fwd_file'])
                    right.append(read['rev_file'])
                else:
                    interlaced.append(read['fwd_file'])
            elif read['type'] == "single":
                single.append(read['fwd_file'])

            if seq_tech == "IonTorrent":
                iontorrent_present = 1
            elif seq_tech == "Illumina":
                illumina_present = 1

        if (illumina_present == 1 and iontorrent_present == 1):
            raise ValueError('Both IonTorrent and Illumina read libraries exist. ' +
                             'SPAdes can not assemble them together.')

        yml = []
        yml_index_counter = 0
        # Pacbio CLR ahs to be run with at least one single end or paired end library
        other_reads_present_for_pacbio = 0
        if left or interlaced:
            yml.append({'type': 'paired-end',
                        'orientation': 'fr'})
            if left:
                yml[yml_index_counter]['left reads'] = left
                yml[yml_index_counter]['right reads'] = right
            if interlaced:
                yml[yml_index_counter]['interlaced reads'] = interlaced
            yml_index_counter += 1
            other_reads_present_for_pacbio = 1
        if single:
            yml.append({'type': "single"})
            yml[yml_index_counter]['single reads'] = single
            yml_index_counter += 1
            other_reads_present_for_pacbio = 1
        if pacbio:
            if other_reads_present_for_pacbio == 1:
                yml.append({'type': "pacbio"})
                yml[yml_index_counter]['single reads'] = pacbio
                yml_index_counter += 1
            else:
                # RAISE AN ERROR AS PACBIO REQUIRES AT LEAST
                # ONE SINGLE OR PAIRED ENDS LIBRARY
                raise ValueError('Per SPAdes requirements : If doing PacBio CLR reads, you must ' +
                                 'also supply at least one paired end or single end reads library')
        yml_path = os.path.join(self.scratch, 'run.yaml')
        with open(yml_path, 'w') as yml_file:
            yaml.safe_dump(yml, yml_file)
        return yml_path, iontorrent_present

    def exec_spades(self, dna_source, reads_data, phred_type, kmer_sizes, skip_error_correction):
        mem = (psutil.virtual_memory().available / self.GB -
               self.MEMORY_OFFSET_GB)
        if mem < self.MIN_MEMORY_GB:
            raise ValueError(
                'Only ' + str(psutil.virtual_memory().available) +
                ' bytes of memory are available. The SPAdes wrapper will' +
                ' not run without at least ' +
                str(self.MIN_MEMORY_GB + self.MEMORY_OFFSET_GB) +
                ' gigabytes available')

        if dna_source == self.PARAM_IN_METAGENOME:
            max_mem = self.MAX_MEMORY_GB_META_SPADES
            max_threads = self.MAX_THREADS_META
        else:
            max_mem = self.MAX_MEMORY_GB_SPADES
            max_threads = self.MAX_THREADS

        threads = min(max_threads, psutil.cpu_count() * self.THREADS_PER_CORE)

        if mem > max_mem:
            mem = max_mem

        outdir = os.path.join(self.scratch, 'spades_output_dir')
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        tmpdir = os.path.join(self.scratch, 'spades_tmp_dir')
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)

        cmd = ['spades.py', '--threads', str(threads),
               '--memory', str(mem), '-o', outdir, '--tmp-dir', tmpdir]

        print("THE DNA SOURCE IS : " + str(dna_source))
        if dna_source == self.PARAM_IN_SINGLE_CELL:
            cmd += ['--sc']
        if dna_source == self.PARAM_IN_PLASMID:
            cmd += ['--plasmid']
            # The plasmid assembly can only be run on a single library
            if len(reads_data) > 1:
                raise ValueError('Plasmid assembly requires that one ' +
                                 'and only one library as input. ' +
                                 str(len(reads_data)) + ' libraries detected.')
        if dna_source == self.PARAM_IN_METAGENOME:
            cmd += ['--meta']
            # The metagenome assembly can only be run on a single library
            # The library must be paired end.
            if len(reads_data) > 1 or reads_data[0]['type'] != 'paired':
                error_msg = 'Metagenome assembly requires that one and ' + \
                            'only one paired end library as input.'
                if len(reads_data) > 1:
                    error_msg += ' ' + str(len(reads_data)) + \
                                 ' libraries detected.'
                raise ValueError(error_msg)
        else:
            cmd += ['--careful']
        cmd += ['--phred-offset', phred_type]

        if kmer_sizes is not None:
            cmd += ['-k ' + kmer_sizes]
        if skip_error_correction == 1:
            cmd += ['--only-assembler']

#        print("LENGTH OF READSDATA IN EXEC: " + str(len(reads_data)))
#        print("READS DATA: " + str(reads_data))
#        print("SPADES YAML: " + str(self.generate_spades_yaml(reads_data)))
        spades_yaml_path, iontorrent_present = self.generate_spades_yaml(reads_data)
        if iontorrent_present == 1:
            cmd += ['--iontorrent']
        cmd += ['--dataset', spades_yaml_path]
        self.log('Running SPAdes command line:')
        print("SPADES CMD:" + str(cmd))
        self.log(cmd)

        if self.DISABLE_SPADES_OUTPUT:
            with open(os.devnull, 'w') as null:
                p = subprocess.Popen(cmd, cwd=self.scratch, shell=False,
                                     stdout=null)
        else:
            p = subprocess.Popen(cmd, cwd=self.scratch, shell=False)
        retcode = p.wait()

        self.log('Return code: ' + str(retcode))
        if p.returncode != 0:
            raise ValueError('Error running SPAdes, return code: ' +
                             str(retcode) + '\n')
        return outdir

    # adapted from
    # https://github.com/kbase/transform/blob/master/plugins/scripts/convert/trns_transform_KBaseFile_AssemblyFile_to_KBaseGenomes_ContigSet.py
    # which was adapted from an early version of
    # https://github.com/kbase/transform/blob/master/plugins/scripts/upload/trns_transform_FASTA_DNA_Assembly_to_KBaseGenomes_ContigSet.py
    def load_stats(self, input_file_name):
        self.log('Starting conversion of FASTA to KBaseGenomeAnnotations.Assembly')
        self.log('Building Object.')
        if not os.path.isfile(input_file_name):
            raise Exception('The input file name {0} is not a file!'.format(
                input_file_name))
        with open(input_file_name, 'r') as input_file_handle:
            contig_id = None
            sequence_len = 0
            fasta_dict = dict()
            first_header_found = False
            # Pattern for replacing white space
            pattern = re.compile(r'\s+')
            for current_line in input_file_handle:
                if (current_line[0] == '>'):
                    # found a header line
                    # Wrap up previous fasta sequence
                    if not first_header_found:
                        first_header_found = True
                    else:
                        fasta_dict[contig_id] = sequence_len
                        sequence_len = 0
                    fasta_header = current_line.replace('>', '').strip()
                    try:
                        contig_id = fasta_header.strip().split(' ', 1)[0]
                    except:
                        contig_id = fasta_header.strip()
                else:
                    sequence_len += len(re.sub(pattern, '', current_line))
        # wrap up last fasta sequence, should really make this a method
        if not first_header_found:
            raise Exception("There are no contigs in this file")
        else:
            fasta_dict[contig_id] = sequence_len
        return fasta_dict

    def load_report(self, input_file_name, params, wsname):
        fasta_stats = self.load_stats(input_file_name)
        lengths = [fasta_stats[contig_id] for contig_id in fasta_stats]

        assembly_ref = params[self.PARAM_IN_WS] + '/' + params[self.PARAM_IN_CS_NAME]

        report = ''
        report += 'Assembly saved to: ' + assembly_ref + '\n'
        report += 'Assembled into ' + str(len(lengths)) + ' contigs.\n'
        report += 'Avg Length: ' + str(sum(lengths) / float(len(lengths))) + \
            ' bp.\n'

        # compute a simple contig length distribution
        bins = 10
        counts, edges = np.histogram(lengths, bins)  # @UndefinedVariable
        report += 'Contig Length Distribution (# of contigs -- min to max ' +\
            'basepairs):\n'
        for c in range(bins):
            report += '   ' + str(counts[c]) + '\t--\t' + str(edges[c]) +\
                ' to ' + str(edges[c + 1]) + ' bp\n'
        print('Running QUAST')
        kbq = kb_quast(self.callbackURL)
        quastret = kbq.run_QUAST({'files': [{'path': input_file_name,
                                             'label': params[self.PARAM_IN_CS_NAME]}]})
        print('Saving report')
        kbr = KBaseReport(self.callbackURL)
        report_info = kbr.create_extended_report({
            'message': report,
            'objects_created': [{'ref': assembly_ref, 'description': 'Assembled contigs'}],
            'direct_html_link_index': 0,
            'html_links': [{'shock_id': quastret['shock_id'],
                            'name': 'report.html',
                            'label': 'QUAST report'}],
            'report_object_name': 'kb_megahit_report_' + str(uuid.uuid4()),
            'workspace_name': params['workspace_name']
        })
        reportName = report_info['name']
        reportRef = report_info['ref']
        return reportName, reportRef

    def make_ref(self, object_info):
        return str(object_info[6]) + '/' + str(object_info[0]) + \
            '/' + str(object_info[4])

    def determine_unknown_phreds(self, reads,
                                 phred64_reads,
                                 phred33_reads,
                                 unknown_phred_reads,
                                 reftoname):
        print("IN UNKNOWN CHECKING")
        eautils = kb_ea_utils(self.callbackURL)
        for ref in unknown_phred_reads:
            rds = reads[ref]
            obj_name = reftoname[ref]
            files_to_check = []
            f = rds['files']
            if f['type'] == 'interleaved':
                files_to_check.append(f['fwd'])
            elif f['type'] == 'paired':
                files_to_check.append(f['fwd'])
                files_to_check.append(f['rev'])
            elif f['type'] == 'single':
                files_to_check.append(f['fwd'])
            # print("FILES TO CHECK:" + str(files_to_check))
            for file_path in files_to_check:
                ea_stats_dict = eautils.calculate_fastq_stats({'read_library_path': file_path})
                # print("EA UTILS STATS : " + str(ea_stats_dict))
                if ea_stats_dict['phred_type'] == '33':
                    phred33_reads.add(obj_name)
                elif ea_stats_dict['phred_type'] == '64':
                    phred64_reads.add(obj_name)
                else:
                    raise ValueError(('Reads object {} ({}) phred type is not of the ' +
                                      'expected value of 33 or 64. It had a phred type of ' +
                                      '{}').format(obj_name, rds, ea_stats_dict['phred_type']))
        return phred64_reads, phred33_reads

    def check_reads(self, params, reads, reftoname):

        phred64_reads, phred33_reads, unknown_phred_reads = (set() for i in range(3))

        for ref in reads:
            rds = reads[ref]
            obj_name = reftoname[ref]
            obj_ref = rds['ref']
            if rds['phred_type'] == '33':
                phred33_reads.add(obj_name)
            elif rds['phred_type'] == '64':
                phred64_reads.add(obj_name)
            else:
                unknown_phred_reads.add(ref)

            if rds['read_orientation_outward'] == self.TRUE:
                raise ValueError(
                    ('Reads object {} ({}) is marked as having outward ' +
                     'oriented reads, which SPAdes does not ' +
                     'support.').format(obj_name, obj_ref))

            # ideally types would be firm enough that we could rely on the
            # metagenomic boolean. However KBaseAssembly doesn't have the field
            # and it's optional anyway. Ideally fix those issues and then set
            # the --meta command line flag automatically based on the type

            # Dylan: removing these requirements because too much work for user to go all the way
            # back and reimport reads with "single_genome" flag set opposite.  Additionally, now
            # that "metagenomic" assembly is now an explicit App instead of an option, this check
            #  is far less necessary

#            if (rds['single_genome'] == self.TRUE and
#                    params[self.PARAM_IN_DNA_SOURCE] ==
#                    self.PARAM_IN_METAGENOME):
#                raise ValueError(
#                    ('Reads object {} ({}) is marked as containing dna from ' +
#                     'a single genome but the assembly method was specified ' +
#                     'as metagenomic').format(obj_name, obj_ref))
            if (rds['single_genome'] == self.FALSE and
                    params[self.PARAM_IN_DNA_SOURCE] !=
                    self.PARAM_IN_METAGENOME):
                raise ValueError(
                    ('Reads object {} ({}) is marked as containing ' +
                     'metagenomic data but the assembly method was not ' +
                     'specified as metagenomic').format(obj_name, obj_ref))

        # IF UNKNOWN TYPE NEED TO DETERMINE PHRED TYPE USING EAUTILS
        if len(unknown_phred_reads) > 0:
            phred64_reads, phred33_reads = \
                self.determine_unknown_phreds(reads, phred64_reads, phred33_reads,
                                              unknown_phred_reads, reftoname)
        # IF THERE ARE READS OF BOTH PHRED 33 and 64, throw an error
        if (len(phred64_reads) > 0) and (len(phred33_reads) > 0):
            raise ValueError(
                ('The set of Reads objects passed in have reads that have different ' +
                 'phred type scores. SPAdes does not support assemblies of ' +
                 'reads with different phred type scores.\nThe following read objects ' +
                 'have phred 33 scores : {}.\nThe following read objects have phred 64 ' +
                 'scores : {}').format(", ".join(phred33_reads), ", ".join(phred64_reads)))
        elif len(phred64_reads) > 0:
            return '64'
        elif len(phred33_reads) > 0:
            return '33'
        else:
            raise ValueError('The phred type of the read(s) was unable to be determined')

    def process_params(self, params):
        if (self.PARAM_IN_WS not in params or
                not params[self.PARAM_IN_WS]):
            raise ValueError(self.PARAM_IN_WS + ' parameter is required')
        if self.INVALID_WS_NAME_RE.search(params[self.PARAM_IN_WS]):
            raise ValueError('Invalid workspace name ' +
                             params[self.PARAM_IN_WS])
        if self.PARAM_IN_LIB not in params:
            raise ValueError(self.PARAM_IN_LIB + ' parameter is required')
        if type(params[self.PARAM_IN_LIB]) != list:
            raise ValueError(self.PARAM_IN_LIB + ' must be a list')
        if not params[self.PARAM_IN_LIB]:
            raise ValueError('At least one reads library must be provided')
        # for l in params[self.PARAM_IN_LIB]:
        #    print("PARAM_IN_LIB : " + str(l))
        #    if self.INVALID_WS_OBJ_NAME_RE.search(l):
        #        raise ValueError('Invalid workspace object name ' + l)
        if (self.PARAM_IN_CS_NAME not in params or
                not params[self.PARAM_IN_CS_NAME]):
            raise ValueError(self.PARAM_IN_CS_NAME + ' parameter is required')
        if self.INVALID_WS_OBJ_NAME_RE.search(params[self.PARAM_IN_CS_NAME]):
            raise ValueError('Invalid workspace object name ' +
                             params[self.PARAM_IN_CS_NAME])
        if self.PARAM_IN_DNA_SOURCE in params:
            s = params[self.PARAM_IN_DNA_SOURCE]
#            print("FOUND THE DNA SOURCE: " + str(params[self.PARAM_IN_DNA_SOURCE]))
            if s not in [self.PARAM_IN_SINGLE_CELL,
                         self.PARAM_IN_METAGENOME,
                         self.PARAM_IN_PLASMID]:
                params[self.PARAM_IN_DNA_SOURCE] = None
        else:
            params[self.PARAM_IN_DNA_SOURCE] = None
#            print("PARAMS ARE:" + str(params))
        if self.PARAM_IN_MIN_CONTIG_LENGTH in params:
            if not isinstance(params[self.PARAM_IN_MIN_CONTIG_LENGTH], int):
                raise ValueError('min_contig_length must be of type int')
        if self.PARAM_IN_KMER_SIZES in params and params[self.PARAM_IN_KMER_SIZES] is not None:
            print("KMER_SIZES: " + ",".join(str(num) for num in params[self.PARAM_IN_KMER_SIZES]))
        if self.PARAM_IN_SKIP_ERR_CORRECT in params and params[self.PARAM_IN_SKIP_ERR_CORRECT] is not None:
            print("SKIP ERR CORRECTION: " + str(params[self.PARAM_IN_SKIP_ERR_CORRECT]))

    #END_CLASS_HEADER

    # config contains contents of config file in a hash or None if it couldn't
    # be found
    def __init__(self, config):
        #BEGIN_CONSTRUCTOR
        self.cfg = config
        self.cfg['SDK_CALLBACK_URL'] = os.environ['SDK_CALLBACK_URL']
        self.cfg['KB_AUTH_TOKEN'] = os.environ['KB_AUTH_TOKEN']
        self.callbackURL = self.cfg['SDK_CALLBACK_URL']
        self.log('Callback URL: ' + self.callbackURL)
        self.workspaceURL = config[self.URL_WS]
        self.shockURL = config[self.URL_SHOCK]
        self.catalogURL = config[self.URL_KB_END] + '/catalog'
        self.scratch = os.path.abspath(config['scratch'])
        if not os.path.exists(self.scratch):
            os.makedirs(self.scratch)
        #END_CONSTRUCTOR
        pass


    def run_SPAdes(self, ctx, params):
        """
        Run SPAdes on paired end libraries
        :param params: instance of type "SPAdesParams" (Input parameters for
           running SPAdes. workspace_name - the name of the workspace from
           which to take input and store output. output_contigset_name - the
           name of the output contigset read_libraries - a list of Illumina
           PairedEndLibrary files in FASTQ or BAM format. dna_source -
           (optional) the source of the DNA used for sequencing
           'single_cell': DNA amplified from a single cell via MDA anything
           else: Standard DNA sample from multiple cells. Default value is
           None. min_contig_length - (optional) integer to filter out contigs
           with length < min_contig_length from the SPAdes output. Default
           value is 0 implying no filter. kmer_sizes - (optional) K-mer
           sizes, Default values: 33, 55, 77, 99, 127 (all values must be
           odd, less than 128 and listed in ascending order) In the absence
           of these values, K values are automatically selected.
           skip_error_correction - (optional) Assembly only (No error
           correction). By default this is disabled.) -> structure: parameter
           "workspace_name" of String, parameter "output_contigset_name" of
           String, parameter "read_libraries" of list of type
           "paired_end_lib" (The workspace object name of a PairedEndLibrary
           file, whether of the KBaseAssembly or KBaseFile type.), parameter
           "dna_source" of String, parameter "min_contig_length" of Long,
           parameter "kmer_sizes" of list of Long, parameter
           "skip_error_correction" of type "bool" (A boolean. 0 = false,
           anything else = true.)
        :returns: instance of type "SPAdesOutput" (Output parameters for
           SPAdes run. report_name - the name of the KBaseReport.Report
           workspace object. report_ref - the workspace reference of the
           report.) -> structure: parameter "report_name" of String,
           parameter "report_ref" of String
        """
        # ctx is the context object
        # return variables are: output
        #BEGIN run_SPAdes

        # A whole lot of this is adapted or outright copied from
        # https://github.com/msneddon/MEGAHIT
        self.log('Running run_SPAdes with params:\n' + pformat(params))

        token = ctx['token']

        # the reads should really be specified as a list of absolute ws refs
        # but the narrative doesn't do that yet
        self.process_params(params)

        # get absolute refs from ws
        wsname = params[self.PARAM_IN_WS]
        obj_ids = []
        for r in params[self.PARAM_IN_LIB]:
            obj_ids.append({'ref': r if '/' in r else (wsname + '/' + r)})
        ws = Workspace(self.workspaceURL, token=token)
        ws_info = ws.get_object_info_new({'objects': obj_ids})
        reads_params = []

        reftoname = {}
        for wsi, oid in zip(ws_info, obj_ids):
            ref = oid['ref']
            reads_params.append(ref)
            obj_name = wsi[1]
            reftoname[ref] = wsi[7] + '/' + obj_name

        readcli = ReadsUtils(self.callbackURL, token=ctx['token'])

        typeerr = ('Supported types: KBaseFile.SingleEndLibrary ' +
                   'KBaseFile.PairedEndLibrary ' +
                   'KBaseAssembly.SingleEndLibrary ' +
                   'KBaseAssembly.PairedEndLibrary')
        try:
            reads = readcli.download_reads({'read_libraries': reads_params,
                                            'interleaved': 'false',
                                            'gzipped': None
                                            })['files']
        except ServerError as se:
            self.log('logging stacktrace from dynamic client error')
            self.log(se.data)
            if typeerr in se.message:
                prefix = se.message.split('.')[0]
                raise ValueError(
                    prefix + '. Only the types ' +
                    'KBaseAssembly.PairedEndLibrary ' +
                    'and KBaseFile.PairedEndLibrary are supported')
            else:
                raise

        self.log('Got reads data from converter:\n' + pformat(reads))

        phred_type = self.check_reads(params, reads, reftoname)

        reads_data = []
        for ref in reads:
            reads_name = reftoname[ref]
            f = reads[ref]['files']
#            print ("REF:" + str(ref))
#            print ("READS REF:" + str(reads[ref]))
            seq_tech = reads[ref]["sequencing_tech"]
            if f['type'] == 'interleaved':
                reads_data.append({'fwd_file': f['fwd'], 'type': 'paired',
                                   'seq_tech': seq_tech})
            elif f['type'] == 'paired':
                reads_data.append({'fwd_file': f['fwd'], 'rev_file': f['rev'],
                                   'type': 'paired', 'seq_tech': seq_tech})
            elif f['type'] == 'single':
                reads_data.append({'fwd_file': f['fwd'], 'type': 'single',
                                   'seq_tech': seq_tech})
            else:
                raise ValueError('Something is very wrong with read lib' + reads_name)

        kmer_sizes = None
        if self.PARAM_IN_KMER_SIZES in params and params[self.PARAM_IN_KMER_SIZES] is not None:
            if (len(params[self.PARAM_IN_KMER_SIZES])) > 0:
                kmer_sizes = ",".join(str(num) for num in params[self.PARAM_IN_KMER_SIZES])

        skip_error_correction = 0
        if self.PARAM_IN_SKIP_ERR_CORRECT in params and params[self.PARAM_IN_SKIP_ERR_CORRECT] is not None:
            if params[self.PARAM_IN_SKIP_ERR_CORRECT] == 1:
                skip_error_correction = 1

        spades_out = self.exec_spades(params[self.PARAM_IN_DNA_SOURCE],
                                      reads_data,
                                      phred_type,
                                      kmer_sizes,
                                      skip_error_correction)

        self.log('SPAdes output dir: ' + spades_out)

        # parse the output and save back to KBase
        output_contigs = os.path.join(spades_out, 'scaffolds.fasta')

        self.log('Uploading FASTA file to Assembly')

        assemblyUtil = AssemblyUtil(self.callbackURL, token=ctx['token'], service_ver='release')

        if params.get('min_contig_length', 0) > 0:
            assemblyUtil.save_assembly_from_fasta(
                {'file': {'path': output_contigs},
                 'workspace_name': wsname,
                 'assembly_name': params[self.PARAM_IN_CS_NAME],
                 'min_contig_length': params['min_contig_length']
                 })
            # load report from scaffolds.fasta.filtered.fa
            report_name, report_ref = self.load_report(
                output_contigs + '.filtered.fa', params, wsname)
        else:
            assemblyUtil.save_assembly_from_fasta(
                {'file': {'path': output_contigs},
                 'workspace_name': wsname,
                 'assembly_name': params[self.PARAM_IN_CS_NAME]
                 })
            # load report from scaffolds.fasta
            report_name, report_ref = self.load_report(
                output_contigs, params, wsname)

        output = {'report_name': report_name,
                  'report_ref': report_ref
                  }
        #END run_SPAdes

        # At some point might do deeper type checking...
        if not isinstance(output, dict):
            raise ValueError('Method run_SPAdes return value ' +
                             'output is not type dict as required.')
        # return the results
        return [output]

    def run_HybridSPAdes(self, ctx, params):
        """
        Run HybridSPAdes on paired end libraries with PacBio CLR and Oxford Nanopore reads
        :param params: instance of type "HybridSPAdesParams" (------To run
           HybridSPAdes 3.13.0 you need at least one library of the following
           types:------ 1) Illumina paired-end/high-quality
           mate-pairs/unpaired reads 2) IonTorrent paired-end/high-quality
           mate-pairs/unpaired reads 3) PacBio CCS reads Version 3.13.0 of
           SPAdes supports paired-end reads, mate-pairs and unpaired reads.
           SPAdes can take as input several paired-end and mate-pair
           libraries simultaneously. workspace_name - the name of the
           workspace from which to take input and store output.
           output_contigset_name - the name of the output contigset
           read_libraries - a list of Illumina or IonTorrent
           paired-end/high-quality mate-pairs/unpaired reads
           long_reads_libraries - a list of PacBio, Oxford Nanopore Sanger
           reads and/or additional contigs dna_source - the source of the DNA
           used for sequencing 'single_cell': DNA amplified from a single
           cell via MDA anything else: Standard DNA sample from multiple
           cells. Default value is None. pipeline_options - a list of string
           specifying how the SPAdes pipeline should be run kmer_sizes -
           (optional) K-mer sizes, Default values: 21, 33, 55, 77, 99, 127
           (all values must be odd, less than 128 and listed in ascending
           order) In the absence of these values, K values are automatically
           selected. min_contig_length - integer to filter out contigs with
           length < min_contig_length from the HybridSPAdes output. Default
           value is 0 implying no filter. @optional dna_source @optional
           pipeline_options @optional kmer_sizes @optional min_contig_length)
           -> structure: parameter "workspace_name" of String, parameter
           "output_contigset_name" of String, parameter "reads_libraries" of
           list of type "ReadsParams" (parameter groups--define attributes
           for specifying inputs with YAML data set file (advanced) The
           following attributes are available: - orientation ("fr", "rf",
           "ff") - type ("paired-end", "mate-pairs", "hq-mate-pairs",
           "single", "pacbio", "nanopore", "sanger", "trusted-contigs",
           "untrusted-contigs") - interlaced reads (comma-separated list of
           files with interlaced reads) - left reads (comma-separated list of
           files with left reads) - right reads (comma-separated list of
           files with right reads) - single reads (comma-separated list of
           files with single reads or unpaired reads from paired library) -
           merged reads (comma-separated list of files with merged reads)) ->
           structure: parameter "lib_ref" of type "obj_ref" (An X/Y/Z style
           KBase object reference), parameter "orientation" of String,
           parameter "lib_type" of String, parameter "long_reads_libraries"
           of list of type "LongReadsParams" -> structure: parameter
           "long_reads_ref" of type "obj_ref" (An X/Y/Z style KBase object
           reference), parameter "long_reads_type" of String, parameter
           "dna_source" of String, parameter "pipeline_options" of list of
           String, parameter "kmer_sizes" of list of Long, parameter
           "min_contig_length" of Long, parameter "create_report" of type
           "bool" (A boolean. 0 = false, anything else = true.)
        :returns: instance of type "SPAdesOutput" (Output parameters for
           SPAdes run. report_name - the name of the KBaseReport.Report
           workspace object. report_ref - the workspace reference of the
           report.) -> structure: parameter "report_name" of String,
           parameter "report_ref" of String
        """
        # ctx is the context object
        # return variables are: output
        #BEGIN run_HybridSPAdes
        self.log('Running run_HybridSPAdes with params:\n{}'.format(
                 json.dumps(params, indent=1)))

        spades_assembler = SPAdesAssembler(self.cfg, ctx.provenance())

        output = spades_assembler.run_hybrid_spades(params)
        #END run_HybridSPAdes

        # At some point might do deeper type checking...
        if not isinstance(output, dict):
            raise ValueError('Method run_HybridSPAdes return value ' +
                             'output is not type dict as required.')
        # return the results
        return [output]

    def run_metaSPAdes(self, ctx, params):
        """
        Run SPAdes on paired end libraries for metagenomes
        :param params: instance of type "SPAdesParams" (Input parameters for
           running SPAdes. workspace_name - the name of the workspace from
           which to take input and store output. output_contigset_name - the
           name of the output contigset read_libraries - a list of Illumina
           PairedEndLibrary files in FASTQ or BAM format. dna_source -
           (optional) the source of the DNA used for sequencing
           'single_cell': DNA amplified from a single cell via MDA anything
           else: Standard DNA sample from multiple cells. Default value is
           None. min_contig_length - (optional) integer to filter out contigs
           with length < min_contig_length from the SPAdes output. Default
           value is 0 implying no filter. kmer_sizes - (optional) K-mer
           sizes, Default values: 33, 55, 77, 99, 127 (all values must be
           odd, less than 128 and listed in ascending order) In the absence
           of these values, K values are automatically selected.
           skip_error_correction - (optional) Assembly only (No error
           correction). By default this is disabled.) -> structure: parameter
           "workspace_name" of String, parameter "output_contigset_name" of
           String, parameter "read_libraries" of list of type
           "paired_end_lib" (The workspace object name of a PairedEndLibrary
           file, whether of the KBaseAssembly or KBaseFile type.), parameter
           "dna_source" of String, parameter "min_contig_length" of Long,
           parameter "kmer_sizes" of list of Long, parameter
           "skip_error_correction" of type "bool" (A boolean. 0 = false,
           anything else = true.)
        :returns: instance of type "SPAdesOutput" (Output parameters for
           SPAdes run. report_name - the name of the KBaseReport.Report
           workspace object. report_ref - the workspace reference of the
           report.) -> structure: parameter "report_name" of String,
           parameter "report_ref" of String
        """
        # ctx is the context object
        # return variables are: output
        #BEGIN run_metaSPAdes

        output = self.run_SPAdes(ctx,params)[0]
        #END run_metaSPAdes

        # At some point might do deeper type checking...
        if not isinstance(output, dict):
            raise ValueError('Method run_metaSPAdes return value ' +
                             'output is not type dict as required.')
        # return the results
        return [output]
    def status(self, ctx):
        #BEGIN_STATUS
        returnVal = {'state': "OK",
                     'message': "",
                     'version': self.VERSION,
                     'git_url': self.GIT_URL,
                     'git_commit_hash': self.GIT_COMMIT_HASH}
        del ctx  # shut up pep8
        #END_STATUS
        return [returnVal]
