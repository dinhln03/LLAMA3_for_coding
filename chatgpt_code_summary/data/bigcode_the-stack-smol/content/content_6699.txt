# System
import json
# SBaaS
from .stage02_physiology_pairWiseTest_query import stage02_physiology_pairWiseTest_query
from SBaaS_base.sbaas_template_io import sbaas_template_io
# Resources
from io_utilities.base_importData import base_importData
from io_utilities.base_exportData import base_exportData
from ddt_python.ddt_container import ddt_container

class stage02_physiology_pairWiseTest_io(stage02_physiology_pairWiseTest_query,
                                     sbaas_template_io):
    def import_data_stage02_physiology_pairWiseTest_add(self, filename):
        '''table adds'''
        data = base_importData();
        data.read_csv(filename);
        data.format_data();
        self.add_data_stage02_physiology_pairWiseTest(data.data);
        data.clear_data();
    def export_dataStage02PhysiologyPairWiseTest_js(self,analysis_id_I,data_dir_I='tmp'):
        '''Export data for a volcano plot
        Visuals:
        1. volcano plot
        2. sample vs. sample (FC)
        3. sample vs. sample (concentration)
        4. sample vs. sample (p-value)'''
        
        #get the data for the analysis
        data_O = [];
        data_O = self.get_rows_analysisID_dataStage02PhysiologyPairWiseTest(analysis_id_I);
        # make the data parameters
        data1_keys = ['analysis_id','simulation_id_1','simulation_id_2',
                      'rxn_id','flux_units','test_description'
                    ];
        data1_nestkeys = ['analysis_id'];
        data1_keymap = {'ydata':'pvalue_negLog10',
                        'xdata':'fold_change',
                        'serieslabel':'',
                        'featureslabel':'rxn_id'};
        # make the data object
        dataobject_O = [{"data":data_O,"datakeys":data1_keys,"datanestkeys":data1_nestkeys}];
        # make the tile parameter objects
        formtileparameters_O = {'tileheader':'Filter menu','tiletype':'html','tileid':"filtermenu1",'rowid':"row1",'colid':"col1",
            'tileclass':"panel panel-default",'rowclass':"row",'colclass':"col-sm-4"};
        formparameters_O = {'htmlid':'filtermenuform1',"htmltype":'form_01',"formsubmitbuttonidtext":{'id':'submit1','text':'submit'},"formresetbuttonidtext":{'id':'reset1','text':'reset'},"formupdatebuttonidtext":{'id':'update1','text':'update'}};
        formtileparameters_O.update(formparameters_O);
        svgparameters_O = {"svgtype":'volcanoplot2d_01',
                           "svgkeymap":[data1_keymap],
                            'svgid':'svg1',
                            "svgmargin":{ 'top': 50, 'right': 50, 'bottom': 50, 'left': 50 },
                            "svgwidth":500,
                            "svgheight":350,
                            "svgx1axislabel":'Fold Change [geometric]',
                            "svgy1axislabel":'Probability [-log10(P)]'};
        svgtileparameters_O = {'tileheader':'Volcano plot','tiletype':'svg','tileid':"tile2",'rowid':"row1",'colid':"col2",
            'tileclass':"panel panel-default",'rowclass':"row",'colclass':"col-sm-8"};
        svgtileparameters_O.update(svgparameters_O);
        tableparameters_O = {"tabletype":'responsivetable_01',
                    'tableid':'table1',
                    "tableclass":"table  table-condensed table-hover",
                    "tablefilters":None,
    			    'tableformtileid':'filtermenu1','tableresetbuttonid':'reset1','tablesubmitbuttonid':'submit1'};
        tabletileparameters_O = {'tileheader':'pairWiseTest','tiletype':'table','tileid':"tile3",'rowid':"row2",'colid':"col1",
            'tileclass':"panel panel-default",'rowclass':"row",'colclass':"col-sm-12"};
        tabletileparameters_O.update(tableparameters_O);
        parametersobject_O = [formtileparameters_O,svgtileparameters_O,tabletileparameters_O];
        tile2datamap_O = {"filtermenu1":[0],"tile2":[0],"tile3":[0]};
        # dump the data to a json file
        filtermenuobject_O = None;
        ddtutilities = ddt_container(parameters_I = parametersobject_O,data_I = dataobject_O,tile2datamap_I = tile2datamap_O,filtermenu_I = filtermenuobject_O);
        if data_dir_I=='tmp':
            filename_str = self.settings['visualization_data'] + '/tmp/ddt_data.js'
        elif data_dir_I=='data_json':
            data_json_O = ddtutilities.get_allObjects_js();
            return data_json_O;
        with open(filename_str,'w') as file:
            file.write(ddtutilities.get_allObjects());
    def export_dataStage02PhysiologyPairWiseTestMetabolites_js(self,analysis_id_I,data_dir_I='tmp'):
        '''Export data for a volcano plot
        Visuals:
        1. volcano plot
        2. sample vs. sample (FC)
        3. sample vs. sample (concentration)
        4. sample vs. sample (p-value)'''
        
        #get the data for the analysis
        data_O = [];
        data_O = self.get_rows_analysisID_dataStage02PhysiologyPairWiseTestMetabolites(analysis_id_I);
        # make the data parameters
        data1_keys = ['analysis_id','simulation_id_1','simulation_id_2',
                      'met_id','flux_units','test_description'
                    ];
        data1_nestkeys = ['analysis_id'];
        data1_keymap = {'ydata':'pvalue_negLog10',
                        'xdata':'fold_change',
                        'serieslabel':'',
                        'featureslabel':'met_id'};
        # make the data object
        dataobject_O = [{"data":data_O,"datakeys":data1_keys,"datanestkeys":data1_nestkeys}];
        # make the tile parameter objects
        formtileparameters_O = {'tileheader':'Filter menu','tiletype':'html','tileid':"filtermenu1",'rowid':"row1",'colid':"col1",
            'tileclass':"panel panel-default",'rowclass':"row",'colclass':"col-sm-4"};
        formparameters_O = {'htmlid':'filtermenuform1',"htmltype":'form_01',"formsubmitbuttonidtext":{'id':'submit1','text':'submit'},"formresetbuttonidtext":{'id':'reset1','text':'reset'},"formupdatebuttonidtext":{'id':'update1','text':'update'}};
        formtileparameters_O.update(formparameters_O);
        svgparameters_O = {"svgtype":'volcanoplot2d_01',
                           "svgkeymap":[data1_keymap],
                            'svgid':'svg1',
                            "svgmargin":{ 'top': 50, 'right': 50, 'bottom': 50, 'left': 50 },
                            "svgwidth":500,
                            "svgheight":350,
                            "svgx1axislabel":'Fold Change [geometric]',
                            "svgy1axislabel":'Probability [-log10(P)]'};
        svgtileparameters_O = {'tileheader':'Volcano plot','tiletype':'svg','tileid':"tile2",'rowid':"row1",'colid':"col2",
            'tileclass':"panel panel-default",'rowclass':"row",'colclass':"col-sm-8"};
        svgtileparameters_O.update(svgparameters_O);
        tableparameters_O = {"tabletype":'responsivetable_01',
                    'tableid':'table1',
                    "tableclass":"table  table-condensed table-hover",
                    "tablefilters":None,
    			    'tableformtileid':'filtermenu1','tableresetbuttonid':'reset1','tablesubmitbuttonid':'submit1'};
        tabletileparameters_O = {'tileheader':'pairWiseTest','tiletype':'table','tileid':"tile3",'rowid':"row2",'colid':"col1",
            'tileclass':"panel panel-default",'rowclass':"row",'colclass':"col-sm-12"};
        tabletileparameters_O.update(tableparameters_O);
        parametersobject_O = [formtileparameters_O,svgtileparameters_O,tabletileparameters_O];
        tile2datamap_O = {"filtermenu1":[0],"tile2":[0],"tile3":[0]};
        # dump the data to a json file
        filtermenuobject_O = None;
        ddtutilities = ddt_container(parameters_I = parametersobject_O,data_I = dataobject_O,tile2datamap_I = tile2datamap_O,filtermenu_I = filtermenuobject_O);
        if data_dir_I=='tmp':
            filename_str = self.settings['visualization_data'] + '/tmp/ddt_data.js'
        elif data_dir_I=='data_json':
            data_json_O = ddtutilities.get_allObjects_js();
            return data_json_O;
        with open(filename_str,'w') as file:
            file.write(ddtutilities.get_allObjects());
    def export_dataStage02PhysiologyPairWiseTestSubsystems_js(self,analysis_id_I,data_dir_I='tmp'):
        '''Export data for a volcano plot
        Visuals:
        1. volcano plot
        2. sample vs. sample (FC)
        3. sample vs. sample (concentration)
        4. sample vs. sample (p-value)'''
        
        #get the data for the analysis
        data_O = [];
        data_O = self.get_rows_analysisID_dataStage02PhysiologyPairWiseTestSubsystems(analysis_id_I);
        # make the data parameters
        data1_keys = ['analysis_id','simulation_id_1','simulation_id_2',
                      'subsystem_id','flux_units','test_description'
                    ];
        data1_nestkeys = ['analysis_id'];
        data1_keymap = {'ydata':'pvalue_negLog10',
                        'xdata':'fold_change',
                        'serieslabel':'',
                        'featureslabel':'subsystem_id'};
        # make the data object
        dataobject_O = [{"data":data_O,"datakeys":data1_keys,"datanestkeys":data1_nestkeys}];
        # make the tile parameter objects
        formtileparameters_O = {'tileheader':'Filter menu','tiletype':'html','tileid':"filtermenu1",'rowid':"row1",'colid':"col1",
            'tileclass':"panel panel-default",'rowclass':"row",'colclass':"col-sm-4"};
        formparameters_O = {'htmlid':'filtermenuform1',"htmltype":'form_01',"formsubmitbuttonidtext":{'id':'submit1','text':'submit'},"formresetbuttonidtext":{'id':'reset1','text':'reset'},"formupdatebuttonidtext":{'id':'update1','text':'update'}};
        formtileparameters_O.update(formparameters_O);
        svgparameters_O = {"svgtype":'volcanoplot2d_01',
                           "svgkeymap":[data1_keymap],
                            'svgid':'svg1',
                            "svgmargin":{ 'top': 50, 'right': 50, 'bottom': 50, 'left': 50 },
                            "svgwidth":500,
                            "svgheight":350,
                            "svgx1axislabel":'Fold Change [geometric]',
                            "svgy1axislabel":'Probability [-log10(P)]'};
        svgtileparameters_O = {'tileheader':'Volcano plot','tiletype':'svg','tileid':"tile2",'rowid':"row1",'colid':"col2",
            'tileclass':"panel panel-default",'rowclass':"row",'colclass':"col-sm-8"};
        svgtileparameters_O.update(svgparameters_O);
        tableparameters_O = {"tabletype":'responsivetable_01',
                    'tableid':'table1',
                    "tableclass":"table  table-condensed table-hover",
                    "tablefilters":None,
    			    'tableformtileid':'filtermenu1','tableresetbuttonid':'reset1','tablesubmitbuttonid':'submit1'};
        tabletileparameters_O = {'tileheader':'pairWiseTest','tiletype':'table','tileid':"tile3",'rowid':"row2",'colid':"col1",
            'tileclass':"panel panel-default",'rowclass':"row",'colclass':"col-sm-12"};
        tabletileparameters_O.update(tableparameters_O);
        parametersobject_O = [formtileparameters_O,svgtileparameters_O,tabletileparameters_O];
        tile2datamap_O = {"filtermenu1":[0],"tile2":[0],"tile3":[0]};
        # dump the data to a json file
        filtermenuobject_O = None;
        ddtutilities = ddt_container(parameters_I = parametersobject_O,data_I = dataobject_O,tile2datamap_I = tile2datamap_O,filtermenu_I = filtermenuobject_O);
        if data_dir_I=='tmp':
            filename_str = self.settings['visualization_data'] + '/tmp/ddt_data.js'
        elif data_dir_I=='data_json':
            data_json_O = ddtutilities.get_allObjects_js();
            return data_json_O;
        with open(filename_str,'w') as file:
            file.write(ddtutilities.get_allObjects());