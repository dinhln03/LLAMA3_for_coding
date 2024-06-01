from soup import soup_collector

def name_collector(spl_id, spl_type):
    soup = soup_collector(spl_id, spl_type)
    sample_info_type = soup.findAll('a')
    
    #unwanted till now START
    try:
        sample_info_name1 = sample_info_type[0].get('name').split('_')[1].strip()
        sample_info_name2 = sample_info_type[0].get('name').split('_')[2].strip()
        sample_info_name = sample_info_name1 + "_" + sample_info_name2
    except:
        sample_info_name = sample_info_type[0].get('name').split('_')[1].strip()
    return sample_info_name
    #END
#intro
