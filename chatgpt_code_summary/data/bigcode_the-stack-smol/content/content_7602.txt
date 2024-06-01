id=732
f = open('sumo_script_pokes', 'r')
for lines in f:
  print 'INSERT INTO pokemon_species_names (pokemon_species_id,local_language_id,name) VALUES('+str(id)+',7,"'+lines.strip('\n\r')+'");'
  print 'INSERT INTO pokemon_species_names (pokemon_species_id,local_language_id,name) VALUES('+str(id)+',9,"'+lines.strip('\n\r')+'");'
  id+=1
f = open('sumo_script_items', 'r')
id=763
for lines in f:
  print 'INSERT INTO item_names (item_id,local_language_id,name) VALUES('+str(id)+',7,"'+lines.strip('\n\r')+'");'
  print 'INSERT INTO item_names (item_id,local_language_id,name) VALUES('+str(id)+',9,"'+lines.strip('\n\r')+'");'
  id+=1
id=189
f = open('sumo_script_abilities', 'r')
for lines in f:
  print 'INSERT INTO ability_names (ability_id,local_language_id,name) VALUES('+str(id)+',7,"'+lines.strip('\n\r')+'");'
  print 'INSERT INTO ability_names (ability_id,local_language_id,name) VALUES('+str(id)+',9,"'+lines.strip('\n\r')+'");'
  id+=1
id=622
f = open('sumo_script_moves', 'r')
for lines in f:
  print 'INSERT INTO move_names (move_id,local_language_id,name) VALUES('+str(id)+',7,"'+lines.strip('\n\r')+'");'
  print 'INSERT INTO move_names (move_id,local_language_id,name) VALUES('+str(id)+',9,"'+lines.strip('\n\r')+'");'
  id+=1
