import os

__author__ = "Aaron Koeppel"
__version__ = 1.0
 
def xmlMarkup(games, team_ab, team_name, team_record):
   '''Markup the RSS feed using the data obtained.

   :param games: list of games that the team played this season
   :type games: list of GameData
   :param team_ab: the team's abbreviated name
   :type team_ab: string
   :param team_name: the team's name
   :type team_name: string'''
   
   file_name = team_ab + "_feed.xml"
   
   '''Used code from http://stackoverflow.com/questions/7935972/
   writing-to-a-new-directory-in-python-without-changing-directory'''
   
   script_dir = os.path.dirname(os.path.abspath(__file__))
   dest_dir = os.path.join(script_dir, "feeds", team_ab)
   
   try:
      os.makedirs(dest_dir)
   except OSError:
      pass

   path = os.path.join(dest_dir, file_name)
   
   with open(path, 'w') as xml:
      xml.write('<?xml version="1.0" encoding="UTF-8" ?>\n')
      xml.write("<rss version='2.0'>\n")
      xml.write("<channel>\n")
      xml.write("<title>%s - %s</title>\n" % (team_name, team_record))
      xml.write("<description>Latest %s scores</description>\n" % team_name)
      xml.write("<link>http://espn.go.com/nhl/team/schedule/_/name/%s</link>\n"
                % team_ab)
   
      for game in games:
         xml.write("<item>\n")
         xml.write("<title>%s</title>\n" % game.headline)
         xml.write("<link>%s</link>\n" % game.link)
         xml.write("</item>\n")
      xml.write("</channel>\n</rss>")
   xml.close()