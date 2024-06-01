#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Update the location of a adrespositie and and add a terrein koppeling using a shapeFile

import os, sys, codecs, datetime, argparse
import osgeo.ogr as ogr
from pyspatialite import dbapi2 as sqlite3 #import sqlite3

def updateTerrein(cur, TERREINOBJECTID , HUISNUMMERID):
    cur.execute("""INSERT INTO TERREINOBJECT_HUISNUMMER_RELATIES
          (ID, TERREINOBJECTID , HUISNUMMERID, BEGINDATUM, BEGINORGANISATIE, BEGINBEWERKING, BEGINTIJD )
          VALUES ( (SELECT MAX("ID")+ 1 FROM "TERREINOBJECT_HUISNUMMER_RELATIES"),
          ?, ?, date('now'), 1, 1, strftime('%Y-%m-%dT%H:%M:%S','now')) ;""", (TERREINOBJECTID , HUISNUMMERID))

def updateAdresPosistie(cur, X , Y , herkomst, ADRESID ):
    'herkomst: 2= perceel, 3= gebouw'
    cur.execute("""UPDATE ADRESPOSITIES
                 SET X=?, Y=?, BEGINORGANISATIE=1, BEGINBEWERKING=3, BEGINTIJD=strftime('%Y-%m-%dT%H:%M:%S','now'),
                 HERKOMSTADRESPOSITIE=? WHERE ID=? ;""", (X, Y, herkomst, ADRESID))

def removeDoubleTerreinKoppeling(cur):
    #joined twice or more
    cmd1 = """DELETE FROM TERREINOBJECT_HUISNUMMER_RELATIES
           WHERE  BEGINTIJD IS NULL OR BEGINTIJD > DATE('now', '-1 day')
           AND EXISTS (
                    SELECT t2.terreinobjectid , t2.huisnummerid , t2.begindatum
                    FROM TERREINOBJECT_HUISNUMMER_RELATIES t2
                    WHERE eindtijd IS NULL
                    AND TERREINOBJECT_HUISNUMMER_RELATIES.terreinobjectid = t2.terreinobjectid
                    AND TERREINOBJECT_HUISNUMMER_RELATIES.huisnummerid = t2.huisnummerid
                    AND TERREINOBJECT_HUISNUMMER_RELATIES.begindatum = t2.begindatum

                    GROUP BY  t2.terreinobjectid,  t2.huisnummerid,  t2.begindatum
                    HAVING COUNT(*) > 1
                    AND MAX(t2.ID) <> TERREINOBJECT_HUISNUMMER_RELATIES.ID
            ); """
    #joined to a adres with an enddate
    cmd2 = """DELETE FROM TERREINOBJECT_HUISNUMMER_RELATIES
            WHERE BEGINTIJD IS NULL OR BEGINTIJD > DATE('now', '-1 day')
            AND EXISTS (
                SELECT einddatum FROM HUISNUMMERS
                WHERE
                ID = TERREINOBJECT_HUISNUMMER_RELATIES.huisnummerid
                AND IFNULL(einddatum, '9999-01-01') <
                IFNULL(TERREINOBJECT_HUISNUMMER_RELATIES.einddatum, '9999-01-01')
                );"""
    cur.execute( cmd1 )
    cur.execute( cmd2 )

def readShape( shapefile, xgrabDB , koppelType=3 ):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shapefile, 0)
    layer = dataSource.GetLayer(0)

    con = sqlite3.connect( xgrabDB )
    with con:
        cur = con.cursor()
        
        cur.execute( "CREATE INDEX IF NOT EXISTS adresID_index ON ADRESPOSITIES (ID);" )
        con.commit()

        for feature in layer:
            geom = feature.GetGeometryRef()
            adresID = feature.GetFieldAsInteger("ADRESID")
            terreinID = feature.GetFieldAsInteger("TERREINOBJ")
            huisnrID = feature.GetFieldAsInteger("HUISNR_ID")
            X, Y = ( geom.GetX() , geom.GetY() )
            updateAdresPosistie(cur, X, Y, koppelType, adresID)
            updateTerrein(cur, terreinID , huisnrID)

            removeDoubleTerreinKoppeling(cur)

        con.commit()

    if con:
        con.close()

def main():
    readShape(args.shapeFile, args.xgrabDB, int( args.koppelType) )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='update adresposities in a xgrab-db using a shapefile, requires spatialite and gdal-python')
    parser.add_argument('xgrabDB', help='The input database (.sqlite)' )
    parser.add_argument('shapeFile', help='The path to the shapefile, has a TERREINOBJ, HUISNR_ID and adresID')
    parser.add_argument('koppelType', help='2 for parcel and 3 for building', default='3')                 
    args = parser.parse_args()

    main()
