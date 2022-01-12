'''
Created on Jun 10, 2020

@author: banyait
'''
import json
 
# filein="result_vectorization.json"
# fileout="result_vectorization.converted.json"


 
def print_geojson(filein,fileout):
    
    protogeom={
        "type":"FeatureCollection",
        "name":"small_field",
    #    "crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features":[
        ]
    }

    with open(filein) as fin:
        polys=json.load(fin)
        for ipoly in polys:
            protogeom['features'].append(
                {
                    "type":"Feature",
                    "properties":{},
                    "geometry": ipoly 
                }
            )

    with open(fileout,'w') as fout:
        json.dump(protogeom, fout, indent=2)
        
