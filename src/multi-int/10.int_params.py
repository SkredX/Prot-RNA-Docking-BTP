#!/usr/bin/python3
#  int_params.py
#  
#  Copyright 2016 NITHIN C <snape@Gryffindor>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
from sys import argv, exit
from math import sqrt

if len(argv)!=2 and len(argv)!=4:
	print ("Incorrect useage \nCorrect Useage is:\nint_params.py  Complex.int\nOR\nint_params.py  Complex.int Subunit1.int Subunit2.int");
	exit(1);

def calculate(fname):
	total_bsa = polar_bsa = 0.0;
	count = buried_count = 0.0;
	coordinates = [];
	with open (fname,"r") as intf:
		for line in intf:
			if line.startswith("ATOM"):
				atom = line[13:16];
				residue = line[17:20];
				chain = line[21];
				res_num = int(line[22:26]);
				coordinates.append([float(line[30:38]),float(line[38:46]),float(line[46:54])]);
				sasa_free = float(line[54:60]);
				sasa_complx = float(line[60:66]);
				bsa = float(line[66:72]);
				total_bsa += bsa;
				count += 1;
				if atom[0] == "C": polar_bsa += bsa;
				if sasa_complx == 0.0 :  buried_count += 1;
	ld = 0.0;
	for i in range(0,len(coordinates)):
		for j in range(0,len(coordinates)):
			dist = 0.0;
			if not i == j:
				for k in {0,1,2}: dist += (coordinates[i][k]-coordinates[j][k])**2;
				dist = sqrt(dist);
				if dist <= 12.0 : ld += 1;
	ld = round(ld/count,2);
	return {'bsa':round(total_bsa,2), 'fnp':round(polar_bsa/total_bsa*100,2), 'fbu':round(buried_count/count*100,2),'ld':ld};

result = calculate(argv[1]);
if (len(argv) == 2): print ("PDB id",argv[1][:4],"bsa",result['bsa'],"fnp",result['fnp'],"fbu",result['fbu'],"ld",result['ld'],sep="\t");
if (len(argv) == 4):
	result1 = calculate (argv[2]);
	result2 = calculate (argv[3]);
	print ("PDB id",argv[1][:4],"bsa",result['bsa'],result1['bsa'],result2['bsa'],"fnp",
	result['fnp'],result1['fnp'],result2['fnp'],"fbu",result['fbu'],
	result1['fbu'],result2['fbu'],"ld",result['ld'],result1['ld'],
	result2['ld'],sep="\t");

if (len(argv) == 5):
	result1 = calculate (argv[2]);
	result2 = calculate (argv[3]);
	result3 = calculate (argv[4]);
	print ("PDB id",argv[1][:4],"bsa",result['bsa'],result1['bsa'],result2['bsa'],result3['bsa'],"fnp",
	result['fnp'],result1['fnp'],result2['fnp'],result3['fnp'],"fbu",result['fbu'],
	result1['fbu'],result2['fbu'],result3['fbu'],"ld",result['ld'],result1['ld'],
	result2['ld'],result3['ld'],sep="\t");
